import time, math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Any
import torch
from tqdm import tqdm
import torch.nn as nn
import json
import statistics
from pathlib import Path
import copy


@torch.no_grad()
def _flatten_like(x: torch.Tensor) -> torch.Tensor:
    """Keep batch dim, flatten the rest."""
    return x.view(x.size(0), -1)

def _infer_label_from_grad(
    original_grads: List[torch.Tensor], num_classes: int
) -> Optional[int]:
    """
    iDLG label inference: find a grad tensor whose first dim == num_classes
    (last linear's weight or bias). Sum over non-class dims and take argmin.
    Returns None if nothing suitable is found.
    """
    candidate = None
    for g in original_grads[::-1]:  # scan from last params
        if g is None:
            continue
        shape = list(g.shape)
        if len(shape) == 0:
            continue
        if shape[0] == num_classes:
            candidate = g
            break
        # also try bias vector of shape [num_classes]
        if len(shape) == 1 and shape[0] == num_classes:
            candidate = g
            break
    if candidate is None:
        return None
    vec = candidate
    while vec.dim() > 1:
        vec = vec.sum(dim=-1)
    # iDLG uses argmin
    return int(torch.argmin(vec).item())

def _project_to_feature_space(
    x: torch.Tensor,
    bounds: Optional[Union[Tuple[float, float], torch.Tensor]] = None,
    feature_specs: Optional[List[Dict]] = None,
) -> torch.Tensor:
    """
    Simple projection:
      - if bounds is (min,max): clip globally
      - if bounds is tensor [d,2]: per-feature clip
      - if feature_specs contains categorical one-hot ranges, softly
        project by normalising those slices to sum to 1.
    Feature specs format (optional):
      [{"type":"continuous","min":0.0,"max":1.0},
       {"type":"categorical","one_hot_idx":(start,end)}, ...]
    """
    x = x
    if bounds is not None:
        if isinstance(bounds, tuple):
            lo, hi = bounds
            x.data.clamp_(lo, hi)
        elif torch.is_tensor(bounds):
            # bounds: [d,2] with (lo,hi) per feature; expects x to be [1,d,...]
            flat = _flatten_like(x)
            lo, hi = bounds[:, 0], bounds[:, 1]
            flat.data = torch.max(torch.min(flat.data, hi), lo)
            x = flat.view_as(x)
    if feature_specs is not None:
        flat = _flatten_like(x)
        for spec in feature_specs:
            if spec.get("type") == "categorical" and "one_hot_idx" in spec:
                s, e = spec["one_hot_idx"]
                seg = flat[:, s:e]
                # project to probability simplex (avoid NaN)
                seg = torch.clamp(seg, min=-20.0, max=20.0)
                seg = torch.softmax(seg, dim=-1)
                flat[:, s:e] = seg
        x = flat.view_as(x)
    return x


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())
        

def run_dra_attack(
    # model: nn.Module,
    test_dataloader: Iterable,
    *,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    methods: Tuple[str, ...] = ("DLG", "iDLG"),
    max_attacks_per_loader: int = 16,
    iters: int = 300,
    lr: float = 1.0,
    optimizer_name: str = "LBFGS",  # "LBFGS" or "Adam"
    criterion: Optional[nn.Module] = None,  # defaults to CrossEntropyLoss
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    inverse_preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    # Feature constraints for tabular data:
    bounds: Optional[Union[Tuple[float, float], torch.Tensor]] = None,
    feature_specs: Optional[List[Dict]] = None,
    weight_decay_on_dummy: float = 0.0,   # small L2 prior on dummy helps stability
    early_stop_tol: float = 1e-6,
    log_every: int = 20,
    cfg = None,
) -> Dict[str, List[Dict]]:
    """
    Run DLG/iDLG gradient inversion on samples from a dataloader (batch size 1 recommended).

    Returns a dict with per-method results:
      {
        "DLG": [
          {"mse_model":..., "mse_raw":..., "label_true":..., "label_recovered":..., "loss":..., "iters":...,
           "x_rec_model": Tensor, "x_rec_raw": Tensor (if inverse_preprocess)}
        ],
        "iDLG": [ ... ],
      }

    Notes:
      - Works for any nn.Module that outputs logits for CrossEntropy.
      - For non-image/tabular, pass (bounds, feature_specs) to keep solutions realistic.
      - If your training used normalisation/standardisation, pass preprocess and inverse_preprocess.
      - If your model is not differentiable (e.g., tree/boosted models), this will be skipped.
    """
    results = {m: [] for m in methods}
    # model = model.to(device)#.eval()
    # loss_fn = criterion if criterion is not None else nn.CrossEntropyLoss().to(device)

    n_run = 0
    pbar = tqdm(total=max_attacks_per_loader, desc="DRA Attack Progress")
    for batch in test_dataloader:

        if n_run >= max_attacks_per_loader:
            break

        x_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        c_batch = batch.get("c")
        c_batch = c_batch.to(device) if c_batch is not None else None
        
        for ii in range(x_batch.size(0)):
            
            # re-instantiate model to reset weights
            model = copy.deepcopy(cfg.engine.model).to(device)
            model.apply(weights_init)
            
            if n_run >= max_attacks_per_loader:
                break

            pbar.update(1)

            # ensure batch size 1 for a clean per-sample attack
            if x_batch.dim() == 1:
                x = x_batch.unsqueeze(0)
            elif x_batch.size(0) != 1:
                # take the first sample
                x = x_batch[ii,:].unsqueeze(0)
                y = y_batch[ii,:].unsqueeze(0) if y_batch is not None else None
                c = c_batch[ii,:].unsqueeze(0) if c_batch is not None else None
            
            x = x.to(device).float()
            if preprocess is not None:
                x_model = preprocess(x)
            else:
                x_model = x

            # forward to know output size and to compute "original" grads
            x_model.requires_grad_(False)
            logits, c_hat = model(x_model)

            # if logits.dim() == 1:
            #     logits = logits.unsqueeze(0)
            num_classes = logits.size(-1)

            if y is None:
                # if label is unknown, use argmax as a stand-in for DLG's dummy initialisation
                y_true = torch.argmax(logits.detach(), dim=-1)
            else:
                y_true = y.to(device).long().view(-1)

            # compute "original" gradient dy/dtheta at (x_model, y_true)
            # loss = loss_fn(logits, y_true)

            y_hat_loss, c_hat_loss = model.filter_output_for_loss(logits, c_hat)
            loss = model.loss(
                y_hat_loss, y, c_hat_loss, c, ignore_index=-1
            ) 
            # ---
            params = [p for p in model.parameters() if p.requires_grad]
            
            # original grads
            original_dy_dx = torch.autograd.grad(
                loss, params, retain_graph=False, allow_unused=True
            )
            # replace Nones with zeros to keep shapes aligned
            original_dy_dx = [g if g is not None else torch.zeros_like(p) 
                    for g, p in zip(original_dy_dx, params)]

            # original_dy_dx = torch.autograd.grad(loss, list(model.parameters()), retain_graph=False)
            # ---

            # attack across selected methods
            for method in methods:
                # ---------- init dummy variables ----------
                dummy_x = torch.randn_like(x_model, device=device, requires_grad=True)
                if method.upper() == "DLG":
                    dummy_label = torch.randn((1, num_classes), device=device, requires_grad=True)
                    optim_params = [dummy_x, dummy_label]
                else:
                    # iDLG: infer label from gradient; fallback to y_true if inference fails
                    label_pred = _infer_label_from_grad(original_dy_dx, num_classes)
                    if label_pred is None:
                        label_pred = int(y_true.item())
                    label_pred_t = torch.tensor([label_pred], device=device, dtype=torch.long)
                    optim_params = [dummy_x]

                if optimizer_name.upper() == "LBFGS":
                    optimizer = torch.optim.LBFGS(optim_params, lr=lr)
                elif optimizer_name.upper() == "ADAM":
                    optimizer = torch.optim.Adam(optim_params, lr=lr)
                else:
                    raise ValueError("optimizer_name must be 'LBFGS' or 'Adam'.")

                losses, mses = [], []
                t_start = time.time()

                def _closure():
                    optimizer.zero_grad()
                    pred, c_hat = model(dummy_x)
                    # if pred.dim() == 1:
                    #     pred = pred.unsqueeze(0)

                    if method.upper() == "DLG":
                        # cross-entropy between soft labels and logits
                        p = torch.softmax(dummy_label, dim=-1)
                        logp = torch.log_softmax(pred, dim=-1)
                        dummy_loss = -(p * logp).sum(dim=-1).mean()
                    else:
                        # dummy_loss = loss_fn(pred, label_pred_t)
                        y_hat_loss, c_hat_loss = model.filter_output_for_loss(pred, c_hat)
                        dummy_loss = model.loss(
                            y_hat_loss, label_pred_t, c_hat_loss, c, ignore_index=-1
                        )

                    # gradient-matching term
                    # ---
                    # dummy_grads = torch.autograd.grad(dummy_loss, list(model.parameters()), create_graph=True)
                    
                    # grad_diff = 0.0
                    # for g_hat, g in zip(dummy_grads, original_dy_dx):
                    #     if g_hat is None or g is None:
                    #         continue
                    #     grad_diff = grad_diff + ((g_hat - g) ** 2).sum()
                    
                    dummy_grads = torch.autograd.grad(
                    dummy_loss, params, create_graph=True, allow_unused=True
                    )
                    # when matching
                    grad_diff = 0.0
                    for g_hat, g in zip(dummy_grads, original_dy_dx):
                        if g_hat is None: 
                            g_hat = torch.zeros_like(g)
                        grad_diff = grad_diff + ((g_hat - g) ** 2).sum()
                    
                    # ---

                    # small L2 prior on dummy_x (helps when features are unconstrained)
                    if weight_decay_on_dummy > 0.0:
                        grad_diff = grad_diff + weight_decay_on_dummy * (dummy_x ** 2).sum()

                    grad_diff.backward()
                    return grad_diff

                last_loss = None
                for it in range(iters):
                    if isinstance(optimizer, torch.optim.LBFGS):
                        loss_val = optimizer.step(_closure)
                    else:
                        # Adam: manual step
                        loss_val = _closure()
                        optimizer.step()

                    # project to plausible feature space
                    with torch.no_grad():
                        _project_to_feature_space(dummy_x, bounds=bounds, feature_specs=feature_specs)

                    # log
                    cur_loss = float(loss_val.detach())
                    with torch.no_grad():
                        mse_model = torch.mean((_flatten_like(dummy_x) - _flatten_like(x_model)) ** 2).item()
                    losses.append(cur_loss)
                    mses.append(mse_model)

                    if log_every and (it % log_every == 0 or it + 1 == iters):
                        pass  # hook for your logger if needed

                    if last_loss is not None and abs(last_loss - cur_loss) < early_stop_tol:
                        break
                    last_loss = cur_loss

                # pack result
                with torch.no_grad():
                    x_rec_model = dummy_x.detach().clone()
                    if inverse_preprocess is not None:
                        x_rec_raw = inverse_preprocess(x_rec_model)
                        mse_raw = torch.mean((_flatten_like(x_rec_raw) - _flatten_like(x)) ** 2).item()
                    else:
                        x_rec_raw, mse_raw = None, None

                    label_rec = (
                        int(torch.argmax(dummy_label.detach(), dim=-1).item())
                        if method.upper() == "DLG"
                        else int(label_pred_t.item())
                    )

                results[method].append(
                    {
                        "mse": mses[-1],
                        "mse_raw": mse_raw,
                        "label_true": int(y_true.item()),
                        "label_recovered": label_rec,
                        "loss": losses[-1],
                        "iters": len(losses),
                        "time_sec": round(time.time() - t_start, 4),
                        "x_rec_model": x_rec_model.cpu().tolist(),
                        "x_model": x_model.cpu().tolist()
                        # "x_rec_raw": x_rec_raw.cpu().tolist() if x_rec_raw is not None else None,
                    }
                )

            n_run += 1

    pbar.close()

    return results


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))

def summarize_dra_results(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    metrics: Iterable[str] = ("mse_model", "loss"),
    methods: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Read DRA results, exclude samples that are NaN/Inf/None/missing for ANY selected metric
    in EITHER method, and compute mean/std per method.

    Parameters
    ----------
    input_path : path to 'dra_results.json'
    output_path: where to save the summary (default: same dir, 'dra_results_summary.json')
    metrics    : metrics to aggregate, e.g. ('mse_model', 'loss')
    methods    : which methods to read; default uses all top-level keys in the JSON

    Returns
    -------
    summary dict (also saved to JSON if output_path is provided)
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name("dra_results_summary.json")
    else:
        output_path = Path(output_path)

    with input_path.open("r") as f:
        data = json.load(f)

    # Determine which methods to process
    if methods is None:
        methods = list(data.keys())
    else:
        methods = list(methods)

    # Sanity: ensure all selected methods exist and are lists
    method_lists: Dict[str, List[Dict[str, Any]]] = {}
    for m in methods:
        if m not in data or not isinstance(data[m], list):
            raise ValueError(f"Method '{m}' missing or not a list in {input_path}")
        method_lists[m] = data[m]

    # Align by index; use the shortest length to be safe
    n_total = min(len(lst) for lst in method_lists.values())
    if n_total == 0:
        raise ValueError("No samples found.")

    # Decide which indices to keep: keep i only if all metrics are finite for all methods
    kept_indices: List[int] = []
    excluded_indices: List[int] = []
    metrics = list(metrics)

    for i in range(n_total):
        ok = True
        for m in methods:
            entry = method_lists[m][i]
            # If the entry isn't a dict, exclude
            if not isinstance(entry, dict):
                ok = False
                break
            # Every requested metric must be finite
            for met in metrics:
                val = entry.get(met, None)
                if not _is_finite_number(val):
                    ok = False
                    break
            if not ok:
                break
        (kept_indices if ok else excluded_indices).append(i)

    # Aggregate
    per_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for m in methods:
        per_method[m] = {}
        for met in metrics:
            vals = [float(method_lists[m][i][met]) for i in kept_indices]
            if len(vals) == 0:
                mean_val, std_val = None, None
            elif len(vals) == 1:
                mean_val, std_val = vals[0], 0.0
            else:
                mean_val = statistics.fmean(vals)
                # sample std (ddof=1); use pstdev for population if you prefer
                std_val = statistics.stdev(vals)
            per_method[m][met] = {
                "mean": mean_val,
                "std": std_val,
                "n": len(vals),
            }

    summary = {
        "settings": {
            "metrics": metrics,
            "excluded_if_nan_in_either_method": True,
            "methods": methods,
        },
        "counts": {
            "n_total_aligned": n_total,
            "n_kept": len(kept_indices),
            "n_excluded": len(excluded_indices),
            "excluded_indices": excluded_indices,
        },
        "per_method": per_method,
    }

    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return summary
