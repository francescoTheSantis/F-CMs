import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from sklearn.linear_model import LogisticRegression


@dataclass
class CachedSplit:
    h: np.ndarray
    c: np.ndarray
    y: np.ndarray


@dataclass
class CAVTrainingResult:
    w: np.ndarray
    round_history: List[Dict[str, float]]


class _ConstantHead:
    def __init__(self, constant_label: int):
        self.constant_label = int(constant_label)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((x.shape[0],), self.constant_label, dtype=np.int64)


def _as_int_labels(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    return np.rint(arr).astype(np.int64, copy=False)


def _pick_positive_value(values: np.ndarray) -> Optional[int]:
    values = _as_int_labels(values)
    values = values[values != -1]
    unique = np.unique(values)
    if unique.size < 2:
        return None
    if 0 in unique and 1 in unique:
        return 1
    nonneg = unique[unique >= 0]
    if nonneg.size > 0:
        return int(nonneg[-1])
    return int(unique[-1])


def _extract_split(
    dataloader,
    backbone: Optional[torch.nn.Module],
    device: str,
) -> CachedSplit:
    h_all: List[torch.Tensor] = []
    c_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for batch in dataloader:
        x = batch["x"].to(device)
        c = batch["c"]
        y = batch["y"]

        with torch.no_grad():
            if backbone is None:
                h = x
            else:
                h = backbone(x)
            if h.dim() > 2:
                h = h.reshape(h.shape[0], -1)

        h_all.append(h.detach().cpu().float())
        c_all.append(c.detach().cpu().float())
        y_all.append(y.detach().cpu().view(-1).float())

    if len(h_all) == 0:
        return CachedSplit(
            h=np.zeros((0, 0), dtype=np.float32),
            c=np.zeros((0, 0), dtype=np.float32),
            y=np.zeros((0,), dtype=np.float32),
        )

    return CachedSplit(
        h=torch.cat(h_all, dim=0).numpy().astype(np.float32, copy=False),
        c=torch.cat(c_all, dim=0).numpy().astype(np.float32, copy=False),
        y=torch.cat(y_all, dim=0).numpy().astype(np.float32, copy=False),
    )


def _project_embeddings(embeddings: np.ndarray, cav_bank: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, cav_bank.shape[0]), dtype=np.float32)
    norms_sq = np.sum(cav_bank * cav_bank, axis=1)
    denom = np.where(norms_sq > 1e-12, norms_sq, 1.0).astype(np.float32)
    return (embeddings @ cav_bank.T) / denom[None, :]


def _compute_svm_objective(
    client_buffers: Sequence[Tuple[np.ndarray, np.ndarray, int]],
    w: np.ndarray,
    reg_lambda: float,
) -> Tuple[float, float, float]:
    total_samples = 0
    total_hinge = 0.0
    active_count = 0
    for x_local, y_local, n_local in client_buffers:
        if n_local <= 0:
            continue
        margins = y_local * (x_local @ w)
        hinge = np.maximum(0.0, 1.0 - margins)
        total_hinge += float(np.sum(hinge))
        active_count += int(np.sum(margins < 1.0))
        total_samples += n_local

    if total_samples <= 0:
        return 0.0, 0.0, 0.0

    hinge_loss = float(total_hinge / total_samples)
    reg_loss = float(reg_lambda * np.dot(w, w))
    active_fraction = float(active_count / total_samples)
    return hinge_loss, reg_loss, active_fraction


def _train_single_cav(
    client_buffers: Sequence[Tuple[np.ndarray, np.ndarray, int]],
    embedding_dim: int,
    n_rounds: int,
    local_epochs: int,
    lr: float,
    reg_lambda: float,
    log_every: int = 0,
    concept_name: Optional[str] = None,
) -> CAVTrainingResult:
    w_global = np.zeros((embedding_dim,), dtype=np.float32)
    round_history: List[Dict[str, float]] = []
    if len(client_buffers) == 0:
        return CAVTrainingResult(w=w_global, round_history=round_history)

    n_rounds = max(1, int(n_rounds))
    local_epochs = max(1, int(local_epochs))
    log_every = max(0, int(log_every))

    for rnd in range(1, n_rounds + 1):
        grad_norm = 0.0
        # Stage-A FedCBM in the paper aggregates client subgradients per round.
        for _ in range(local_epochs):
            grad_sum = np.zeros_like(w_global, dtype=np.float32)
            total_samples = 0

            for x_local, y_local, n_local in client_buffers:
                if n_local <= 0:
                    continue
                margins = y_local * (x_local @ w_global)
                active = margins < 1.0
                if np.any(active):
                    grad_sum += -np.sum(y_local[active, None] * x_local[active], axis=0).astype(np.float32, copy=False)
                total_samples += n_local

            if total_samples <= 0:
                continue

            grad_hinge = grad_sum / float(total_samples)
            grad = grad_hinge + (reg_lambda * w_global)
            w_global = (w_global - (lr * grad)).astype(np.float32, copy=False)
            grad_norm = float(np.linalg.norm(grad))

        hinge_loss, reg_loss, active_fraction = _compute_svm_objective(
            client_buffers=client_buffers,
            w=w_global,
            reg_lambda=reg_lambda,
        )
        objective = float(hinge_loss + reg_loss)
        round_stats = {
            "round": int(rnd),
            "objective": objective,
            "hinge_loss": hinge_loss,
            "reg_loss": reg_loss,
            "active_fraction": active_fraction,
            "grad_norm": grad_norm,
        }
        round_history.append(round_stats)

        if log_every > 0 and (rnd == 1 or rnd == n_rounds or (rnd % log_every) == 0):
            cname = concept_name if concept_name is not None else "concept"
            print(
                f"[FedCBM][Stage A] {cname} round {rnd}/{n_rounds} "
                f"objective={objective:.5f} hinge={hinge_loss:.5f} active={active_fraction:.3f}"
            )

    return CAVTrainingResult(w=w_global, round_history=round_history)


def _train_local_head(
    z_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    max_iter: int,
) -> Tuple[Optional[Any], int]:
    y_train_int = _as_int_labels(y_train)
    labeled_mask = y_train_int != -1
    if not np.any(labeled_mask):
        return None, 0

    x_labeled = z_train[labeled_mask]
    y_labeled = y_train_int[labeled_mask]
    n_labeled = int(y_labeled.shape[0])

    classes = np.unique(y_labeled)
    if classes.size < 2:
        return _ConstantHead(classes[0]), n_labeled

    clf = LogisticRegression(
        max_iter=max(50, int(max_iter)),
        solver="lbfgs",
        random_state=int(seed),
        # multi_class="auto",
    )
    clf.fit(x_labeled, y_labeled)
    return clf, n_labeled


def _evaluate_head(head: Any, z_eval: np.ndarray, y_eval: np.ndarray) -> Tuple[float, int]:
    y_eval_int = _as_int_labels(y_eval)
    valid_mask = y_eval_int != -1
    if head is None or not np.any(valid_mask):
        return float("nan"), 0
    preds = head.predict(z_eval[valid_mask])
    truth = y_eval_int[valid_mask]
    accuracy = float((preds == truth).mean()) if truth.shape[0] > 0 else float("nan")
    return accuracy, int(truth.shape[0])


def _compute_concept_accuracy(
    test_caches: Sequence[CachedSplit],
    test_scores: Sequence[np.ndarray],
    concept_names: Sequence[str],
    positive_values: Dict[str, Optional[int]],
) -> Dict[str, float]:
    n_concepts = len(concept_names)
    total = np.zeros((n_concepts,), dtype=np.int64)
    correct = np.zeros((n_concepts,), dtype=np.int64)

    for cache, scores in zip(test_caches, test_scores):
        for concept_idx, concept_name in enumerate(concept_names):
            positive_value = positive_values.get(concept_name, None)
            if positive_value is None:
                continue
            labels = _as_int_labels(cache.c[:, concept_idx])
            valid = labels != -1
            if not np.any(valid):
                continue
            gt = (labels[valid] == positive_value).astype(np.int64)
            pred = (scores[valid, concept_idx] >= 0.0).astype(np.int64)
            total[concept_idx] += gt.shape[0]
            correct[concept_idx] += int((gt == pred).sum())

    out: Dict[str, float] = {}
    for concept_idx, concept_name in enumerate(concept_names):
        if total[concept_idx] == 0:
            out[concept_name] = float("nan")
        else:
            out[concept_name] = float(correct[concept_idx] / total[concept_idx])
    return out


def _infer_task_num_classes(
    cfg,
    train_cache: Dict[int, CachedSplit],
    val_cache: Dict[int, CachedSplit],
    test_cache: Sequence[CachedSplit],
) -> int:
    # Prefer dataset/config metadata: this is global and not biased by active-client selection.
    try:
        return max(1, int(cfg.engine.model.y_info.cardinality[0]))
    except Exception:
        pass
    try:
        return max(1, int(cfg.engine.model.output_size))
    except Exception:
        pass

    # Fallback to observed labels only when metadata is unavailable.
    labeled: List[np.ndarray] = []
    for split in list(train_cache.values()) + list(val_cache.values()) + list(test_cache):
        y = _as_int_labels(split.y)
        y = y[y != -1]
        if y.size > 0:
            labeled.append(y)
    if labeled:
        return max(1, int(np.unique(np.concatenate(labeled, axis=0)).size))

    return 2


def run_fedcbm_baseline(
    cfg,
    train_dataloaders: Sequence[Any],
    val_dataloaders: Sequence[Any],
    test_dataloaders: Sequence[Any],
    active_client_indices: Sequence[int],
    concept_names: Sequence[str],
    essential_concepts: Optional[Sequence[str]],
    device: str,
) -> Dict[str, Any]:
    os.makedirs("results", exist_ok=True)

    settings = cfg.learning.settings
    n_rounds = int(settings.get("n_rounds", cfg.trainer.max_epochs))
    local_epochs = int(settings.get("local_epochs", 1))
    svm_lr = float(settings.get("svm_lr", 0.01))
    svm_weight_decay = float(settings.get("svm_weight_decay", 1e-4))
    head_max_iter = int(settings.get("head_max_iter", cfg.trainer.max_epochs))
    svm_log_every = int(settings.get("svm_log_every", max(1, n_rounds // 4)))
    feature_source = str(settings.get("feature_source", "input")).lower()

    if feature_source not in {"input", "model_encoder"}:
        raise ValueError(
            f"Unsupported FedCBM feature_source='{feature_source}'. "
            "Use one of: input, model_encoder."
        )

    backbone: Optional[torch.nn.Module] = None
    if feature_source == "model_encoder":
        model = instantiate(cfg.engine.model)
        model.to(device)
        model.eval()
        backbone = getattr(model, "encoder", None)
        if backbone is None:
            raise ValueError("FedCBM requested feature_source='model_encoder' but model has no encoder.")
        backbone.to(device)
        backbone.eval()

    train_cache: Dict[int, CachedSplit] = {}
    val_cache: Dict[int, CachedSplit] = {}
    for client_idx in active_client_indices:
        train_cache[client_idx] = _extract_split(train_dataloaders[client_idx], backbone, device)
        if val_dataloaders[client_idx] is not None:
            val_cache[client_idx] = _extract_split(val_dataloaders[client_idx], backbone, device)

    filtered_test_loaders = [loader for loader in test_dataloaders if loader is not None]
    if len(filtered_test_loaders) == 0:
        raise ValueError("FedCBM requires at least one test dataloader.")
    test_cache = [_extract_split(loader, backbone, device) for loader in filtered_test_loaders]

    if len(train_cache) == 0:
        raise ValueError("FedCBM received no active clients.")
    first_nonempty = next((v for v in train_cache.values() if v.h.shape[0] > 0), None)
    if first_nonempty is None:
        raise ValueError("FedCBM could not find training samples for active clients.")

    embedding_dim = int(first_nonempty.h.shape[1])
    n_concepts = len(concept_names)
    cav_bank = np.zeros((n_concepts, embedding_dim), dtype=np.float32)
    concept_positive_values: Dict[str, Optional[int]] = {}
    concept_training_details: Dict[str, Dict[str, Any]] = {}
    concept_round_history: Dict[str, List[Dict[str, float]]] = {}

    for concept_idx, concept_name in enumerate(concept_names):
        labeled_values_per_client: List[np.ndarray] = []
        supervised_clients = 0
        for client_idx in active_client_indices:
            labels = _as_int_labels(train_cache[client_idx].c[:, concept_idx])
            labels = labels[labels != -1]
            if labels.size > 0:
                supervised_clients += 1
                labeled_values_per_client.append(labels)

        if len(labeled_values_per_client) == 0:
            concept_positive_values[concept_name] = None
            concept_training_details[concept_name] = {
                "status": "no_supervision",
                "supervised_clients": 0,
                "samples": 0,
            }
            continue

        global_labeled_values = np.concatenate(labeled_values_per_client, axis=0)
        positive_value = _pick_positive_value(global_labeled_values)
        concept_positive_values[concept_name] = positive_value

        if positive_value is None:
            concept_training_details[concept_name] = {
                "status": "single_class",
                "supervised_clients": supervised_clients,
                "samples": int(global_labeled_values.shape[0]),
            }
            continue

        client_buffers: List[Tuple[np.ndarray, np.ndarray, int]] = []
        for client_idx in active_client_indices:
            labels_full = _as_int_labels(train_cache[client_idx].c[:, concept_idx])
            valid = labels_full != -1
            if not np.any(valid):
                continue
            x_local = train_cache[client_idx].h[valid]
            y_local = np.where(labels_full[valid] == positive_value, 1.0, -1.0).astype(np.float32)
            client_buffers.append((x_local.astype(np.float32, copy=False), y_local, int(x_local.shape[0])))

        if len(client_buffers) == 0:
            concept_training_details[concept_name] = {
                "status": "no_valid_buffers",
                "supervised_clients": supervised_clients,
                "samples": int(global_labeled_values.shape[0]),
            }
            continue

        global_binary = np.concatenate([buf[1] for buf in client_buffers], axis=0)
        if np.unique(global_binary).size < 2:
            concept_training_details[concept_name] = {
                "status": "single_binary_class",
                "supervised_clients": supervised_clients,
                "samples": int(global_binary.shape[0]),
            }
            continue

        cav_result = _train_single_cav(
            client_buffers=client_buffers,
            embedding_dim=embedding_dim,
            n_rounds=n_rounds,
            local_epochs=local_epochs,
            lr=svm_lr,
            reg_lambda=svm_weight_decay,
            log_every=svm_log_every,
            concept_name=concept_name,
        )
        cav_bank[concept_idx] = cav_result.w
        concept_round_history[concept_name] = cav_result.round_history
        first_obj = cav_result.round_history[0]["objective"] if cav_result.round_history else float("nan")
        last_obj = cav_result.round_history[-1]["objective"] if cav_result.round_history else float("nan")
        concept_training_details[concept_name] = {
            "status": "trained",
            "supervised_clients": supervised_clients,
            "samples": int(global_binary.shape[0]),
            "n_rounds": int(len(cav_result.round_history)),
            "first_round_objective": first_obj,
            "last_round_objective": last_obj,
        }

    client_heads: Dict[int, Any] = {}
    client_train_labeled: Dict[str, int] = {}
    client_val_acc: Dict[str, float] = {}
    task_fallback_mode: Optional[str] = None

    for offset, client_idx in enumerate(active_client_indices):
        z_train = _project_embeddings(train_cache[client_idx].h, cav_bank)
        head, labeled_samples = _train_local_head(
            z_train=z_train,
            y_train=train_cache[client_idx].y,
            seed=int(cfg.seed) + int(client_idx) + 1,
            max_iter=head_max_iter,
        )
        if head is None:
            continue
        client_heads[client_idx] = head
        client_train_labeled[str(client_idx + 1)] = labeled_samples

        if client_idx in val_cache:
            z_val = _project_embeddings(val_cache[client_idx].h, cav_bank)
            val_acc, _ = _evaluate_head(head, z_val, val_cache[client_idx].y)
            client_val_acc[str(client_idx + 1)] = val_acc

    test_scores = [_project_embeddings(split.h, cav_bank) for split in test_cache]
    client_test_acc: Dict[str, float] = {}
    if len(client_heads) == 0:
        n_classes = _infer_task_num_classes(cfg, train_cache, val_cache, test_cache)
        random_guess_acc = float(1.0 / max(1, n_classes))
        task_fallback_mode = "no_local_heads"
        print(
            f"[FedCBM] Warning: no client could train a local head. "
            f"Using random-guess task accuracy={random_guess_acc:.4f} (n_classes={n_classes})."
        )
        client_test_acc = {str(client_idx + 1): random_guess_acc for client_idx in active_client_indices}
        client_val_acc = {str(client_idx + 1): random_guess_acc for client_idx in active_client_indices if client_idx in val_cache}
        client_train_labeled = {str(client_idx + 1): 0 for client_idx in active_client_indices}
        weighted_acc = random_guess_acc
        macro_acc = random_guess_acc
    else:
        weighted_correct = 0.0
        weighted_total = 0
        for offset, client_idx in enumerate(active_client_indices):
            if client_idx not in client_heads:
                continue
            test_id = offset if len(test_cache) == len(active_client_indices) else 0
            acc, n_eval = _evaluate_head(client_heads[client_idx], test_scores[test_id], test_cache[test_id].y)
            if np.isfinite(acc) and n_eval > 0:
                client_test_acc[str(client_idx + 1)] = acc
                weighted_correct += acc * n_eval
                weighted_total += n_eval

        if weighted_total <= 0:
            n_classes = _infer_task_num_classes(cfg, train_cache, val_cache, test_cache)
            random_guess_acc = float(1.0 / max(1, n_classes))
            task_fallback_mode = "no_labeled_test_samples"
            print(
                f"[FedCBM] Warning: no labeled test samples for local heads. "
                f"Using random-guess task accuracy={random_guess_acc:.4f} (n_classes={n_classes})."
            )
            eval_clients = list(client_heads.keys()) if len(client_heads) > 0 else list(active_client_indices)
            client_test_acc = {str(client_idx + 1): random_guess_acc for client_idx in eval_clients}
            weighted_acc = random_guess_acc
            macro_acc = random_guess_acc
        else:
            weighted_acc = float(weighted_correct / weighted_total)
            macro_acc = float(np.mean(list(client_test_acc.values()))) if len(client_test_acc) > 0 else float("nan")

    concept_accuracy = _compute_concept_accuracy(
        test_caches=test_cache,
        test_scores=test_scores,
        concept_names=concept_names,
        positive_values=concept_positive_values,
    )

    trained_concepts = {
        concept_name
        for concept_name, details in concept_training_details.items()
        if details.get("status") == "trained"
    }
    n_trained_concepts = int(len(trained_concepts))
    concept_coverage_raw = float(n_trained_concepts / n_concepts) if n_concepts > 0 else float("nan")

    if essential_concepts is not None and len(essential_concepts) > 0:
        essential_set = {name for name in essential_concepts if name in concept_names}
    else:
        essential_set = set(concept_names)

    covered_essential = trained_concepts & essential_set
    concept_coverage = (
        float(len(covered_essential) / len(essential_set))
        if len(essential_set) > 0
        else float("nan")
    )

    with open("results/y_accuracy.pkl", "wb") as f:
        pickle.dump({"_baseline": weighted_acc, "macro": macro_acc}, f)
    with open("results/c_accuracy.pkl", "wb") as f:
        pickle.dump(concept_accuracy, f)
    with open("results/single_c_interventions_on_y.pkl", "wb") as f:
        pickle.dump({"_baseline": weighted_acc}, f)
    np.save("results/fedcbm_cav_bank.npy", cav_bank)
    with open("results/fedcbm_positive_values.json", "w") as f:
        json.dump(concept_positive_values, f, indent=2)
    with open("results/fedcbm_svm_round_history.json", "w") as f:
        json.dump(concept_round_history, f, indent=2)
    with open("results/additional_metrics.json", "w") as f:
        json.dump(
            {
                "concept_coverage": concept_coverage,
                "concept_coverage_raw": concept_coverage_raw,
                "percent_params_changed": 0.0,
                "drift_happened": False,
                "last_round": n_rounds,
                "n_concepts_total": int(n_concepts),
                "n_concepts_trained": n_trained_concepts,
                "n_essential_concepts": int(len(essential_set)),
                "n_covered_essential_concepts": int(len(covered_essential)),
            },
            f,
            indent=2,
        )

    fedcbm_metrics = {
        "task_accuracy_weighted": weighted_acc,
        "task_accuracy_macro": macro_acc,
        "client_test_accuracy": client_test_acc,
        "client_val_accuracy": client_val_acc,
        "client_labeled_train_samples": client_train_labeled,
        "n_clients_with_local_head": len(client_heads),
        "n_active_clients": len(active_client_indices),
        "task_fallback_mode": task_fallback_mode,
        "concept_coverage": concept_coverage,
        "concept_coverage_raw": concept_coverage_raw,
        "n_concepts_total": int(n_concepts),
        "n_concepts_trained": n_trained_concepts,
        "n_essential_concepts": int(len(essential_set)),
        "n_covered_essential_concepts": int(len(covered_essential)),
        "concept_training": concept_training_details,
        "feature_source": feature_source,
        "svm_rounds": n_rounds,
        "svm_local_epochs": local_epochs,
        "svm_log_every": svm_log_every,
        "svm_lr": svm_lr,
        "svm_weight_decay": svm_weight_decay,
        "head_max_iter": head_max_iter,
    }
    with open("results/fedcbm_metrics.json", "w") as f:
        json.dump(fedcbm_metrics, f, indent=2)

    return fedcbm_metrics
