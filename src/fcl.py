import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EvalStats:
    loss: float
    accuracy: float
    n_samples: int
    n_labeled: int


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


class BotCLStaticModel(nn.Module):
    """
    BotCL-inspired static concept bottleneck:
    encoder -> concept activations -> bias-free class-concept matrix.

    NOTE:
    The original BotCL/FCL image model uses slot-attention and positional encoding.
    Here we keep the core semantics but use a lighter encoder for clean integration
    with this framework across tabular and image datasets.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_concepts: int,
        n_classes: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.n_classes = int(n_classes)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.is_image_input = len(self.input_shape) >= 2

        if self.is_image_input:
            in_channels = int(self.input_shape[0])
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, hidden_dim),
                nn.ReLU(inplace=True),
            )
            encoder_out = int(hidden_dim)
        else:
            flat_dim = int(np.prod(self.input_shape))
            self.encoder = nn.Sequential(
                nn.Linear(flat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            encoder_out = int(hidden_dim)

        self.concept_layer = nn.Linear(encoder_out, self.n_concepts, bias=True)
        # Paper Eq. (9): class score from concept vector using matrix Z in R^{w x k}.
        self.classifier = nn.Linear(self.n_concepts, self.n_classes, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        if self.is_image_input:
            if x.dim() != 4:
                x = x.view(x.shape[0], *self.input_shape)
        else:
            x = x.view(x.shape[0], -1)

        h = self.encoder(x)
        # BotCL-like concept activation path:
        # non-negative concept scores are squashed to [0, 1), then scaled to [-1, 1)
        # for the quantization term.
        concept_raw = torch.tanh(F.relu(self.concept_layer(h)))
        concept_scaled = (concept_raw - 0.5) * 2.0

        # Classifier matrix Z acts on concept activations (paper Eq. 9).
        logits = self.classifier(concept_raw)
        return logits, concept_scaled

    def get_classifier_matrix(self) -> np.ndarray:
        return self.classifier.weight.detach().cpu().numpy().astype(np.float32, copy=True)

    def set_classifier_matrix(self, matrix: np.ndarray) -> None:
        weight = torch.as_tensor(matrix, dtype=self.classifier.weight.dtype, device=self.classifier.weight.device)
        if weight.shape != self.classifier.weight.shape:
            raise ValueError(
                f"Classifier matrix shape mismatch: got {tuple(weight.shape)}, "
                f"expected {tuple(self.classifier.weight.shape)}"
            )
        with torch.no_grad():
            self.classifier.weight.copy_(weight)


def _quantization_loss(concept_act: torch.Tensor) -> torch.Tensor:
    # Matches original repository implementation:
    # q_loss = mean((|cpt| - 1)^2) where cpt is the scaled concept output in [-1, 1).
    return torch.mean((torch.abs(concept_act) - 1.0) ** 2)


def _train_local_model(
    model: BotCLStaticModel,
    train_loader: Any,
    device: str,
    local_epochs: int,
    lr: float,
    lr_gamma: float,
    weight_decay: float,
    quantization_bias: float,
    matrix_l2_weight: float,
) -> Dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=float(lr_gamma),
    )

    total_loss = 0.0
    total_samples = 0
    total_labeled = 0
    total_correct = 0

    for _ in range(max(1, int(local_epochs))):
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].view(-1).long().to(device)

            logits, concept_act = model(x)
            valid = y != -1
            if valid.any():
                cls_loss = F.cross_entropy(logits[valid], y[valid])
                preds = logits[valid].argmax(dim=1)
                total_correct += int((preds == y[valid]).sum().item())
                total_labeled += int(valid.sum().item())
            else:
                cls_loss = logits.sum() * 0.0

            qua_loss = _quantization_loss(concept_act)
            l2_matrix = torch.mean(model.classifier.weight ** 2)
            loss = cls_loss + (quantization_bias * qua_loss) + (matrix_l2_weight * l2_matrix)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = int(x.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        scheduler.step()

    avg_loss = float(total_loss / total_samples) if total_samples > 0 else float("nan")
    accuracy = float(total_correct / total_labeled) if total_labeled > 0 else float("nan")
    return {
        "train_loss": avg_loss,
        "train_accuracy": accuracy,
        "n_samples": int(total_samples),
        "n_labeled": int(total_labeled),
    }


@torch.no_grad()
def _evaluate_model(
    model: BotCLStaticModel,
    data_loader: Any,
    device: str,
    quantization_bias: float,
    matrix_l2_weight: float,
) -> EvalStats:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_labeled = 0
    total_correct = 0

    for batch in data_loader:
        x = batch["x"].to(device)
        y = batch["y"].view(-1).long().to(device)

        logits, concept_act = model(x)
        valid = y != -1
        if valid.any():
            cls_loss = F.cross_entropy(logits[valid], y[valid])
            preds = logits[valid].argmax(dim=1)
            total_correct += int((preds == y[valid]).sum().item())
            total_labeled += int(valid.sum().item())
        else:
            cls_loss = logits.sum() * 0.0

        qua_loss = _quantization_loss(concept_act)
        l2_matrix = torch.mean(model.classifier.weight ** 2)
        loss = cls_loss + (quantization_bias * qua_loss) + (matrix_l2_weight * l2_matrix)

        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    avg_loss = float(total_loss / total_samples) if total_samples > 0 else float("nan")
    accuracy = float(total_correct / total_labeled) if total_labeled > 0 else float("nan")
    return EvalStats(
        loss=avg_loss,
        accuracy=accuracy,
        n_samples=int(total_samples),
        n_labeled=int(total_labeled),
    )


def _aggregate_class_concept_matrices(
    local_matrices: Sequence[np.ndarray],
    sample_counts: Sequence[int],
    robust_aggregation: bool,
    trim_ratio: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(local_matrices) == 0:
        raise ValueError("No local matrices provided for aggregation.")

    matrices = np.stack(local_matrices, axis=0).astype(np.float64, copy=False)
    weights = np.asarray(sample_counts, dtype=np.float64)
    if weights.shape[0] != matrices.shape[0]:
        raise ValueError(
            f"Sample count length mismatch: {weights.shape[0]} vs {matrices.shape[0]} local matrices."
        )

    n_clients, n_classes, n_concepts = matrices.shape
    trim_count = 0
    use_trimmed = bool(robust_aggregation and trim_ratio > 0.0 and n_clients >= 3)

    if not use_trimmed:
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            aggregated = np.mean(matrices, axis=0)
        else:
            aggregated = np.tensordot(weights, matrices, axes=([0], [0])) / weight_sum
    else:
        trim_count = int(math.ceil(float(trim_ratio) * n_clients))
        trim_count = min(trim_count, max(0, (n_clients - 1) // 2))
        if trim_count <= 0:
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0:
                aggregated = np.mean(matrices, axis=0)
            else:
                aggregated = np.tensordot(weights, matrices, axes=([0], [0])) / weight_sum
        else:
            aggregated = np.zeros((n_classes, n_concepts), dtype=np.float64)
            for class_idx in range(n_classes):
                for concept_idx in range(n_concepts):
                    values = matrices[:, class_idx, concept_idx]
                    order = np.argsort(values)
                    kept = order[trim_count : n_clients - trim_count]
                    if kept.size == 0:
                        kept = order
                    kept_weights = weights[kept]
                    weight_sum = float(np.sum(kept_weights))
                    if weight_sum <= 0:
                        aggregated[class_idx, concept_idx] = float(np.mean(values[kept]))
                    else:
                        aggregated[class_idx, concept_idx] = float(
                            np.dot(values[kept], kept_weights) / weight_sum
                        )

    meta = {
        "n_clients": int(n_clients),
        "robust_aggregation": bool(robust_aggregation),
        "trim_ratio": float(trim_ratio),
        "trim_count": int(trim_count),
    }
    return aggregated.astype(np.float32, copy=False), meta


def _infer_positive_values(
    train_dataloaders: Sequence[Any],
    active_client_indices: Sequence[int],
    concept_names: Sequence[str],
) -> Dict[str, Optional[int]]:
    positive_values: Dict[str, Optional[int]] = {}
    n_concepts = len(concept_names)

    collected: List[List[np.ndarray]] = [[] for _ in range(n_concepts)]
    for client_idx in active_client_indices:
        loader = train_dataloaders[client_idx]
        for batch in loader:
            c = batch["c"].detach().cpu().numpy()
            for concept_idx in range(n_concepts):
                labels = _as_int_labels(c[:, concept_idx])
                labels = labels[labels != -1]
                if labels.size > 0:
                    collected[concept_idx].append(labels)

    for concept_idx, concept_name in enumerate(concept_names):
        if len(collected[concept_idx]) == 0:
            positive_values[concept_name] = None
            continue
        values = np.concatenate(collected[concept_idx], axis=0)
        positive_values[concept_name] = _pick_positive_value(values)

    return positive_values


@torch.no_grad()
def _accumulate_concept_accuracy(
    model: BotCLStaticModel,
    data_loader: Any,
    device: str,
    concept_names: Sequence[str],
    positive_values: Dict[str, Optional[int]],
    totals: np.ndarray,
    corrects: np.ndarray,
) -> None:
    model.eval()
    for batch in data_loader:
        x = batch["x"].to(device)
        c_true = batch["c"].detach().cpu().numpy()
        _, concept_act = model(x)
        concept_pred = concept_act.detach().cpu().numpy()

        for concept_idx, concept_name in enumerate(concept_names):
            pos_value = positive_values.get(concept_name, None)
            if pos_value is None:
                continue
            labels = _as_int_labels(c_true[:, concept_idx])
            valid = labels != -1
            if not np.any(valid):
                continue
            gt = (labels[valid] == pos_value).astype(np.int64, copy=False)
            pred = (concept_pred[valid, concept_idx] >= 0.0).astype(np.int64, copy=False)
            totals[concept_idx] += int(gt.shape[0])
            corrects[concept_idx] += int((gt == pred).sum())


def _resolve_test_loader(test_dataloaders: Sequence[Any], active_position: int):
    if len(test_dataloaders) == 0:
        raise ValueError("FCL baseline requires at least one test dataloader.")
    if len(test_dataloaders) == 1:
        return test_dataloaders[0]
    if active_position < len(test_dataloaders):
        return test_dataloaders[active_position]
    return test_dataloaders[0]


def run_fcl_baseline(
    cfg,
    train_dataloaders: Sequence[Any],
    val_dataloaders: Sequence[Any],
    test_dataloaders: Sequence[Any],
    active_client_indices: Sequence[int],
    concept_names: Sequence[str],
    n_classes: int,
    device: str,
) -> Dict[str, Any]:
    os.makedirs("results", exist_ok=True)

    if len(active_client_indices) == 0:
        raise ValueError("FCL baseline received no active clients.")

    settings = cfg.learning.settings
    n_rounds = int(settings.get("n_rounds", cfg.trainer.max_epochs))
    local_epochs = int(settings.get("local_epochs", 1))
    lr = float(settings.get("lr", 1e-3))
    lr_gamma = float(settings.get("lr_gamma", 0.7))
    weight_decay = float(settings.get("weight_decay", 0.0))
    encoder_hidden_dim = int(settings.get("encoder_hidden_dim", 128))
    quantization_bias = float(settings.get("quantization_bias", 0.05))
    matrix_l2_weight = float(settings.get("matrix_l2_weight", 0.0))
    robust_aggregation = bool(settings.get("robust_aggregation", True))
    trim_ratio = float(settings.get("trim_ratio", 0.2))
    trainer_patience = getattr(cfg.trainer, "patience", 0) if hasattr(cfg, "trainer") else 0
    patience = int(settings.get("patience", trainer_patience))
    min_delta = float(settings.get("min_delta", 1e-8))

    first_batch = None
    for client_idx in active_client_indices:
        loader = train_dataloaders[client_idx]
        try:
            first_batch = next(iter(loader))
            break
        except StopIteration:
            continue
    if first_batch is None:
        raise ValueError("FCL baseline could not find samples in active client train loaders.")

    input_shape = tuple(int(v) for v in first_batch["x"].shape[1:])
    n_concepts = int(len(concept_names))
    if n_concepts <= 0:
        raise ValueError("FCL baseline requires at least one concept.")

    template_model = BotCLStaticModel(
        input_shape=input_shape,
        n_concepts=n_concepts,
        n_classes=int(n_classes),
        hidden_dim=encoder_hidden_dim,
    ).to(device)
    template_state = {k: v.detach().cpu().clone() for k, v in template_model.state_dict().items()}

    client_models: Dict[int, BotCLStaticModel] = {}
    for client_idx in active_client_indices:
        model = BotCLStaticModel(
            input_shape=input_shape,
            n_concepts=n_concepts,
            n_classes=int(n_classes),
            hidden_dim=encoder_hidden_dim,
        ).to(device)
        model.load_state_dict(template_state, strict=True)
        client_models[client_idx] = model

    # Server-side initialization of the class-concept matrix Z.
    global_matrix = template_model.get_classifier_matrix()
    best_global_matrix = global_matrix.copy()
    best_round = 0
    best_val_loss = float("inf")
    rounds_without_improvement = 0

    history: Dict[str, Any] = {
        "round": [],
        "val_loss_avg": [],
        "val_y_acc_avg": [],
        "aggregation": [],
        "client_train": {},
        "client_val": {},
    }

    executed_rounds = 0

    for rnd in range(1, n_rounds + 1):
        executed_rounds = rnd
        local_matrices: List[np.ndarray] = []
        local_sizes: List[int] = []
        history["client_train"][str(rnd)] = {}
        history["client_val"][str(rnd)] = {}

        print(f"\033[92m[FCL] Round {rnd}/{n_rounds}\033[0m")

        # Broadcast aggregated matrix to local models (only classifier/co-occurrence block).
        for client_idx in active_client_indices:
            client_models[client_idx].set_classifier_matrix(global_matrix)

        for client_idx in active_client_indices:
            train_stats = _train_local_model(
                model=client_models[client_idx],
                train_loader=train_dataloaders[client_idx],
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                lr_gamma=lr_gamma,
                weight_decay=weight_decay,
                quantization_bias=quantization_bias,
                matrix_l2_weight=matrix_l2_weight,
            )
            history["client_train"][str(rnd)][str(client_idx + 1)] = train_stats

            local_matrices.append(client_models[client_idx].get_classifier_matrix())
            local_sizes.append(max(1, int(len(train_dataloaders[client_idx].dataset))))

        global_matrix, agg_meta = _aggregate_class_concept_matrices(
            local_matrices=local_matrices,
            sample_counts=local_sizes,
            robust_aggregation=robust_aggregation,
            trim_ratio=trim_ratio,
        )
        history["aggregation"].append(agg_meta)

        # Synchronize only class-concept matrix back to clients.
        for client_idx in active_client_indices:
            client_models[client_idx].set_classifier_matrix(global_matrix)

        val_losses: List[float] = []
        val_loss_weights: List[int] = []
        val_acc_values: List[float] = []
        val_acc_weights: List[int] = []

        for client_idx in active_client_indices:
            if client_idx >= len(val_dataloaders):
                continue
            val_loader = val_dataloaders[client_idx]
            if val_loader is None:
                continue
            val_stats = _evaluate_model(
                model=client_models[client_idx],
                data_loader=val_loader,
                device=device,
                quantization_bias=quantization_bias,
                matrix_l2_weight=matrix_l2_weight,
            )
            history["client_val"][str(rnd)][str(client_idx + 1)] = {
                "val_loss": val_stats.loss,
                "val_accuracy": val_stats.accuracy,
                "n_samples": val_stats.n_samples,
                "n_labeled": val_stats.n_labeled,
            }

            if np.isfinite(val_stats.loss) and val_stats.n_samples > 0:
                val_losses.append(float(val_stats.loss))
                val_loss_weights.append(int(val_stats.n_samples))
            if np.isfinite(val_stats.accuracy) and val_stats.n_labeled > 0:
                val_acc_values.append(float(val_stats.accuracy))
                val_acc_weights.append(int(val_stats.n_labeled))

        if len(val_losses) > 0:
            w = np.asarray(val_loss_weights, dtype=np.float64)
            v = np.asarray(val_losses, dtype=np.float64)
            val_loss_avg = float(np.dot(v, w) / np.sum(w))
        else:
            val_loss_avg = float("nan")

        if len(val_acc_values) > 0:
            w = np.asarray(val_acc_weights, dtype=np.float64)
            v = np.asarray(val_acc_values, dtype=np.float64)
            val_acc_avg = float(np.dot(v, w) / np.sum(w))
        else:
            val_acc_avg = float("nan")

        history["round"].append(int(rnd))
        history["val_loss_avg"].append(val_loss_avg)
        history["val_y_acc_avg"].append(val_acc_avg)

        print(
            f"\033[94m[FCL] round={rnd} val_loss={val_loss_avg:.5f} "
            f"val/y/y_accuracy={val_acc_avg:.5f}\033[0m"
        )

        improved = False
        if np.isfinite(val_loss_avg):
            if val_loss_avg + min_delta < best_val_loss:
                improved = True
                best_val_loss = float(val_loss_avg)
                best_global_matrix = global_matrix.copy()
                best_round = int(rnd)
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
        elif rnd == n_rounds and best_round == 0:
            best_round = int(rnd)
            best_global_matrix = global_matrix.copy()

        if patience > 0 and rounds_without_improvement >= patience:
            print(f"\033[93m[FCL] Early stopping at round {rnd} (patience={patience}).\033[0m")
            break

    if best_round == 0:
        best_round = int(executed_rounds)
        best_global_matrix = global_matrix.copy()
    print(
        f"\033[96m[FCL] Selected best_round={best_round} "
        f"with val_loss={best_val_loss if np.isfinite(best_val_loss) else float('nan'):.5f}\033[0m"
    )

    for client_idx in active_client_indices:
        client_models[client_idx].set_classifier_matrix(best_global_matrix)

    client_test_accuracy: Dict[str, float] = {}
    weighted_correct = 0.0
    weighted_total = 0

    concept_positive_values = _infer_positive_values(
        train_dataloaders=train_dataloaders,
        active_client_indices=active_client_indices,
        concept_names=concept_names,
    )
    concept_totals = np.zeros((n_concepts,), dtype=np.int64)
    concept_corrects = np.zeros((n_concepts,), dtype=np.int64)

    for active_pos, client_idx in enumerate(active_client_indices):
        test_loader = _resolve_test_loader(test_dataloaders, active_pos)
        test_stats = _evaluate_model(
            model=client_models[client_idx],
            data_loader=test_loader,
            device=device,
            quantization_bias=quantization_bias,
            matrix_l2_weight=matrix_l2_weight,
        )
        if np.isfinite(test_stats.accuracy) and test_stats.n_labeled > 0:
            client_test_accuracy[str(client_idx + 1)] = float(test_stats.accuracy)
            weighted_correct += float(test_stats.accuracy) * int(test_stats.n_labeled)
            weighted_total += int(test_stats.n_labeled)

        _accumulate_concept_accuracy(
            model=client_models[client_idx],
            data_loader=test_loader,
            device=device,
            concept_names=concept_names,
            positive_values=concept_positive_values,
            totals=concept_totals,
            corrects=concept_corrects,
        )

    if weighted_total <= 0:
        raise ValueError(
            "[FCL] No labeled test samples were available for clients with trained local models."
        )

    weighted_acc = float(weighted_correct / weighted_total)
    macro_acc = float(np.mean(list(client_test_accuracy.values()))) if len(client_test_accuracy) > 0 else float("nan")

    concept_accuracy: Dict[str, float] = {}
    for concept_idx, concept_name in enumerate(concept_names):
        if concept_totals[concept_idx] <= 0:
            concept_accuracy[concept_name] = float("nan")
        else:
            concept_accuracy[concept_name] = float(
                concept_corrects[concept_idx] / concept_totals[concept_idx]
            )

    n_supervised_concepts = int(sum(1 for v in concept_positive_values.values() if v is not None))
    concept_coverage = float(n_supervised_concepts / n_concepts) if n_concepts > 0 else float("nan")

    with open("results/y_accuracy.pkl", "wb") as f:
        pickle.dump({"_baseline": weighted_acc, "macro": macro_acc}, f)
    with open("results/c_accuracy.pkl", "wb") as f:
        pickle.dump(concept_accuracy, f)
    with open("results/single_c_interventions_on_y.pkl", "wb") as f:
        pickle.dump({"_baseline": weighted_acc}, f)
    with open("results/fcl_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("results/additional_metrics.json", "w") as f:
        json.dump(
            {
                "concept_coverage": concept_coverage,
                "structural_concept_coverage": float(
                    cfg.learning.subgraphs.get(
                        "structural_concept_coverage", float("nan")
                    )
                ),
                "percent_params_changed": 0.0,
                "drift_happened": False,
                "last_round": int(executed_rounds),
                "n_concepts_total": int(n_concepts),
                "n_concepts_supervised": int(n_supervised_concepts),
            },
            f,
            indent=2,
        )

    np.save("results/fcl_global_matrix.npy", best_global_matrix)

    # Track explicit approximations vs original BotCL/FCL implementation.
    implementation_notes = [
        "Original BotCL uses slot-attention with positional encoding; this baseline uses a lighter encoder + concept layer while preserving encoder->concept->class-matrix semantics.",
        "Paper and method description aggregate class-concept matrix Z only; this implementation follows that design and does not federate full model weights.",
        "Server uses per-entry trimmed weighted aggregation (top/bottom removal) when enabled, aligned with the paper's malicious-client mitigation.",
    ]

    fcl_metrics = {
        "task_accuracy_weighted": weighted_acc,
        "task_accuracy_macro": macro_acc,
        "client_test_accuracy": client_test_accuracy,
        "n_active_clients": int(len(active_client_indices)),
        "n_rounds_executed": int(executed_rounds),
        "best_round": int(best_round),
        "local_epochs": int(local_epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "lr_gamma": float(lr_gamma),
        "quantization_bias": float(quantization_bias),
        "matrix_l2_weight": float(matrix_l2_weight),
        "robust_aggregation": bool(robust_aggregation),
        "trim_ratio": float(trim_ratio),
        "concept_coverage": concept_coverage,
        "n_concepts_total": int(n_concepts),
        "n_concepts_supervised": int(n_supervised_concepts),
        "positive_values": concept_positive_values,
        "aggregation_history": history["aggregation"],
        "implementation_notes": implementation_notes,
    }
    with open("results/fcl_metrics.json", "w") as f:
        json.dump(fcl_metrics, f, indent=2)

    return fcl_metrics
