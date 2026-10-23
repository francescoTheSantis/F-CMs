import torch
import torch.nn as nn
from typing import Dict, Optional

from src.models.base import BaseModel
from src.models.layers.base import MLP
from src.models.layers.c_encoder import ConceptBlock
from src.models.layers.intervention import maybe_intervene
from src.utils import get_graph_levels, get_parents


class MultimodalCGM(BaseModel):
    """
    Multimodal adaptation of CGM.

    Each modality has its own input encoder. Concept encoders and graph
    propagation structure are shared across modalities.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        concept_hidden_size,
        modality_input_sizes: Optional[Dict[str, int]] = None,
        output_size=2,
        n_layers_encoder=1,
        n_layers_concept_encoder=1,
        n_layers_propagation=1,
        activation="leaky_relu",
        concept_loss_weight=0.5,
        c_info={},
        y_info={},
        graph=None,
        graph_labels=None,
        cat_latent=False,
        c_name_index=None,
        name: str = "cgm_multi",
    ):
        super(MultimodalCGM, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            c_info=c_info,
            y_info=y_info,
            c_name_index=c_name_index,
            name=name,
        )

        self.has_concepts = True
        self.is_causal = True
        self.concept_hidden_size = concept_hidden_size
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_concept_encoder = n_layers_concept_encoder
        self.n_layers_propagation = n_layers_propagation
        self.graph = torch.Tensor(graph).int() if graph is not None else None
        self.graph_labels = graph_labels
        self.cat_latent = cat_latent
        self.modality_input_sizes = modality_input_sizes or {}
        if len(self.modality_input_sizes) == 0:
            raise ValueError("modality_input_sizes must be provided for MultimodalCGM.")
        self.supported_modalities = list(self.modality_input_sizes.keys())

        self._setup_concept_loss_weight(concept_loss_weight)
        self._build_model()

    def _build_model(self):
        if hasattr(self, "encoder"):
            del self.encoder

        self.modality_encoders = nn.ModuleDict()
        for modality, input_size in self.modality_input_sizes.items():
            self.modality_encoders[modality] = MLP(
                input_size=input_size,
                hidden_size=self.hidden_size,
                output_size=None,
                n_layers=self.n_layers_encoder,
                activation=self.activation,
            )

        self.c_names = self.c_info["names"]
        self.y_names = self.y_info["names"]
        self.combo_info = {
            "names": self.c_info["names"] + self.y_info["names"],
            "cardinality": self.c_info["cardinality"] + self.y_info["cardinality"],
        }

        assert self.combo_info["names"] == self.graph_labels
        if len(self.combo_info["names"]) == len(self.c_name_index.keys()):
            assert self.combo_info["names"] == list(self.c_name_index.keys())

        task_index = self.combo_info["names"].index(self.y_names[0])
        self.graph_levels = get_graph_levels(self.graph, task_index)
        self.roots = self.graph_levels[0]
        flat_levels = [node for level in self.graph_levels for node in level]
        self.predicted_concepts = [self.graph_labels[i] for i in flat_levels]
        self.predicted_concepts = [c for c in self.predicted_concepts if c != self.y_names[0]]
        self.roots_info = {
            "names": [name for i, name in enumerate(self.combo_info["names"]) if i in self.roots],
            "cardinality": [card for i, card in enumerate(self.combo_info["cardinality"]) if i in self.roots],
        }
        if self.y_names[0] in self.roots_info["names"]:
            raise ValueError("The target variable cannot be a root concept")

        self.concept_encoders = nn.ModuleDict()
        for name in self.combo_info["names"]:
            if name in self.roots_info["names"]:
                self.concept_encoders[name] = ConceptBlock(
                    input_size=self.hidden_size,
                    hidden_size=self.concept_hidden_size,
                    n_layers=self.n_layers_concept_encoder,
                    activation=self.activation,
                    c_cardinality=[c for n, c in zip(self.combo_info["names"], self.combo_info["cardinality"]) if n == name][0],
                )
            else:
                node_idx = self.combo_info["names"].index(name)
                parents = get_parents(self.graph, node_idx).tolist()
                self.concept_encoders[name] = ConceptBlock(
                    input_size=len(parents) * self.concept_hidden_size,
                    hidden_size=self.concept_hidden_size,
                    n_layers=self.n_layers_concept_encoder,
                    activation=self.activation,
                    c_cardinality=[c for n, c in zip(self.combo_info["names"], self.combo_info["cardinality"]) if n == name][0],
                )

    def _normalize_modality(self, modality) -> Optional[str]:
        if modality is None:
            return None
        if isinstance(modality, (list, tuple)):
            unique_modalities = set(modality)
            if len(unique_modalities) != 1:
                raise ValueError("A batch must contain a single modality.")
            modality = next(iter(unique_modalities))
        if isinstance(modality, torch.Tensor):
            if modality.numel() != 1:
                raise ValueError("Tensor modality is supported only for scalar values.")
            modality = modality.item()
        modality = str(modality)
        if modality not in self.modality_encoders:
            raise ValueError(f"Unsupported modality '{modality}'. Expected one of {self.supported_modalities}.")
        return modality

    def _infer_modality_from_input(self, x: torch.Tensor) -> str:
        matches = [
            modality
            for modality, input_size in self.modality_input_sizes.items()
            if x.shape[-1] == input_size
        ]
        if len(matches) != 1:
            raise ValueError(
                "Unable to infer modality from input shape. "
                "Pass modality explicitly or ensure modality input sizes are unique."
            )
        return matches[0]

    def _encode_input(self, x: torch.Tensor, modality=None) -> torch.Tensor:
        modality_name = self._normalize_modality(modality)
        if modality_name is None:
            modality_name = self._infer_modality_from_input(x)
        return self.modality_encoders[modality_name](x)

    def forward(self, x, c=None, intervention_index=None, modality=None):
        x_encoded = self._encode_input(x, modality=modality)
        intervention_index = self._concept_availability_checker(c, intervention_index)

        c_embs, c_probs = {}, {}
        for level in self.graph_levels:
            for i in level:
                name = self.combo_info["names"][i]
                c_index = self.c_name_index[name]

                if name in self.roots_info["names"]:
                    concept_encoder_input = x_encoded
                    c_int = c[:, c_index] if c is not None else None
                    int_idx = intervention_index[:, c_index] if intervention_index is not None else None
                elif name in self.c_names:
                    p_indices = get_parents(self.graph, i).tolist()
                    p_names = [self.combo_info["names"][p] for p in p_indices]
                    concept_encoder_input = torch.cat([c_embs[p_name] for p_name in p_names], dim=1)
                    c_int = c[:, c_index] if c is not None else None
                    int_idx = intervention_index[:, c_index] if intervention_index is not None else None
                else:
                    p_indices = get_parents(self.graph, i).tolist()
                    p_names = [self.combo_info["names"][p] for p in p_indices]
                    concept_encoder_input = torch.cat([c_embs[p_name] for p_name in p_names], dim=1)
                    c_int = None
                    int_idx = None

                c_embs[name], c_probs[name] = self.concept_encoders[name](
                    concept_encoder_input,
                    c_int,
                    int_idx,
                    to_return=["embs", "probs"],
                )

                if name not in self.y_names and c is not None and intervention_index is not None:
                    c_probs[name] = maybe_intervene(c_probs[name], c[:, c_index], intervention_index[:, c_index])

        y_hat_probs = c_probs[self.y_names[0]]
        c_hat_probs = {k: v for k, v in c_probs.items() if k in self.c_names and k not in self.virtual_roots}
        return y_hat_probs, c_hat_probs

    def filter_output_for_loss(self, y_output, c_output):
        return y_output, c_output

    def filter_output_for_metric(self, y_output, c_output):
        return y_output, c_output

    def loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        c_hat_dict: Dict[str, torch.Tensor],
        c: torch.Tensor,
        reduction: str = "mean",
        ignore_index: int = -1,
        multi_output: bool = False,
    ) -> torch.Tensor:
        return self._concept_based_loss(
            y_hat,
            y,
            c_hat_dict,
            c,
            reduction,
            ignore_index,
            multi_output=multi_output,
        )
