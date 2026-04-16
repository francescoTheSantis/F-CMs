import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.models.base import BaseModel
from src.models.layers.base import MLP
from src.models.layers.intervention import maybe_intervene


class MultimodalCBM(BaseModel):
    """
    Multimodal Concept Bottleneck Model.

    Each modality has its own input adapter, while the concept heads and the
    task decoder are shared across modalities.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        concept_hidden_size: int,
        modality_input_sizes: Optional[Dict[str, int]] = None,
        output_size: int = 2,
        n_layers_encoder: int = 1,
        n_layers_concept_encoder: int = 1,
        n_layers_decoder: int = 1,
        activation: str = "leaky_relu",
        concept_loss_weight: float = 0.5,
        decoder_type: str = "mlp",
        c_info: Dict = {},
        y_info: Dict = {},
        c_name_index: Optional[Dict] = None,
        name: str = "cbm_multi",
    ):
        super(MultimodalCBM, self).__init__(
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
        self.is_causal = False
        self.concept_hidden_size = concept_hidden_size
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_concept_encoder = n_layers_concept_encoder
        self.n_layers_decoder = n_layers_decoder
        self.decoder_type = decoder_type
        self.modality_input_sizes = modality_input_sizes or {}
        if len(self.modality_input_sizes) == 0:
            raise ValueError("modality_input_sizes must be provided for MultimodalCBM.")
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

        self.c_mlp = nn.ModuleDict()
        for i, name in enumerate(self.c_info["names"]):
            self.c_mlp[name] = MLP(
                input_size=self.hidden_size,
                hidden_size=self.concept_hidden_size,
                output_size=self.c_info["cardinality"][i],
                n_layers=self.n_layers_concept_encoder,
                activation=self.activation,
            )

        total_concept_dim = sum(self.c_info["cardinality"])
        if self.decoder_type == "mlp":
            self.decoder = MLP(
                input_size=total_concept_dim,
                hidden_size=max(1, total_concept_dim // 2),
                output_size=self.output_size,
                n_layers=self.n_layers_decoder,
                activation=self.activation,
            )
        elif self.decoder_type == "linear":
            self.decoder = nn.Linear(total_concept_dim, self.output_size)
        else:
            raise ValueError(f"Decoder type {self.decoder_type} not supported")

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

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        intervention_index: Optional[torch.Tensor] = None,
        modality=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_encoded = self._encode_input(x, modality=modality)

        intervention_index = self._concept_availability_checker(c, intervention_index)

        c_hat_logits = {}
        c_hat_probs = {}

        for name in self.c_info["names"]:
            i = self.c_name_index[name]
            c_hat_logits[name] = self.c_mlp[name](x_encoded)
            c_hat_probs[name] = torch.softmax(c_hat_logits[name], dim=1)
            c_hat_probs[name] = maybe_intervene(
                c_hat_probs[name],
                c[:, i] if c is not None else None,
                intervention_index[:, i] if intervention_index is not None else torch.zeros_like(c_hat_probs[name]),
            )

        c_probs_concat = torch.cat(list(c_hat_probs.values()), dim=1)
        y_hat_logits = self.decoder(c_probs_concat)
        y_hat_probs = torch.softmax(y_hat_logits, dim=1)
        return y_hat_probs, c_hat_probs

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
