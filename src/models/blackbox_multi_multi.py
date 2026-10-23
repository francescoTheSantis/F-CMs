import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.models.base import BaseModel
from src.models.layers.base import MLP


class MultimodalBlackBoxMulti(BaseModel):
    """
    Multimodal opaque model with concept supervision.

    Each modality has its own encoder, while the output decoder is shared and
    predicts both the task and all concepts from the shared hidden space.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        modality_input_sizes: Optional[Dict[str, int]] = None,
        output_size: int = 2,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        activation: str = "leaky_relu",
        concept_loss_weight: float = 0.5,
        c_info: Dict = {},
        y_info: Dict = {},
        c_name_index: Optional[Dict] = None,
        name: str = "blackbox_multi_multi",
    ):
        super(MultimodalBlackBoxMulti, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_layers_encoder=n_layers_encoder,
            activation=activation,
            c_info=c_info,
            y_info=y_info,
            c_name_index=c_name_index,
            name=name,
        )

        self.has_concepts = True
        self.is_causal = False
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.concept_cardinality = c_info["cardinality"]
        self.task_cardinality = y_info["cardinality"]
        self.modality_input_sizes = modality_input_sizes or {}
        if len(self.modality_input_sizes) == 0:
            raise ValueError("modality_input_sizes must be provided for MultimodalBlackBoxMulti.")
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

        total_output_size = self.output_size + sum(self.concept_cardinality)
        decoder_hidden_size = max(1, total_output_size * 2)
        self.decoder = MLP(
            input_size=self.hidden_size,
            hidden_size=decoder_hidden_size,
            output_size=total_output_size,
            n_layers=self.n_layers_decoder,
            activation=self.activation,
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

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        intervention_index: Optional[torch.Tensor] = None,
        modality=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden = self._encode_input(x, modality=modality)
        all_hat_logits = self.decoder(hidden)

        y_hat_probs = torch.softmax(all_hat_logits[:, : self.task_cardinality[0]], dim=1)

        c_hat_probs = {}
        concept_offset = self.task_cardinality[0]
        for i, name in enumerate(self.concept_names):
            start_idx = concept_offset + sum(self.concept_cardinality[:i])
            end_idx = concept_offset + sum(self.concept_cardinality[: i + 1])
            c_hat_probs[name] = torch.softmax(all_hat_logits[:, start_idx:end_idx], dim=1)

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
