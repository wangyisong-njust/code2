from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer used by DANN.

    Forward is identity, backward multiplies the upstream gradient by ``-lambda_``.
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


def _make_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class SDAEClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        stats_dim: int,
        stats_hidden_dims: list[int],
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty.")

        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.stats_projector = _make_mlp(stats_dim, stats_hidden_dims, stats_hidden_dims[-1], dropout)
        self.feature_dim = latent_dim + stats_hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, max(self.feature_dim // 2, num_classes)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(self.feature_dim // 2, num_classes), num_classes),
        )
        self.source_only_classifier = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim // 2, num_classes)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim // 2, num_classes), num_classes),
        )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim // 2, 32)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim // 2, 32), 2),
        )

    def discriminate_domain(self, latent: torch.Tensor, lambda_: float) -> torch.Tensor:
        reversed_latent = grad_reverse(latent, lambda_)
        return self.domain_discriminator(reversed_latent)

    def forward(self, signals: torch.Tensor, stats: torch.Tensor, use_stats: bool) -> Dict[str, torch.Tensor]:
        flat = signals.flatten(start_dim=1)
        latent = self.encoder(flat)
        reconstruction = self.decoder(latent)
        if use_stats:
            stats_features = self.stats_projector(stats)
            features = torch.cat([latent, stats_features], dim=1)
            logits = self.classifier(features)
        else:
            stats_features = None
            features = latent
            logits = self.source_only_classifier(latent)
        return {
            "logits": logits,
            "latent": latent,
            "features": features,
            "stats_features": stats_features,
            "reconstruction": reconstruction,
        }


def build_model(config: dict, input_dim: int, stats_dim: int, num_classes: int) -> SDAEClassifier:
    model_cfg = config["model"]
    return SDAEClassifier(
        input_dim=input_dim,
        hidden_dims=list(model_cfg["hidden_dims"]),
        latent_dim=int(model_cfg["latent_dim"]),
        stats_dim=stats_dim,
        stats_hidden_dims=list(model_cfg["stats_hidden_dims"]),
        num_classes=num_classes,
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
