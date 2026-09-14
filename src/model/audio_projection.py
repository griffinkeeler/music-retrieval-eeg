import torch
import torch.nn.functional as F
from torch import nn


class AudioProjection(nn.Module):
    """Shared projection for vector or sequence audio embeddings.

    The parameter shapes match nn.Linear so older checkpoints remain loadable,
    while sequence inputs [B, C, T] are projected with a 1x1 convolution.
    """

    def __init__(self, in_features=768, out_features=512, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.ndim == 2:
            return F.linear(x, self.weight, self.bias)
        if x.ndim == 3:
            return F.conv1d(x, self.weight.unsqueeze(-1), self.bias)
        raise ValueError(
            f"AudioProjection expected [B, C] or [B, C, T], got shape {tuple(x.shape)}"
        )


class IdentityAudioProjection(nn.Module):
    """No-op projection used for direct EEG-to-audio alignment."""

    def forward(self, x):
        if x.ndim not in {2, 3}:
            raise ValueError(
                "IdentityAudioProjection expected [B, C] or [B, C, T], "
                f"got shape {tuple(x.shape)}"
            )
        return x


def build_audio_projection(in_features=768, out_features=512, mode="linear"):
    if mode == "identity":
        if in_features != out_features:
            raise ValueError(
                "Identity audio projection requires matching in/out feature dims, "
                f"got {in_features} -> {out_features}."
            )
        return IdentityAudioProjection()

    if mode != "linear":
        raise ValueError("Audio projection mode must be one of ['linear', 'identity'].")

    return AudioProjection(
        in_features=in_features,
        out_features=out_features,
        bias=True,
    )
