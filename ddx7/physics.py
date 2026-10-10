import torch
import torch.nn as nn
from ddx7.core import get_mlp


class BowPhysicsModule(nn.Module):
    """
    Bridges TCNBowDecoder's 3 bow-control outputs into FMSynth's 6-channel
    raw ('ol') logits, via a fixed friction-nonlinearity feature plus a
    small learned MLP. Output stays unbounded - FMSynth applies its own
    sigmoid+scale.
    """

    def __init__(
        self,
        hidden_size=64,
        n_layers=2,
        out_channels=6,
        friction_a=5.0,
        learnable_a=True,
        fr=[1, 1, 1, 1, 3, 14],
    ):
        super().__init__()
        a_init = torch.tensor(float(friction_a))
        self.learnable_a = learnable_a
        if learnable_a:
            self.raw_a = nn.Parameter(
                torch.log(torch.exp(a_init) - 1)
            )  # softplus param
        else:
            self.register_buffer("fixed_a", a_init)

        # in: [bow_force, bow_velocity, bow_bridge_distance, friction_signal]
        self.in_mlp = get_mlp(4, hidden_size, n_layers)
        self.out_proj = nn.Linear(hidden_size, out_channels)

        fr_t = torch.tensor(fr, dtype=torch.float32)
        prior = (
            torch.log(fr_t) / torch.log(fr_t).max()
        )  # structural bridge-distance prior from FM ratios
        self.bridge_weight = nn.Parameter(prior.clone())

    def _a(self):
        return nn.functional.softplus(self.raw_a) if self.learnable_a else self.fixed_a

    def forward(self, x):
        bow_force, bow_velocity, bow_bridge_distance = (
            x["bow_force"],
            x["bow_velocity"],
            x["bow_bridge_distance"],
        )
        a = self._a()
        eta = bow_velocity  # approximation: no simulated string state available
        phi = torch.sqrt(2 * a) * torch.exp(-a * eta**2 + 0.5) * eta
        friction_signal = bow_force * phi

        features = torch.cat(
            [bow_force, bow_velocity, bow_bridge_distance, friction_signal], -1
        )
        ol = self.out_proj(self.in_mlp(features))
        ol = (
            ol + bow_bridge_distance * self.bridge_weight
        )  # structural bridge-distance bias

        return {"f0_hz": x["f0_hz"], "ol": ol}
