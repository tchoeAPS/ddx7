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
        use_friction=True,
        bound_controls=False,
        velocity_k=2.0,
        bridge_range=(0.02, 0.2),
    ):
        super().__init__()
        # use_friction=False zeroes the friction feature but keeps every layer,
        # as a control for whether the friction curve itself matters.
        self.use_friction = use_friction
        # bound_controls=True squashes the decoder's raw outputs into plausible
        # ranges: force >= 0, bridge distance in bridge_range (fraction of string
        # length), |velocity| <= velocity_k / sqrt(2 * friction_a) so it stays
        # where the friction curve has shape (its peak is at 1 / sqrt(2a)).
        # The velocity bound uses the *initial* a: tying it to the learned a
        # would cancel a out of the friction curve entirely, and would give the
        # friction and no-friction models different velocity ranges.
        # Plain floats, no parameters or buffers, so unbounded checkpoints still load.
        self.bound_controls = bound_controls
        self.velocity_k = velocity_k
        self.velocity_bound = velocity_k / float(friction_a * 2) ** 0.5
        self.bridge_range = tuple(bridge_range)
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

    def controls(self, x):
        """The (force, velocity, bridge distance) the physics layer actually uses."""
        bow_force, bow_velocity, bow_bridge_distance = (
            x["bow_force"],
            x["bow_velocity"],
            x["bow_bridge_distance"],
        )
        if self.bound_controls:
            lo, hi = self.bridge_range
            bow_force = nn.functional.softplus(bow_force)
            bow_velocity = torch.tanh(bow_velocity) * self.velocity_bound
            bow_bridge_distance = lo + (hi - lo) * torch.sigmoid(bow_bridge_distance)
        return bow_force, bow_velocity, bow_bridge_distance

    def friction_curve(self, eta):
        a = self._a()
        return torch.sqrt(2 * a) * torch.exp(-a * eta**2 + 0.5) * eta

    def forward(self, x):
        bow_force, bow_velocity, bow_bridge_distance = self.controls(x)
        if self.use_friction:
            eta = bow_velocity  # approximation: no simulated string state available
            friction_signal = bow_force * self.friction_curve(eta)
        else:
            friction_signal = torch.zeros_like(bow_force)

        features = torch.cat(
            [bow_force, bow_velocity, bow_bridge_distance, friction_signal], -1
        )
        ol = self.out_proj(self.in_mlp(features))
        ol = (
            ol + bow_bridge_distance * self.bridge_weight
        )  # structural bridge-distance bias

        return {"f0_hz": x["f0_hz"], "ol": ol}
