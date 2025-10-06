import torch
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from typing import Callable


class OverShootingSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int | None = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
        c: int = 1.0,
        overshooting_method: str | Callable = "t + dt * (1 - t)",
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )

        self.c = c
        # Define overshooting method
        if callable(overshooting_method):
            self.overshooting = overshooting_method
        elif isinstance(overshooting_method, str):
            try:
                self.overshooting = eval(f"lambda t, dt: {overshooting_method}")
            except SyntaxError:
                raise ValueError(f"Invalid overshooting method: {overshooting_method}")
        else:
            raise ValueError(
                "Invalid overshooting method provided. Must be a string or callable."
            )

        # Ensure rf meets required conditions
        if not (
            self.rectified_flow.is_pi_0_zero_mean_gaussian
            and self.rectified_flow.independent_coupling
        ):
            raise ValueError(
                "pi0 must be a zero-mean Gaussian distribution, and the coupling must be independent."
            )

    def step(self, **model_kwargs):
        """Perform a single overshooting step."""
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)
        dtype = v_t.dtype
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)

        alpha = self.rectified_flow.interp.alpha
        beta = self.rectified_flow.interp.beta

        # Calculate overshoot time and enforce constraints
        t_overshoot = min(self.overshooting(t_next, (t_next - t) * self.c), 1.0)
        if t_overshoot < t_next:
            raise ValueError("t_overshoot cannot be smaller than t_next.")

        # Advance to t_overshoot using ODE
        x_t_overshoot = x_t + (t_overshoot - t) * v_t

        # Apply noise to step back to t_next
        a_t = alpha(t_next) / alpha(t_overshoot)
        b_t = (beta(t_next) ** 2 - (a_t * beta(t_overshoot)) ** 2) ** 0.5
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        noise = noise.to(torch.float32)

        self.x_t = x_t_overshoot * a_t + noise * b_t
        self.x_t = self.x_t.to(dtype)
