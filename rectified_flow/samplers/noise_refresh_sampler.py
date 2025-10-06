import torch
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from typing import Callable


class NoiseRefreshSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int | None = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
        noise_replacement_rate: Callable = lambda t: 0.5,
        euler_method: str = "curved",
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )
        self.noise_replacement_rate = noise_replacement_rate
        self.euler_method = euler_method
        assert (
            self.rectified_flow.independent_coupling
            and self.rectified_flow.is_pi_0_zero_mean_gaussian
        ), "pi0 must be a zero mean gaussian and must use indepdent coupling"

    def step(self, **model_kwargs):
        """Perform a single step of the sampling process."""
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)
        dtype = v_t.dtype
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)

        # Given x_t and dot_x_t = vt, find the corresponding endpoints x_0 and x_1
        self.rectified_flow.interp.solve(t, x_t=x_t, dot_x_t=v_t)
        x_1_pred = self.rectified_flow.interp.x_1
        x_0_pred = self.rectified_flow.interp.x_0

        # Randomize x_0_pred by replacing part of it with new noise
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        noise = noise.to(torch.float32)

        noise_replacement_factor = self.noise_replacement_rate(t)
        x_0_pred_refreshed = (
            1 - noise_replacement_factor**2
        ) ** 0.5 * x_0_pred + noise * noise_replacement_factor

        if self.euler_method.lower() == "curved":
            # Interpolate to find x_t at t_next
            self.rectified_flow.interp.solve(
                t_next, x_0=x_0_pred_refreshed, x_1=x_1_pred
            )
            self.x_t = self.rectified_flow.interp.x_t

        elif self.euler_method.lower() == "straight":
            # if we use the x + (t_next - t) * v_t as the update for the deterministic part
            self.rectified_flow.interp.solve(t, x_0=x_0_pred_refreshed, x_1=x_1_pred)
            x_t_refreshed = self.rectified_flow.interp.x_t
            v_t_refreshed = (
                self.rectified_flow.interp.dot_x_t
            )  # we can also use v_t here, it will be effectively using a smaller noise_refresh_rate
            x_t_next = x_t_refreshed + (t_next - t) * v_t_refreshed
            self.x_t = x_t_next

        self.x_t = self.x_t.to(dtype)
