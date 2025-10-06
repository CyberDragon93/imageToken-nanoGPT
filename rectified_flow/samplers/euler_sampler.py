import torch
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from typing import Callable


class EulerSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int | None = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )

    def step(self, **model_kwargs):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)
        dtype = x_t.dtype
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)
        self.x_t = x_t + (t_next - t) * v_t
        self.x_t = self.x_t.to(dtype)
