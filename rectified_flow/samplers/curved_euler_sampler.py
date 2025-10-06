import torch
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.flow_components.interpolation_solver import AffineInterp
from typing import Callable


class CurvedEulerSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int | None = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
        interp_inference: AffineInterp | str = "natural",
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )
        if isinstance(interp_inference, str):
            if interp_inference.lower() == "natural":
                self.interp_inference = self.rectified_flow.interp
            else:
                self.interp_inference = AffineInterp(interp_inference)
        elif isinstance(interp_inference, AffineInterp):
            self.interp_inference = interp_inference
        else:
            raise ValueError(
                "interp_inference must be 'natural', an AffineInterp object, or a string specifying the interpolation method"
            )

    def step(self, **model_kwargs):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)
        dtype = v_t.dtype
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)

        interp = self.interp_inference.solve(t, x_t=x_t, dot_x_t=v_t)
        x_1_pred = interp.x_1
        x_0_pred = interp.x_0

        # interplate to find x_{t_next}
        self.x_t = self.interp_inference.solve(t_next, x_0=x_0_pred, x_1=x_1_pred).x_t
        self.x_t = self.x_t.to(dtype)
