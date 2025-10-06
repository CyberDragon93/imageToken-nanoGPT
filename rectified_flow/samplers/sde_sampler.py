import torch
import warnings
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from typing import Callable


class SDESampler(Sampler):
    r"""Stochastic sampler for rectified flow with independent coupling and a Gaussian noise distribution (pi0).

    At each iteration, the sampler decomposes X_t into:

        X_t = alpha_t * X_1_pred + beta_t * X_0_pred.

    - If `noise_method=='stable'`, the noise on X_0_pred is refreshed as follows:

            X_t_noised = X_t - beta_t_noised * (X_0_pred - pi_0.mean()) + sqrt(beta_t**2 - (beta_t - beta_t_noised)**2) * (refresh_noise - pi0.mean()),

    where beta_t_noised is computed as:

            beta_t_noised = step_size * noise_scale(t) * beta_t(t)**noise_decay_rate(t),

    and is capped by beta_t:

            beta_t_noised = min(beta_t_noised, beta_t).

    - If `noise_method=='euler'`, the noise term is approximated using:

        sqrt(beta_t**2 - (beta_t - beta_t_noised)**2) ~= sqrt(2 * beta_t * beta_t_noised).


    Notes:
        When using both `noise_method='euler'` and `ode_method='euler'`, the method is equivalent to the Euler method for solving the SDE:

            dX_t = vt(X_t) dt - e_t * (X_0_pred(X_t) - pi_0.mean()) dt
                + sqrt(2 * beta_t * e_t) * sqrt(pi_0.cov()) * dW_t,

        where:

            e_t = beta_t_noised / step_size = noise_scale(t) * beta_t(t)**noise_decay_rate(t).
    """

    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int | None = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
        noise_scale: float | Callable = 1.0,
        noise_decay_rate: float | Callable = 1.0,
        noise_method: str = "stable",
        ode_method: str = "curved",
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )
        self.noise_scale = self._process_coeffs(noise_scale)
        self.noise_decay_rate = self._process_coeffs(noise_decay_rate)
        self.noise_method = noise_method
        self.ode_method = ode_method

        if not self.rectified_flow.independent_coupling:
            warnings.warn(
                "For the sampler to be theoretically correct, the coupling must be independent. Proceed at your own risk."
            )
        if not self.rectified_flow.is_pi_0_gaussian:
            raise ValueError(
                "pi_0 should be Gaussian (torch.distributions.Normal or torch.distributions.MultivariateNormal)."
            )

    @staticmethod
    def _process_coeffs(coeff):
        if isinstance(coeff, (torch.Tensor, int, float)):
            return lambda t: coeff
        elif callable(coeff):
            return coeff
        else:
            raise TypeError("coeff should be a float, int, torch.Tensor, or callable.")

    def step(self, **model_kwargs):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)
        step_size = t_next - t

        # Solve for x_0 and x_1 given x_t and v_t
        interp = self.rectified_flow.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        x_0 = interp.x_0
        x_1 = interp.x_1
        beta_t = interp.beta(t)

        # Part 1: Add noise

        # 1) Calculate beta_t_noised, the fraction of x_0 that will be noised
        beta_t_noised = (
            step_size * self.noise_scale(t) * beta_t ** self.noise_decay_rate(t)
        )
        # Clip beta_t_noised to beta_t, it's not meaningful to have beta_t_noised > beta_t
        if beta_t_noised > beta_t:
            beta_t_noised = beta_t

        refresh_noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        pi_0_mean = self.rectified_flow.pi_0.mean

        # 2) Remove beta_t_noised * x_0 and then add refreshed noise
        if self.noise_method.lower() == "stable":
            noise_std = (beta_t**2 - (beta_t - beta_t_noised) ** 2) ** 0.5
            langevin_term = -beta_t_noised * (x_0 - pi_0_mean) + noise_std * (
                refresh_noise - pi_0_mean
            )
        # this is the taylor approximation of the stable method when beta_t_noised is small, and corresponds to Euler method for the langevin dynamics
        elif self.noise_method.lower() == "euler":
            noise_std = (2 * beta_t * beta_t_noised) ** 0.5
            langevin_term = -beta_t_noised * (x_0 - pi_0_mean) + noise_std * (
                refresh_noise - pi_0_mean
            )

        else:
            raise ValueError(f"Unknown noise_method: {self.noise_method}")

        # print(f"t = {t:.5f}, coeff = {coeff:.5f}, noise_std = {noise_std:.5f}")

        x_t_noised = x_t + langevin_term

        # Advance time using the specified ODE method
        if self.ode_method.lower() == "euler":
            # standard Euler method
            x_t_next = x_t_noised + step_size * v_t

        elif self.ode_method.lower() == "curved":
            # Curved Euler method, following the underlying interpolation curve
            # a. Get x_0_noised from x_t_noised and x_1
            x_0_noised = self.rectified_flow.interp.solve(
                t=t, x_t=x_t_noised, x_1=x_1
            ).x_0

            # b. Interpolate to get x_t_next given x_0_noised and x_1
            x_t_next = self.rectified_flow.interp.solve(
                t=t_next, x_0=x_0_noised, x_1=x_1
            ).x_t

        else:
            raise ValueError(f"Unknown ode_method: {self.ode_method}")

        # Update the current sample
        self.x_t = x_t_next
