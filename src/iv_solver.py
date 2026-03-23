from __future__ import annotations

from scipy.optimize import brentq

from black76 import black76_price, black76_intrinsic_discounted


def implied_vol_black76(
    premium: float,
    F: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    vol_lower: float = 1e-6,
    vol_upper: float = 5.0,
) -> float:
    """
    Solve implied volatility under Black-76 using Brent's method.
    """
    if premium < 0:
        raise ValueError("Option premium cannot be negative.")

    intrinsic = black76_intrinsic_discounted(F, K, T, r, option_type)

    if premium < intrinsic - 1e-10:
        raise ValueError(
            f"Premium {premium:.6f} is below discounted intrinsic value {intrinsic:.6f}."
        )

    if T == 0:
        if abs(premium - intrinsic) < 1e-10:
            return 0.0
        raise ValueError("Cannot infer implied volatility at expiry from non-intrinsic premium.")

    def objective(sigma: float) -> float:
        return black76_price(F, K, T, r, sigma, option_type) - premium

    low_val = objective(vol_lower)
    high_val = objective(vol_upper)

    if abs(low_val) < 1e-12:
        return vol_lower

    if low_val * high_val > 0:
        raise ValueError(
            "Could not bracket implied volatility root. "
            "Check whether the premium is valid or increase vol_upper."
        )

    return float(brentq(objective, vol_lower, vol_upper))
