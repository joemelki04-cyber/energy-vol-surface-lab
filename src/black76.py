from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm

OptionType = Literal["C", "P"]


def _validate_inputs(F: float, K: float, T: float, r: float, sigma: float) -> None:
    if F <= 0:
        raise ValueError("Futures price F must be positive.")
    if K <= 0:
        raise ValueError("Strike K must be positive.")
    if T < 0:
        raise ValueError("Time to expiry T cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")


def black76_price(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """
    Black-76 price for European options on futures.

    Parameters
    ----------
    F : float
        Futures price
    K : float
        Strike
    T : float
        Time to expiry in years
    r : float
        Continuously compounded risk-free rate
    sigma : float
        Volatility
    option_type : {"C", "P"}
        "C" for call, "P" for put
    """
    option_type = option_type.upper()
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    _validate_inputs(F, K, T, r, sigma)

    discount = math.exp(-r * T)

    if T == 0 or sigma == 0:
        intrinsic = max(F - K, 0.0) if option_type == "C" else max(K - F, 0.0)
        return discount * intrinsic

    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "C":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_intrinsic_discounted(
    F: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> float:
    option_type = option_type.upper()
    if option_type == "C":
        return math.exp(-r * T) * max(F - K, 0.0)
    if option_type == "P":
        return math.exp(-r * T) * max(K - F, 0.0)
    raise ValueError("option_type must be 'C' or 'P'.")
