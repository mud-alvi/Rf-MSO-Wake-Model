import numpy as np

AMBIENT_TI = 0.077  # horns Rev wake-model bench mark
TI_SENSITIVITY = [0.06, 0.077, 0.10, 0.14]

FATIGUE_EXPONENT = 4.0
FATIGUE_SENSITIVITY = [3.0, 4.0, 5.0]


def fatigue_index(load_proxy_values, weights=None, exponent=FATIGUE_EXPONENT):
    """Compute a relative fatigue index for one turbine.

    D* = Σ_t w_t × (L*_t)^m

    If every record represents one hour, equal hourly weights are used.
    If the wind data contains only four daily samples, this remains a sampled
    exposure estimate rather than an absolute damage metric.
    """
    load_proxy_values = np.asarray(load_proxy_values, dtype=float)

    if weights is None:
        weights = np.ones_like(load_proxy_values, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != load_proxy_values.shape:
            raise ValueError("weights must match load_proxy_values shape")

    return np.sum(weights * np.power(load_proxy_values, exponent))
