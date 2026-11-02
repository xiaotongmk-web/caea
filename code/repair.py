"""Class-marginal repair with explicit feasibility checks (NumPy only)."""
import numpy as np


def repair_class_marginal(probabilities, target, fraction=1.0):
    """Repair (..., classes, responses_per_class) probability arrays.

    An empty class cannot receive positive target mass. Reject that case
    instead of silently returning a sub-probability distribution.
    """
    p = np.asarray(probabilities, dtype=float)
    tau = np.asarray(target, dtype=float)
    if p.ndim < 2 or tau.ndim < 1 or tau.shape[-1] != p.shape[-2]:
        raise ValueError('Expected matching class axes for probabilities and target')
    if not np.isscalar(fraction) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Repair fraction must be a finite scalar in [0, 1]')
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(tau)):
        raise ValueError('Probabilities and target must be finite')
    if np.any(p < 0) or np.any(tau < 0):
        raise ValueError('Probabilities and target must be nonnegative')
    if not np.allclose(p.sum(axis=(-2, -1)), 1., atol=1e-12, rtol=1e-10):
        raise ValueError('Response probabilities must sum to one')
    if not np.allclose(tau.sum(axis=-1), 1., atol=1e-12, rtol=1e-10):
        raise ValueError('Target probabilities must sum to one')
    marginal, tau = np.broadcast_arrays(p.sum(axis=-1), tau)
    wanted = (1 - fraction) * marginal + fraction * tau
    if np.any((marginal == 0) & (wanted > 0)):
        raise ValueError('Infeasible repair: positive target mass on an empty class')
    conditional = np.divide(p, marginal[..., None],
                            out=np.zeros(np.broadcast_shapes(p.shape, marginal[..., None].shape)),
                            where=marginal[..., None] > 0)
    return wanted[..., None] * conditional
