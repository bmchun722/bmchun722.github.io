"""Toy free-energy surfaces for coupled ion-electron transfer (CIET).

Based on Fraggedakis et al., "Theory of coupled ion-electron transfer
kinetics", Electrochim. Acta (2021), Sec. III.C, eqs. (6a)-(10):

    mu_O(x, xi) = mu_O0 + (kappa/2) (x - x_O)^2 + f_O(xi)
    mu_R(x, xi) = mu_R0 + (kappa/2) (x - x_R)^2 + f_R(xi)

x  : solvent polarization coordinate (Marcus coordinate)
xi : ionic coordinate (distance the ion moves); origin chosen at xi_O = 0

The paper only constrains f_O and f_R (eqs. 7, 9, 10). For visualization we
choose the simplest form satisfying eqs. (7) and (9) with a minimum at each
equilibrium point:

    f_O(xi) = dE_IT * xi^2 / xi_R^2
    f_R(xi) = dE_IT * (xi - xi_R)^2 / xi_R^2

This is our choice, not the paper's. Eq. (10) then holds exactly only at
eta = 0: at the saddle f_O = dE_IT t^2 and f_R = dE_IT (1 - t)^2, which differ
once the driving force shifts t away from 1/2 (the paper allows this, p. 6).

Driving force: mu_R0 - mu_O0. It equals e*eta only with the electron chemical
potential mu_e held at its reference (paper eq. 25: e*eta = mu_R - mu_O - mu_e).
There is no electron energy (Fermi-level) integration in this model.

Sweep variable: s = dE_IT / lam, with lam + dE_IT = C held fixed so the
overall energy scale stays constant. s is dimensionless; the paper discusses
the limits via kappa / dE_IT, so s is roughly the inverse of that ratio.
s << 1 is the lam >> dE_IT (ECIT) limit, s >> 1 the dE_IT >> lam (ICET) limit.

Caveat: with quadratic f and equal kappa, the saddle barrier is
(C + Delta)^2 / (4 C) with Delta = mu_R0 - mu_O0, i.e. Marcus kinetics with
total reorganization energy C for every s. Changing s only changes which
coordinate (x or xi) carries the barrier, not the rate law.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIETParams:
    """All parameters of the two surfaces (energies in arbitrary units)."""

    lam: float      # reorganization energy, lam = (kappa/2) (x_O - x_R)^2
    dE_IT: float    # ion-transfer (ion-electron separation) penalty
    x_O: float = -1.0
    x_R: float = 1.0
    xi_R: float = 2.0
    mu_O0: float = 0.0
    mu_R0: float = 0.0  # mu_R0 - mu_O0 plays the role of the driving force

    @property
    def kappa(self) -> float:
        return 2.0 * self.lam / (self.x_O - self.x_R) ** 2


def params_from_s(s: float, C: float = 2.0, **kwargs) -> CIETParams:
    """Split a fixed total C into lam and dE_IT with ratio s = dE_IT / lam."""
    if s <= 0:
        raise ValueError("s must be positive")
    lam = C / (1.0 + s)
    dE_IT = C * s / (1.0 + s)
    return CIETParams(lam=lam, dE_IT=dE_IT, **kwargs)


def mu_O(x, xi, p: CIETParams):
    """Oxidized-state surface. Accepts scalars or NumPy arrays."""
    return (p.mu_O0
            + 0.5 * p.kappa * (x - p.x_O) ** 2
            + p.dE_IT * xi ** 2 / p.xi_R ** 2)


def mu_R(x, xi, p: CIETParams):
    """Reduced-state surface. Accepts scalars or NumPy arrays."""
    return (p.mu_R0
            + 0.5 * p.kappa * (x - p.x_R) ** 2
            + p.dE_IT * (xi - p.xi_R) ** 2 / p.xi_R ** 2)


def intersection_x(xi, p: CIETParams):
    """x on the line mu_O = mu_R for a given xi (the line is straight here).

    mu_O - mu_R = (mu_O0 - mu_R0)
                  + (kappa/2) (x_R - x_O) (2x - x_O - x_R)
                  + (dE_IT / xi_R) (2 xi - xi_R)  = 0
    """
    d = p.x_R - p.x_O
    rhs = (p.mu_O0 - p.mu_R0) + p.dE_IT / p.xi_R * (2.0 * xi - p.xi_R)
    return 0.5 * (p.x_O + p.x_R) - rhs / (p.kappa * d)


def saddle_point(p: CIETParams):
    """Lowest point on the intersection line (the CIET transition state).

    With equal curvatures the saddle lies on the straight segment joining the
    two minima, at fraction t = 1/2 + (mu_R0 - mu_O0) / (2 (lam + dE_IT)).
    Returns (x, xi, mu).
    """
    t = 0.5 + (p.mu_R0 - p.mu_O0) / (2.0 * (p.lam + p.dE_IT))
    x = p.x_O + t * (p.x_R - p.x_O)
    xi = t * p.xi_R
    return x, xi, mu_O(x, xi, p)


if __name__ == "__main__":
    # Sanity checks with known answers.

    # 1) s = 1, C = 2  ->  lam = dE_IT = 1
    p = params_from_s(1.0, C=2.0)
    assert np.isclose(p.lam, 1.0) and np.isclose(p.dE_IT, 1.0)

    # 2) Hand calculation from the coaching session:
    #    kappa = 2, x_O = 0, dE_IT = 1, xi_R = 2, mu_O0 = 0  ->  mu_O(1, 1) = 1.25
    #    kappa = 2 with x_O = 0 needs lam = (2/2) (0 - x_R)^2 -> x_R = 1, lam = 1
    p_hand = CIETParams(lam=1.0, dE_IT=1.0, x_O=0.0, x_R=1.0, xi_R=2.0)
    assert np.isclose(p_hand.kappa, 2.0)
    assert np.isclose(mu_O(1.0, 1.0, p_hand), 1.25)

    # 3) Constraints (7) and (9) for several s
    for s in (0.1, 1.0, 10.0):
        p = params_from_s(s)
        assert np.isclose(mu_O(p.x_O, 0.0, p), p.mu_O0)                  # eq. 7
        assert np.isclose(mu_R(p.x_R, p.xi_R, p), p.mu_R0)               # eq. 7
        assert np.isclose(mu_O(p.x_O, p.xi_R, p) - p.mu_O0, p.dE_IT)     # eq. 9
        assert np.isclose(mu_R(p.x_R, 0.0, p) - p.mu_R0, p.dE_IT)        # eq. 9

        # 4) Saddle point lies on the intersection and is its minimum
        xs, xis, ms = saddle_point(p)
        assert np.isclose(mu_O(xs, xis, p), mu_R(xs, xis, p))
        xi_line = np.linspace(-0.5, 2.5, 2001)
        x_line = intersection_x(xi_line, p)
        assert np.isclose(intersection_x(xis, p), xs)
        assert np.all(mu_O(x_line, xi_line, p) >= ms - 1e-12)

    # 5) Same with a driving force (eta != 0): saddle shifts but stays minimal
    for s in (0.1, 1.0, 10.0):
        p = params_from_s(s, mu_R0=-0.5)
        xs, xis, ms = saddle_point(p)
        assert np.isclose(mu_O(xs, xis, p), mu_R(xs, xis, p))   # on the line
        assert np.isclose(intersection_x(xis, p), xs)
        xi_line = np.linspace(-0.5, 2.5, 2001)
        assert np.all(mu_O(intersection_x(xi_line, p), xi_line, p) >= ms - 1e-12)

        # 6) Barrier is (C + Delta)^2 / (4C), independent of s (see module caveat)
        C, delta = p.lam + p.dE_IT, p.mu_R0 - p.mu_O0
        assert np.isclose(ms - p.mu_O0, (C + delta) ** 2 / (4 * C))

    for s in (0.1, 1.0, 10.0):
        p = params_from_s(s)
        print(f"s={s:5.1f}  lam={p.lam:.3f}  dE_IT={p.dE_IT:.3f}  "
              f"kappa={p.kappa:.3f}  saddle={tuple(round(v, 3) for v in saddle_point(p))}")
    print("all sanity checks passed")
