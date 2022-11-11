import numpy as np
from styrene.thermodynamics import calc_delta_hr, calc_delta_sr, calc_delta_gibbs
from styrene.data import DELTA_A, DELTA_B, DELTA_C, DELTA_D, DELTA_HR298, DELTA_SR298

# Ea is in kJ/mol.K
# Basic kinetic constant from Arrhenius
def fun_kin(A, Ea, T, R=8.314):
    return A * np.exp(-Ea*1e3 / (R * T))


# Component labels to use in arrays
EB, ST, H2, BZ, ME, TO, EE, H2O = np.arange(8)

# Equilibrium constant:
# Result in [bar]
def Kp1(T, R=8.314):
    vHr = calc_delta_hr(
        T, DELTA_HR298[0], DELTA_A[0], DELTA_B[0], DELTA_C[0], DELTA_D[0])
    vSr = calc_delta_sr(
        T, DELTA_SR298[0], DELTA_A[0], DELTA_B[0], DELTA_C[0], DELTA_D[0])
    vGr = calc_delta_gibbs(T, vHr, vSr)
    return np.exp(-vGr / R / T)

# Thermal reactions:
# p in [bar], T in [K], result in [kmol/m**3.h]
def rt1(p, T):
    Keq = Kp1(T)
    return fun_kin(2.2215e16, 272.23, T) * (p[EB] - (p[ST] * p[H2]) / Keq)


def rt2(p, T):
    return fun_kin(2.4217e20, 352.79, T) * p[EB]


def rt3(p, T):
    return fun_kin(3.8224e17, 313.06, T) * p[EB]


def rt4(p, T):
    return 0

# On adsorption:
# Returns bar**-1
def fKad(T):
    return np.array([fun_kin(1.014e-5, -102.22, T),
                     fun_kin(2.678e-5, -104.56, T),
                     fun_kin(4.519e-7, -117.95, T)])

# Dimensionless result
def fnum_ad(p, Kad):
    return (1 + Kad[EB]*p[EB] + Kad[H2]*p[H2] + Kad[ST]*p[ST])**2

# Catalyst reactions:
# p in [bar], T in [K], result in [kmol/kg-cat.h]
def rc1(p, T):
    Keq = Kp1(T)
    Kad = fKad(T)
    k1 = fun_kin(4.594e9, 175.38, T)
    num = fnum_ad(p, Kad)
    return k1 * Kad[EB] * (p[EB] - (p[ST] * p[H2] / Keq)) / num


def rc2(p, T):
    Kad = fKad(T)
    k2 = fun_kin(1.060e15, 296.29, T)
    num = fnum_ad(p, Kad)
    return k2 * Kad[EB]*p[EB] / num


def rc3(p, T):
    Kad = fKad(T)
    # Original Lee 1e26, suggestion Dimian et.al (2019) 1e22
    k3 = fun_kin(1.246e26, 474.76, T)
    num = num = fnum_ad(p, Kad)
    return k3 * Kad[EB]*p[EB] * Kad[H2]*p[H2] / num


def rc4(p, T):
    Kad = fKad(T)
    k4 = fun_kin(8.024e10, 213.78, T)
    num = num = fnum_ad(p, Kad)
    return k4 * Kad[ST]*p[ST] * Kad[H2]*p[H2] / num

# Transport equation for reactants:
def ft_reactants(r, p, T, D, rhos, es, R=8.314e-2):
    """Second order derivatives in radial coordinates for components partial pressures.

    Args:
        r (float): Pellet radius in [m].
        p (array like of float): Partial pressure of components in [bar].
        T (float): Temperature in [K].
        D (array like of float): Diffusion coefficients of species in [m^2/h].
        rhos (float): Pellet solid density in [kg/m^3].
        es (float): Pellet void fraction.
        R (float, optional): Gases coefficients corresponding units. Defaults to 8.314e-2.

    Returns:
        array like of float: Second order derivatives of partial pressures.
    """

    pre = -(R * T) / D[:3]
    rr1 = es*rt1(p, T) + rhos*rc1(p, T)
    rr2 = es*rt2(p, T) + rhos*rc2(p, T)
    rr3 = es*rt3(p, T) + rhos*rc3(p, T)
    rr4 = es*rt4(p, T) + rhos*rc4(p, T)

    fteb_ = -rr1 - rr2 - rr3
    ftst_ = rr1 - rr4
    fth2_ = rr1 - 2*rr4

    return np.array([fteb_, ftst_, fth2_]) * pre.reshape([3, -1])

# Effectiveness:
def effective_reactions(r, p, T, rhos, es):

    rr1 = es*rt1(p, T) + rhos*rc1(p, T)
    rr2 = es*rt2(p, T) + rhos*rc2(p, T)
    rr3 = es*rt3(p, T) + rhos*rc3(p, T)
    rr4 = es*rt4(p, T) + rhos*rc4(p, T)

    return np.array([rr1, rr2, rr3, rr4])
