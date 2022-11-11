import numpy as np


# ------------------------------------------------------------------------------------------------
# COMPONENTS
# ------------------------------------------------------------------------------------------------

EB = 0
ST = 1
H2 = 2
BZ = 3
ME = 4
TO = 5
EE = 6
H2O = 7

# ------------------------------------------------------------------------------------------------
# THERMODYNAMIC PROPERTIES
# ------------------------------------------------------------------------------------------------

# Coefficients in heat capacity equation Reid et al (1987)
A = np.array([-4.31e1, -2.825e1, 2.714e1, -3.392e1,
             1.925e1, -2.435e1, 3.806, 3.224e1])
B = np.array([7.072e-1, 6.159e-1, 9.274e-3, 4.739e-1,
             5.213e-2, 5.125e-1, 1.566e-1, 1.924e-3])
C = np.array([-4.811e-4, -4.023e-4, -1.381e-5, -3.017e-4,
             1.197e-5, -2.765e-4, -8.348e-5, 1.055e-5])
D = np.array([1.301e-7, 9.935e-8, 7.645e-9, 7.13e-8, -
             1.132e-8, 4.911e-8, 1.755e-8, -3.596e-9])

# Heat, Gibbs free energy, and Entropy of formation
HF298 = np.array([2.981e4, 1.475e5, 0, 8.298e4, -
                 7.49e4, 5.003e4, 5.234e4, -2.42e5])
GF298 = np.array([1.307e5, 2.139e5, 0, 1.297e5, -
                 5.087e4, 1.221e5, 6.816e4, -2.288e4])
SF298 = (np.array(HF298) - np.array(GF298)) / 298.15

# Returns varying coefficient on respective positions


def get_delta_coefficient(coef, positions, y):
    coef = np.array(coef)
    positions = np.array(positions)
    components = np.zeros(coef.shape, dtype=int)
    components[positions] = y
    return coef.dot(components)

# Returns output for several varying components


def get_delta_multiple_coefficient(coefs, positions, y):
    return np.array([get_delta_coefficient(coef, positions, y) for coef in coefs])

DELTA_R1 = get_delta_multiple_coefficient([A, B, C, D, HF298, GF298, SF298], [EB, ST, H2], [-1, 1, 1])
DELTA_R2 = get_delta_multiple_coefficient([A, B, C, D, HF298, GF298, SF298], [EB, BZ, EE], [-1, 1, 1])
DELTA_R3 = get_delta_multiple_coefficient([A, B, C, D, HF298, GF298, SF298], [EB, H2, TO, ME], [-1, -1, 1, 1])
DELTA_R4 = get_delta_multiple_coefficient([A, B, C, D, HF298, GF298, SF298], [ST, H2, TO, ME], [-1, -2, 1, 1])

def get_deltas(*args):
    mat = np.vstack(args)
    return mat.T


DELTA_A, DELTA_B, DELTA_C, DELTA_D, DELTA_HR298, DELTA_GF298, DELTA_SR298 = \
    get_deltas(DELTA_R1, DELTA_R2, DELTA_R3, DELTA_R4)

# ------------------------------------------------------------------------------------------------
# FLUIDDYNAMIC PROPERTIES
# ------------------------------------------------------------------------------------------------

MM = np.array([106.168, 104.152, 2.016, 78.114, 16.043, 92.141, 28.054, 18.015])  # kg/kmol
TC = np.array([617.2, 647.0, 32.2, 562.2, 190.4, 591.8, 282.4, 647.3])  # K
PC = np.array([36.0, 39.9, 13.0, 48.9, 46.0, 41.0, 50.4, 221.2])  # bar
SIGMA = np.array([np.nan, np.nan, 2.827, 5.349, 3.758, np.nan, 4.163, 2.641])  # Angstron
EK = np.array([np.nan, np.nan, 59.7, 412.3, 148.6, np.nan, 224.7, 809.1])  # K
DELTA_POT = np.zeros_like(MM)
DELTA_POT[H2O] = 1.0

# For Ethylbenzene, Styrene, Benzene, Methane, Toluene and Ethylene: Thodos Equation
# For H2 and H2O: Chapman-Enskog

HC = np.array([EB, ST, BZ, ME, TO, EE])
LOW = np.array([H2, H2O])
