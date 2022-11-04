import numpy as np
from styrene.data import HC, LOW


# ------------------------------------------------------------------------------------------------
# VISCOSITY OF INDIVIDUAL COMPONENTS
# ------------------------------------------------------------------------------------------------

def calc_mu_thodos(T, Tc, Pc, Mm):
    """
    Returns the viscosity of pure component by Thodos equation

    Parameters
    ----------
    T : float or int
        Temperature in K.
    Tc : float or int
        Critical temperature in K.
    Pc : float or int
        Critical pressure in bar.
    Mm : float or int
        Molar mass in kg/kmol.

    Returns
    -------
    mu: float
        Viscosity of pure component in micro Poise.

    """
    eps = Tc**(1/6)*Mm**(-1/2)*Pc**(-2/3)
    Tr = T / Tc
    return (4.61*Tr**0.618-2.04*np.exp(-0.449*Tr)+1.94*np.exp(-4.058*Tr)+0.1)/eps


def calc_mu_chap_ens(T, Mm, sigma, potV):
    """
    Returns the viscosity of pure component by Chapman-Enskog equation

    Parameters
    ----------
    T : float or int
        Temperature in K.
    Mm : float or int
        Molar mass in kg/kmol.
    sigma : float or int
        Hard-sphere diameter in Angstron.
    potV : float or int
        Collision integral, 1 if molecules do not interact.

    Returns
    -------
    mu: float
        Viscosity of pure component in micro Poise.

    """
    return 26.69*(Mm * T)**0.5 / (sigma**2 * potV)


def calc_ci_lj(T, ek, A=1.16145, B=0.14874, C=0.52487, D=0.77320, E=2.16178, F=2.43787):
    """
    Returns the collision integral from Lennard-Jones equation

    Parameters
    ----------
    T : float or int
        Temperature in K.
    ek : float or int
        Value in K.
    A : float, optional
        Equation variable. The default is 1.16145.
    B : float, optional
        Equation variable. The default is 0.14874.
    C : float, optional
        Equation variable. The default is 0.52487.
    D : float, optional
        Equation variable. The default is 0.77320.
    E : float, optional
        Equation variable. The default is 2.16178.
    F : float, optional
        Equation variable. The default is 2.43787.

    Returns
    -------
    float
        Lennard-Jones collision integral.

    """
    Tstar = T / ek
    return A / (Tstar**B) + C * np.exp(-D * Tstar) + E * np.exp(-F * Tstar)


def calc_pot_sm(T, ek, delta):
    """
    Returns the collision integral from Stockmayer equation

    Parameters
    ----------
    T : float or int
        Temperature in K.
    ek : float or int
        Value in K.
    delta : float or int
        Dimensionless variable. Use 1 for H2O.

    Returns
    -------
    float
        Collision integral from Stockmayer equation.

    """
    Tstar = T/ek
    potLJ = calc_ci_lj(T, ek)
    return potLJ + 0.2 * delta ** 2 / Tstar

# ------------------------------------------------------------------------------------------------
# VISCOSITY OF MIXTURE
# ------------------------------------------------------------------------------------------------

def calc_phi_matrix(mu, Mm):
    """
    Returns the phi matrix used to obtain viscosity of mixture using Wilke's approximation

    Parameters
    ----------
    mu : 1d array like
        Viscosity of pure components. Units might be arbitraty be suggest micro Poise.
    Mm : 1d array like
        Molar mass of components, kg/kmol.

    Returns
    -------
    phi : 2d array
        phi matrix used to obtain viscosity of mixture.

    """
    mu_ratio = np.atleast_2d(mu).reshape([-1, 1]) / np.atleast_2d(mu)
    Mm_ratio = np.atleast_2d(Mm).reshape([-1, 1]) / np.atleast_2d(Mm)
    
    phi = (1 + mu_ratio**0.5 * (1 / Mm_ratio)**0.25) ** 2\
        / ((8 * (1 + Mm_ratio)) ** 0.5)
        
    return phi


def calc_mu_components(T, Mm, Tc, Pc, sigma, ek, delta):
    """Obtain individual viscosity

    Parameters
    ----------
    T : float
        Temperature, K.
    Mm : 1d array
        Molar mass of components, kg/kmol.
    Tc : 1d array
        Critical temperature of compoents, K.
    Pc : 1d array
        Critical pressure of components, bar.
    sigma : 1d array
        Coefficients in Chapman-Enskog equation.
    ek : 1d array
        Coefficients in Stockmayer equation.
    delta : 1d array
        Coefficients in Stockmayer equation.

    Returns
    -------
    1d array
        Individual viscosities.
    """
    
    mu = np.zeros_like(Mm)
    
    # Hydrocarbons
    mu[HC] = calc_mu_thodos(T, Tc[HC], Pc[HC], Mm[HC])
    
    # Low mass
    potV = calc_pot_sm(T, ek[LOW], delta[LOW])
    mu[LOW] = calc_mu_chap_ens(T, Mm[LOW], sigma[LOW], potV)
    
    return mu


def calc_mu_mist(T, y, Mm, Tc, Pc, sigma, ek, delta):
    """Obtain viscosity of mixture.

    Parameters
    ----------
    T : float
        Temperature, K.
    y : 1d array
        Molar fractions.
    Mm : 1d array
        Molar mass of components, kg/kmol.
    Tc : 1d array
        Critical temperature of compoents, K.
    Pc : 1d array
        Critical pressure of components, bar.
    sigma : 1d array
        Coefficients in Chapman-Enskog equation.
    ek : 1d array
        Coefficients in Stockmayer equation.
    delta : 1d array
        Coefficients in Stockmayer equation.

    Returns
    -------
    float
        Viscosity of mixture.
    """
    mu = calc_mu_components(T, Mm, Tc, Pc, sigma, ek, delta)
    phi = calc_phi_matrix(mu, Mm)
    return calc_mu_from_components(mu, y, phi)


def calc_mu_from_components(mu, y, phi):
    """
    Returns the viscosity of the mixture using Wilke's approximation.

    Parameters
    ----------
    mu : 1d array like
        Viscosity of pure components. Units might be arbitraty be suggest micro Poise.
    y : 1d array like
        Fraction of each component in the phase.
    phi : 2d array
        phi matrix used to obtain viscosity of mixture.

    Returns
    -------
    mu_mix : float
        Viscosity of the mixture, if coherent units are used in obtaining phi, it has the same units as mu.
        Suggestion: use micro Poise for viscosity and kg/kmol for molar mass.
    """
    y_mu = np.array(y) * np.array(mu)
    ratio_ = np.array(phi).dot(np.array(y))

    mu_mix = (y_mu / ratio_).sum()

    return mu_mix

# ------------------------------------------------------------------------------------------------
# PRESSURE DROP
# ------------------------------------------------------------------------------------------------

def calc_pressure_drop(G, rhog, mu, Ac, dp, rhob, eg, a_erg=1.75, b_erg=150):
    """
    Returns dP/dW at a catalyst bed according to Ergun equation.
    Alternative for radial flow: a_erg=1.28,b_erg=458

    Parameters
    ----------
    G : float or int
        Superficial mass velocity in kg/m2.h.
    rhog : float or int
        Gas density in kg/m3.
    mu : float or int
        Viscosity of the mixture in micro Poise.
    Ac : float or int
        Transversal area of the bed in m2.
    dp : float
        Pellet diameter in m.
    rhob : float or int
        Bed density in kg/m3.
    eg : float
        Void fraction of the bed (bulk value).
    a_erg : float, optional
        Parameter of the Ergun equation. The default is 1.75.
    b_erg : float, optional
        Parameter of the Ergun equation. The default is 150.

    Returns
    -------
    float
        Pressure drop in pressure unit/kg.

    """
    G = G/3600
    mu = mu * 1e-7
    return (1 / (Ac*rhob)) * (G / (rhog*dp))\
        * ((1-eg) / eg**3) * (b_erg*(1-eg) * mu/dp + a_erg*G) * 1e-5
        

def calc_pressure_drop_ideal_gas(F, T, P, rhob, dp, eg, Ac, Mm, mu, R=8.314e-2,
                             a_erg=1.75, b_erg=150, print_Re=False):
    """
    Returns dP/dW at a catalyst bed according to Ergun equation considering ideal gas.
    Alternative for radial flow: a_erg=1.28,b_erg=458

    Parameters
    ----------
    F : 1d array
        Molar flow in kmol/h.
    T : float or int
        Temperature in K.
    P : float or int
        Pressure on the same unit as the result.
        Suggestion: use bar.
    rhob : float or int
        Bed density in kg/m3.
    dp : float
        Pellet diameter in m.
    eg : float
        Void fraction of the bed (bulk value).
    Ac : float or int
        Transversal area of the bed in m2.
    Mm : 1d array like
        Molar mass of each component in kg/kmol.
    mu : float or int
        Viscosity of the mixture in micro Poise.
    R : float, optional
        Ideal gases constant SI/10^-5 to convert Pa to bar. The default is 8.314e-2.
    a_erg : float, optional
        Parameter of the Ergun equation. The default is 1.75.
    b_erg : float, optional
        Parameter of the Ergun equation. The default is 150.
    print_Re : bool, optional
        Either to pront of not Reynolds number. The default is False.

    Returns
    -------
    float
        Pressure drop in pressure unit/kg.

    """

    mu = mu * 1e-7
    F = np.array(F) / 3600
    y = np.array(F / F.sum())
    Mm = np.array(Mm)
    rhog = (y.T.dot(Mm)) * P / R / T
    G = (np.array(F).T.dot(Mm)) / Ac
    if print_Re:
        print('Re/(1-eg):', G * dp / mu / (1-eg))
    return (1 / (Ac*rhob)) * (G / (rhog*dp))\
        *((1-eg) / eg**3) * (b_erg * (1-eg) * mu/dp + a_erg*G) *1e-5
        