# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:50:05 2020

@author: bruno
"""
import numpy as np

def heat_capacity(a, b, c, d, T):
    """
    Returns the heat capacity of one component in kJ/kmol at temperature T in K using Reid equation.

    Parameters
    ----------
    a : float
        Coefficient from Reid.
    b : float
        Coefficient from Reid.
    c : float
        Coefficient from Reid.
    d : float
        Coefficient from Reid.
    T : float or int
        Temperature in K.

    Returns
    -------
    Cp: float
        Heat capacity of one component in kJ/kmol.

    """
    return a + b*T + c*T**2 + d*T**3

def f_vcoef(coef, species_list, eq_coef_list):
    """
    Returns variation of coefficient or state property in a reacting system.

    Parameters
    ----------
    coef : 1d array or list
        Property or coefficient of all components in the reacting system.
    species_list : 1d array
        Contains int of the positions of each component in the reaction in the list of coefs.
    eq_coef_list : int
        Stoichiometric coefficients of the species of species_list in the reacting system.

    Returns
    -------
    res: float
        Variation of the property or coefficient in the reacting system.

    """
    _dim = len(species_list)
    return sum([coef[species_list[i]]*eq_coef_list[i] for i in range(0,_dim)])


def f_vabcd(abcd, species_list, eq_coef_list):
    """
    Returns variation of coefficients or state properties in a reacting system.

    Parameters
    ----------
    abcd : 2d array
        Property or coefficient of all components in the reacting system. Each line corresponds to one property.
    species_list : 1d array
        Contains int of the positions of each component in the reaction in the list of coefs.
    eq_coef_list : int
        Stoichiometric coefficients of the species of species_list in the reacting system.

    Returns
    -------
    res : 1d array
        Variation of the properties and/or coefficients in the reacting system.

    """
    _dim = len(abcd)
    res = np.zeros(_dim)
    j = 0
    for coef in abcd:
        res[j] = f_vcoef(coef, species_list, eq_coef_list)
        j += 1
    return res

'''import pandas as pd
df_export = pd.DataFrame([va,vb,vc,vd,vHr_298],index=['va','vb','vc','vd','vHf_298'],columns=['1','2','3','4'])
with pd.ExcelWriter('Styrene_Data'+'.xlsx') as writer:
    df_export.to_excel(writer, sheet_name='Heat_capacities')'''

def get_vHr(T, vHr_298, va, vb, vc, vd):
    """
    Returns the heat of reaction in kJ/kmol.

    Parameters
    ----------
    T : float or int
        Temperature in K.
    vHr_298 : 1d array, float or int
        Contains heat of reaction at 298.15K.
    va : 1d array, float or int
        Contains delta coefficient of the reaction(s).
    vb : 1d array, float or int
        Contains delta coefficient of the reaction(s).
    vc : 1d array, float or int
        Contains delta coefficient of the reaction(s).
    vd : 1d array, float or int
        Contains delta coefficient of the reaction(s).

    Returns
    -------
    1d array, float or int
        Contains heat of reaction at T in kJ/kmol.

    """
    return np.array(vHr_298) + np.array(va)*(T-298.15) + np.array(vb)/2*(T**2-298.15**2)\
        + np.array(vc)/3*(T**3-298.15**3) + np.array(vd)/4*(T**4-298.15**4)


def get_HfT(T, Hf_298, a, b, c, d):
    """
    Returns the heat of formation in kJ/kmol.

    Parameters
    ----------
    T : float or int
        Temperature in K.
    Hf_298 : 1d array, float or int
        Contains heat of formation at 298.15K.
    a : 1d array, float or int
        Contains a coefficient of the heat capacity.
    b : 1d array, float or int
        Contains a coefficient of the heat capacity.
    c : 1d array, float or int
        Contains a coefficient of the heat capacity.
    d : 1d array, float or int
        Contains a coefficient of the heat capacity.

    Returns
    -------
    1d array, float or int
        Contains heat of formation at T in kJ/kmol.

    """
    return np.array(Hf_298) + np.array(a)*(T-298.15) + np.array(b)/2*(T**2-298.15**2)\
        + np.array(c)/3*(T**3-298.15**3) + np.array(d)/4*(T**4-298.15**4)


def get_vSr(T, vSr_298, va, vb, vc, vd):
    """
    Returns the entropy of reaction in kJ/kmol.

    Parameters
    ----------
    T : float or int
        Temperature in K.
    vSr_298 : 1d array, float or int
        Contains entropy of reaction at 298.15K.
    va : 1d array, float or int
        Contains delta coefficient of the reaction(s).
    vb : 1d array, float or int
        Contains delta coefficient of the reaction(s).
    vc : 1d array, float or int
        Contains delta coefficient of the reaction(s).
    vd : 1d array, float or int
        Contains delta coefficient of the reaction(s).

    Returns
    -------
    1d array, float or int
        Contains entropy of reaction at T in kJ/kmol.K.

    """
    return np.array(vSr_298) + np.array(va)*np.log(T/298.15) + np.array(vb)*(T-298.15)\
        + np.array(vc)/2*(T**2-298.15**2) + np.array(vd)/3*(T**3-298.15**3)

def get_vGr(T, vHr, vSr, multireaction=False):
    """
    Returns the gibbs energy of reaction in kJ/kmol.

    Parameters
    ----------
    T : float or int
        Temperature in K.
    vHr : float, int, or 1d array
        Contains heat of reaction at T in kJ/kmol.
    vSr : float, int, or 1d array
        Contains entropy of reaction at T in kJ/kmol.K.

    Returns
    -------
    float, int, or 1d array
        Contains gibbs energy of reaction at T in kJ/kmol.

    """
    return np.array(vHr) - T*np.array(vSr)

def get_Cp(T, a, b, c, d):
    """
    Returns the heat capacities of species at temperature T in K

    Parameters
    ----------
    T : float or int
        Temperature in K.
    a : 1d array or float
        Coefficients for heat capacity from Reid equation.
    b : 1d array or float
        Coefficients for heat capacity from Reid equation.
    c : 1d array or float
        Coefficients for heat capacity from Reid equation.
    d : 1d array or float
        Coefficients for heat capacity from Reid equation.

    Returns
    -------
    res : 1d array or float
        Heat capacities in kJ/kmol.K.

    """
    return heat_capacity(np.array(a), np.array(b), np.array(c), np.array(d), T)
