# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:24:27 2020

@author: bruno
"""
import numpy as np
from styrene.thermodynamics import f_vabcd
from styrene.fluid_dynamics import fmi_Thodos, fpot_Sm, fmi_CE, fphi_mat, fmi_mist

#Component labels to use in arrays
eb=0
st=1
h2=2
bz=3
me=4
to=5
ee=6
h2o=7

components = ['eb','st','h2','bz','me','to','ee','h20']

"""Preparing properties to use in other modules"""

"""Thermodynamics"""

_a = np.array([-4.31e1, -2.825e1, 2.714e1, -3.392e1, 1.925e1, -2.435e1, 3.806, 3.224e1])
_b = np.array([7.072e-1, 6.159e-1, 9.274e-3, 4.739e-1, 5.213e-2, 5.125e-1, 1.566e-1, 1.924e-3])
_c = np.array([-4.811e-4, -4.023e-4, -1.381e-5, -3.017e-4, 1.197e-5, -2.765e-4, -8.348e-5, 1.055e-5])
_d = np.array([1.301e-7, 9.935e-8, 7.645e-9, 7.13e-8, -1.132e-8, 4.911e-8, 1.755e-8, -3.596e-9])
_Hf_298 = np.array([2.981e4, 1.475e5, 0, 8.298e4, -7.49e4, 5.003e4, 5.234e4, -2.42e5])
_Gf_298 = np.array([1.307e5, 2.139e5, 0, 1.297e5, -5.087e4, 1.221e5, 6.816e4, -2.288e4])
_Sf_298 = (np.array(_Hf_298) - np.array(_Gf_298)) / 298.15

_vr1 = f_vabcd([_a, _b, _c, _d, _Hf_298, _Gf_298, _Sf_298], [eb, st, h2], [-1, 1, 1])
_vr2 = f_vabcd([_a, _b, _c, _d, _Hf_298, _Gf_298, _Sf_298], [eb, bz, ee], [-1, 1, 1])
_vr3 = f_vabcd([_a, _b, _c, _d, _Hf_298, _Gf_298, _Sf_298], [eb, h2, to, me], [-1, -1, 1, 1])
_vr4 = f_vabcd([_a, _b, _c, _d, _Hf_298, _Gf_298, _Sf_298], [st, h2, to, me], [-1, -2, 1, 1])

def _get_vr(*vrs):
    mat = np.array([*vrs])
    return tuple(mat.T)

_va, _vb, _vc, _vd, _vHr_298, _vGr_298, _vSr_298 = _get_vr(_vr1, _vr2, _vr3, _vr4)

"""Fluid dynamics"""

_Mm = np.array([106.168, 104.152, 2.016, 78.114, 16.043, 92.141, 28.054, 18.015]) #kg/kmol
_Tc = np.array([617.2, 647.0, 32.2, 562.2, 190.4, 591.8, 282.4, 647.3]) #K
_Pc = np.array([36.0, 39.9, 13.0, 48.9, 46.0, 41.0, 50.4, 221.2]) #bar
_sigma = np.array([np.nan, np.nan, 2.827, 5.349, 3.758, np.nan, 4.163, 2.641]) #Angstron
_ek = np.array([np.nan, np.nan, 59.7, 412.3, 148.6, np.nan, 224.7, 809.1]) #K
_delta = np.zeros(len(components))
_delta[h2o] = 1.0

'''For Ethylbenzene, Styrene, Benzene, Methane, Toluene and Ethylene: Thodos'''
'''For H2 and H2O: Chapman-Enskog'''

def fmi_list(T, Mm, Tc, Pc, sigma, ek, delta):
    
    mi = np.zeros(len(Mm))
    _hc = np.array([eb, st, bz, me, to, ee])
    
    mi[_hc] = fmi_Thodos(T, Tc[_hc], Pc[_hc], Mm[_hc])
    
    _low_mass = np.array([h2, h2o])
    potV = fpot_Sm(T, ek[_low_mass], delta[_low_mass])
    mi[_low_mass] = fmi_CE(T, Mm[_low_mass], sigma[_low_mass], potV)
    
    return mi

def get_mi_mist(T, y, Mm, Tc, Pc, sigma, ek, delta):
    mi_list = fmi_list(T, Mm, Tc, Pc, sigma, ek, delta)
    phi = fphi_mat(mi_list, _Mm)
    return fmi_mist(mi_list, y, phi)