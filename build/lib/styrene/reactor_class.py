# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:31:06 2020

@author: bruno
"""

import numpy as np
import pandas as pd
import math

from styrene.heterogeneous_kinetics import (rt1, rt2, rt3, rt4, rc1, rc2, rc3, rc4)
from styrene.heterogeneous_kinetics import (ft_reactants, effective_reactions)
from styrene.heterogeneous_kinetics import (components, eb, st, h2, bz, me, to, ee, h2o)
from styrene.data_repository import (_a, _b, _c, _d, _va, _vb, _vc, _vd,
                                     _vHr_298, _Hf_298)
from styrene.data_repository import (_Mm, _Tc, _Pc, _sigma, _ek, _delta, get_mi_mist)
from styrene.bvp import OrthogonalCollocation
from styrene.mass_transfer import fnu, fuller_ab_mat, wilke_mist, effective_diff
from styrene.thermodynamics import get_Cp, get_vHr, get_HfT
from styrene.fluid_dynamics import fpressure_drop

from scipy.integrate import solve_ivp
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

class CatalystBed(object):
    
    def __init__(self, W, rhos, rhob, es, dp, tao, inner_R, Pmin=0.5, Pterm=None,
                 heterogeneous=False, n_points=6,
                 terminal=True, components=components, ivp_rtol=1e-6):
        """
        Class for catalyst bed.

        Args:
            W (float): Catalyst loading.
            rhos (float): Catalyst solid density.
            rhob (float): Catalyst bulk density (considers void fraction between pelets).
            es (float): Void fraction in solid.
            dp (float): Pellet diameter.
            tao (float): Pelet tortuosity (mass diffusion).
            inner_R (float): Reactor inner radius.
            Pmin (float, optional): Lowest pressure allowed. Defaults to 0.5.
            Pterm (float, optional): Lowest pressure allowed in intermediate solutions. Defaults to 0.05.
            heterogeneous (bool, optional): Either to consider mass diffusion or not. Defaults to False.
            n_points (int, optional): Points used in orthogonal collocation problem. Defaults to 6.
            terminal (bool, optional): Set solution as terminal when reaching Pterm. Defaults to True.
            components (list, optional): Components. Defaults to components.
            ivp_rtol (float, optional): Relative tolerance of IVP solution. Defaults to 1e-6.
        """
        
        self.components = components
        self.W = W
        self.rhos = rhos
        self.rhob = rhob
        self.es = es
        self.eg = 1 - rhob/rhos
        self.dp = dp
        self.tao = tao
        self.inner_R = inner_R
        self.Pmin = Pmin
        self.terminal = terminal
        self.heterogeneous = heterogeneous
        self.n_points = n_points
        self._ivp_rtol = ivp_rtol
        
        if heterogeneous:
            self.collocation = OrthogonalCollocation(CatalystBed._transport_eq,
                                                     CatalystBed._bc_eq,
                                                     n_points, 3, x0=0, x1=dp/2)
        
        if Pterm is None:
            self.Pterm = Pmin / 10
        else:
            self.Pterm = Pterm
    
    def _w_flow(self, W, params):
        pass

    def _f_term(self, W, params):
        return params[-1] - self.Pterm
    
    _f_term.terminal = True
    
    @property
    def _nu(self):
        _nu = np.array([fnu(8, 10, 0, 1),
                    fnu(8, 8, 0, 1), 7.07,
                    fnu(6, 6, 0, 1),
                    fnu(1, 4, 0, 0),
                    fnu(7, 8, 0, 1),
                    fnu(2, 4, 0, 0),
                    12.7]) * 1e-3
        return _nu
    
    def get_diff_mist(self, y, T, P, Mm=_Mm):
        """T in [K], P in [bar], result in m**2/h"""
        D = fuller_ab_mat(Mm, self._nu, T, P) #values in m**2/s
        Dm = wilke_mist(y, D) * 3600 #convert to m**2/h
        Dme = effective_diff(Dm, self.tao, self.es)
        return Dme
    
    def set_inlet(self, Feb, Fst, Fh2, Fbz, Fme, Fto, Fee, Fh2o, T, P):
        """
        Sets reactor inlet conditions.

        Args:
            Feb (float): Molar flow rate of Ethylbenzene [kmol/h]
            Fst (float): Molar flow rate of Styrene [kmol/h]
            Fh2 (float): Molar flow rate of H2 [kmol/h]
            Fbz (float): Molar flow rate of Benzene [kmol/h]
            Fme (float): Molar flow rate of Methane [kmol/h]
            Fto (float): Molar flow rate of Toluene [kmol/h]
            Fee (float): Molar flow rate of Ethylene [kmol/h]
            Fh2o (float): Molar flow rate of Steam [kmol/h]
            T (float): Inlet temperature [K]
            P (float): Inlet absolute pressure [bar]
        """
        params = np.array([Feb, Fst, Fh2, Fbz, Fme, Fto, Fee, Fh2o, T, P])
        keys = ['Feb', 'Fst', 'Fh2', 'Fbz', 'Fme', 'Fto', 'Fee', 'Fh2o', 'T', 'P']
        self.inlet = dict(zip(keys, params))
        self._inlet_values = params
        self._keys = keys
    
    def reset_inlet(self, **kwargs):
        """
        Resets reactor inlet for given keys.

        Raises:
            KeyError: Unavailable keys.
        """
        for key, value in kwargs.items():
            if key in self._keys:
                self.inlet[key] = value
            else:
                raise KeyError(key + ' is not available for a change')
        self._inlet_values = list(self.inlet.values())
    
    @staticmethod
    def _transport_eq(r, y, dy, d2y, yb, *args):
        
        return d2y - ft_reactants(r, y, *args)
    
    @staticmethod
    def _bc_eq(r, y, dy, d2y, yb, *args):
        
        return y - yb
    
    def solve(self, points_eval=None, **kwargs):
        """
        Solves initial value problem

        Args:
            points_eval (int or None, optional): Number of points to eval in ivp solution. Defaults to None.
        """
        t_eval = None
        if not (points_eval is None):
            t_eval = np.linspace(0, self.W, points_eval)
        
        CatalystBed._f_term.terminal = self.terminal

        self.ivp_solution = solve_ivp(self._w_flow, (0, self.W), self._inlet_values,
                                      events=(self._f_term), t_eval=t_eval,
                                      rtol=self._ivp_rtol, **kwargs)
        self._outlet_values = self.ivp_solution.y[:,-1]
        self.outlet = dict(zip(self._keys, self._outlet_values))
    
    def get_outlet(self, **options):
        """
        Returns catalyst bed outlet.

        Returns:
            dict: Catalyst bed outlet.
        """
        try: return self.outlet
        except:
            self.solve(**options)
            return self.outlet
    
    def get_pre_heating(self, T_before):
        T0 = self.inlet['T']
        h_before = get_HfT(T_before, _Hf_298, _a, _b, _c, _d)
        h0 = get_HfT(T0, _Hf_298, _a, _b, _c, _d)
        self.pre_heat = (h0 - h_before).dot(self._inlet_values[:-2]) / 3.6e6
        return self.pre_heat#MW
    
    def get_dataframe(self, **options):
        if self.ivp_solution is None:
            self.solve(**options)
        df = pd.DataFrame(self.ivp_solution.y.T, columns=self._keys)
        df['W'] = self.ivp_solution.t
        df.set_index('W', inplace=True)
        return df
    
      
        
class AxialBed(CatalystBed):
    
    def __init__(self, W, rhos, rhob, es, dp, tao, inner_R, **options):
        """
        Creates instance of axial-flow bed.

        Args:
            W (float): Catalyst loading.
            rhos (float): Catalyst solid density.
            rhob (float): Catalyst bulk density (considers void fraction between pelets).
            es (float): Void fraction in solid.
            dp (float): Pellet diameter.
            tao (float): Pelet tortuosity (mass diffusion).
            inner_R (float): Reactor inner radius.
            
            **options:
                Pmin (float, optional): Lowest pressure allowed. Defaults to 0.5.
                Pterm (float, optional): Lowest pressure allowed in intermediate solutions. Defaults to 0.05.
                heterogeneous (bool, optional): Either to consider mass diffusion or not. Defaults to False.
                n_points (int, optional): Points used in orthogonal collocation problem. Defaults to 6.
                terminal (bool, optional): Set solution as terminal when reaching Pterm. Defaults to True.
                components (list, optional): Components. Defaults to components.
                ivp_rtol (float, optional): Relative tolerance of IVP solution. Defaults to 1e-6.
        """
        
        super(AxialBed, self).__init__(W, rhos, rhob, es, dp, tao, inner_R, **options)
        self.Ac = math.pi*inner_R**2
    
    def _w_flow(self, W, params):
        
        F = np.array(params[0:-2])
        T = params[-2]
        P = params[-1]
        Ft = F.sum(axis=-1)
        y = F / Ft
        pp = y * P
        eff = np.ones(4)
        
        if self.heterogeneous and (P > self.Pmin):
            
            if W == 0:
                root_method = 'lm'
                pp[pp == 0] = self._ivp_rtol
                y0 = np.column_stack((pp[:3],) * (self.n_points + 1))
            else:
                y0 = self.collocation.y
                root_method = 'hybr'
            
            Dme = self.get_diff_mist(y, T, P, _Mm)
            args_ft = (pp[:3], T, Dme, self.rhos, self.es)
            self.collocation.collocate(y0, args=args_ft,
                                       method=root_method)
     
            args_reactions = (T, self.rhos, self.es)
            eff = self.collocation.effectiveness(effective_reactions, args_reactions)
            
        '''Here I changed the reaction rates inside the catalyst particle for those
        accounting the thermal rates
        rr1 = (eg*rt1(pp,T) + (1-eg)*r1(pp,T,rhos,es)*eff[0])/rhob
        rr2 = (eg*rt2(pp,T) + (1-eg)*r2(pp,T,rhos,es)*eff[1])/rhob
        rr3 = (eg*rt3(pp,T) + (1-eg)*r3(pp,T,rhos,es)*eff[2])/rhob
        rr4 = (eg*rt4(pp,T) + (1-eg)*r4(pp,T,rhos,es)*eff[3])/rhob'''
        
        rr1 = self.eg/self.rhob*rt1(pp,T) + rc1(pp,T)*eff[0]
        rr2 = self.eg/self.rhob*rt2(pp,T) + rc2(pp,T)*eff[1]
        rr3 = self.eg/self.rhob*rt3(pp,T) + rc3(pp,T)*eff[2]
        rr4 = self.eg/self.rhob*rt4(pp,T) + rc4(pp,T)*eff[3]
        
        dF = np.zeros(len(F))
        dF[eb] = -rr1 - rr2 - rr3
        dF[st] = rr1 - rr4
        dF[h2] = rr1 - rr3 - 2*rr4
        dF[bz] = rr2
        dF[me] = rr3 + rr4
        dF[to] = rr3 + rr4
        dF[ee] = rr2
        dF[h2o] = 0
        
        Cp = get_Cp(T, _a, _b, _c, _d)
        vHr = get_vHr(T, _vHr_298, _va, _vb, _vc, _vd)
        dT = -(np.array([rr1, rr2, rr3, rr4]).T.dot(vHr)) / (Cp.T.dot(F))
        
        mi = get_mi_mist(T, y, _Mm, _Tc, _Pc, _sigma, _ek, _delta)
        rhog = (y.T.dot(np.array(_Mm))) * P / 8.314e-2 / T
        G = (np.array(F).T.dot(np.array(_Mm))) / self.Ac
        dP = -fpressure_drop(G, rhog, mi, self.Ac, self.dp, self.rhob, self.eg)
        
        if P < self.Pmin:
            dF = np.zeros(len(dF))
            dT = 0
            if P < self.Pterm:
                dP = 0
                
        return np.append(dF, [dT, dP])
    

class RadialBed(CatalystBed):
    
    def __init__(self, W, rhos, rhob, es, dp, tao, inner_R, z, **options):
        """
        Creates radial-flow bed.

        Args:
            W (float): Catalyst loading.
            rhos (float): Catalyst solid density.
            rhob (float): Catalyst bulk density (considers void fraction between pelets).
            es (float): Void fraction in solid.
            dp (float): Pellet diameter.
            tao (float): Pelet tortuosity (mass diffusion).
            inner_R (float): Reactor inner radius.
            z (float): Bed lenght.
            
            **options:
                Pmin (float, optional): Lowest pressure allowed. Defaults to 0.5.
                Pterm (float, optional): Lowest pressure allowed in intermediate solutions. Defaults to 0.05.
                heterogeneous (bool, optional): Either to consider mass diffusion or not. Defaults to False.
                n_points (int, optional): Points used in orthogonal collocation problem. Defaults to 6.
                terminal (bool, optional): Set solution as terminal when reaching Pterm. Defaults to True.
                components (list, optional): Components. Defaults to components.
                ivp_rtol (float, optional): Relative tolerance of IVP solution. Defaults to 1e-6.
        """
        
        super(RadialBed, self).__init__(W, rhos, rhob, es, dp, tao, inner_R, **options)
        self.z = z
        self.Ac0 = 2 * math.pi * inner_R * z
    
    def _w_flow(self, W, params):
        
        F = np.array(params[0:-2])
        T = params[-2]
        P = params[-1]
        Ft = F.sum(axis=-1)
        y = F / Ft
        pp = y * P
        eff = np.ones(4)
        
        if self.heterogeneous and (P > self.Pmin):
            
            if W == 0:
                root_method = 'lm'
                pp[pp == 0] = self._ivp_rtol
                y0 = np.column_stack((pp[:3],) * (self.n_points + 1))
            else:
                y0 = self.collocation.y
                root_method = 'hybr'
                
            Dme = self.get_diff_mist(y, T, P, _Mm)
            args_ft = (pp[:3], T, Dme, self.rhos, self.es)
            self.collocation.collocate(y0, args=args_ft,
                                       method=root_method)
     
            args_reactions = (T, self.rhos, self.es)
            eff = self.collocation.effectiveness(effective_reactions, args_reactions)
 
        rr1 = self.eg/self.rhob*rt1(pp,T) + rc1(pp,T)*eff[0]
        rr2 = self.eg/self.rhob*rt2(pp,T) + rc2(pp,T)*eff[1]
        rr3 = self.eg/self.rhob*rt3(pp,T) + rc3(pp,T)*eff[2]
        rr4 = self.eg/self.rhob*rt4(pp,T) + rc4(pp,T)*eff[3]
        
        dF = np.zeros(len(F))
        dF[eb] = -rr1 - rr2 - rr3
        dF[st] = rr1 - rr4
        dF[h2] = rr1 - rr3 - 2*rr4
        dF[bz] = rr2
        dF[me] = rr3 + rr4
        dF[to] = rr3 + rr4
        dF[ee] = rr2
        dF[h2o] = 0
        
        Cp = get_Cp(T, _a, _b, _c, _d)
        vHr = get_vHr(T, _vHr_298, _va, _vb, _vc, _vd)
        dT = -(np.array([rr1, rr2, rr3, rr4]).T.dot(vHr)) / (Cp.T.dot(F))
        
        r = (W / (math.pi * self.rhob * self.z) + self.inner_R**2) ** 0.5
        Ac = 2 * math.pi * r * self.z
        
        mi = get_mi_mist(T, y, _Mm, _Tc, _Pc, _sigma, _ek, _delta)
        rhog = (y.T.dot(np.array(_Mm))) * P / 8.314e-2 / T
        G = (np.array(F).T.dot(np.array(_Mm))) / Ac
        dP = -fpressure_drop(G, rhog, mi, Ac, self.dp, self.rhob, self.eg)
        
        if P < self.Pmin:
            dF = np.zeros(len(dF))
            dT = 0
            if P < self.Pterm:
                dP = 0
                
        return np.append(dF, [dT, dP])


class IsoBed(CatalystBed):
    
    def __init__(self, W, rhos, rhob, es, **options):
        """[summary]

        Args:
            W (float): Catalyst loading.
            rhos (float): Catalyst solid density.
            rhob (float): Catalyst bulk density (considers void fraction between pelets).
            es (float): Void fraction in solid.
            
            **options:
                Pmin (float, optional): Lowest pressure allowed. Defaults to 0.5.
                Pterm (float, optional): Lowest pressure allowed in intermediate solutions. Defaults to 0.05.
                heterogeneous (bool, optional): Either to consider mass diffusion or not. Defaults to False.
                n_points (int, optional): Points used in orthogonal collocation problem. Defaults to 6.
                terminal (bool, optional): Set solution as terminal when reaching Pterm. Defaults to True.
                components (list, optional): Components. Defaults to components.
                ivp_rtol (float, optional): Relative tolerance of IVP solution. Defaults to 1e-6.
        """

        super(IsoBed, self).__init__(W, rhos, rhob, es, None, None, None, **options)
    
    def _w_flow(self, W, params):
        F = np.array(params[0:-2])
        T = params[-2]
        P = params[-1]
        Ft = sum(F)
        y = F/Ft
        pp = y*P
        eff = np.ones(4)
        rr1 = self.eg/self.rhob*rt1(pp,T) + rc1(pp,T)*eff[0]
        rr2 = self.eg/self.rhob*rt2(pp,T) + rc2(pp,T)*eff[1]
        rr3 = self.eg/self.rhob*rt3(pp,T) + rc3(pp,T)*eff[2]
        rr4 = self.eg/self.rhob*rt4(pp,T) + rc4(pp,T)*eff[3]
        dF = np.zeros(len(F))
        dF[eb] = -rr1 - rr2 - rr3
        dF[st] = rr1 - rr4
        dF[h2] = rr1 - rr3 - 2*rr4
        dF[bz] = rr2
        dF[me] = rr3 + rr4
        dF[to] = rr3 + rr4
        dF[ee] = rr2
        dF[h2o] = 0

        return np.append(dF,[0,0])


class BedResets(dict):
    
    def _set_resets(self, bed_number, **resets):
        self['bed_'+str(bed_number)] = resets
    
    def _add_resets(self, bed_number, **resets):
        if 'bed_'+str(bed_number) not in self.keys():
            raise KeyError('Please use set_resets before adding new')
        else:
            for key, value in resets.items():
                self['bed_'+str(bed_number)][key] = value


class OptParams(dict):
    
    def _set_params(self, **params):
        "Each param is a list of tuples with (bed_number,guess,lb,ub)."
        for key in params.keys():
            self[key] = params[key]
            continue
        pass
    pass


class OptParamsMultipliers(dict):
    
    def _set_multiplier(self, **params):
        "**param=multiplier"
        for key in params.keys():
            self[key] = params[key]


class MultiBed(object):
    
    def __init__(self):
        """
        Creates multibed reactor.
        """
        self.n_beds = 0
        self.beds = {}
        self.resets = BedResets()
        self.steam_ratios = {}
        self._opt_keys = ['T', 'P', 'W', 'Feb', 'sOeb']
        self.opt_params = OptParams(T=[], P=[], W=[], Feb=[], sOeb=[])
        self.opt_params_mult = OptParamsMultipliers(T=1e2, P=1e0, W=1e1, Feb=1e2, sOeb=1e0)
    
    def add_axial_bed(self, W, rhos, rhob, es, dp, tao, inner_R, **options):
        """
        Adds axial bed to the end of the reactor.

        Args:
            W (float): Catalyst loading.
            rhos (float): Catalyst solid density.
            rhob (float): Catalyst bulk density (considers void fraction between pelets).
            es (float): Void fraction in solid.
            dp (float): Pellet diameter.
            tao (float): Pelet tortuosity (mass diffusion).
            inner_R (float): Reactor inner radius.
            
            **options:
                Pmin (float, optional): Lowest pressure allowed. Defaults to 0.5.
                Pterm (float, optional): Lowest pressure allowed in intermediate solutions. Defaults to 0.05.
                heterogeneous (bool, optional): Either to consider mass diffusion or not. Defaults to False.
                n_points (int, optional): Points used in orthogonal collocation problem. Defaults to 6.
                terminal (bool, optional): Set solution as terminal when reaching Pterm. Defaults to True.
                components (list, optional): Components. Defaults to components.
                ivp_rtol (float, optional): Relative tolerance of IVP solution. Defaults to 1e-6.
        """
        
        self.n_beds = self.n_beds + 1
        self.beds['bed_'+str(self.n_beds)] = AxialBed(W, rhos, rhob, es, dp, tao, inner_R, **options)
    
    def add_radial_bed(self, W, rhos, rhob, es, dp, tao, inner_R, z, **options):
        """
        Adds radial bed to the end of the reactor.

        Args:
            W (float): Catalyst loading.
            rhos (float): Catalyst solid density.
            rhob (float): Catalyst bulk density (considers void fraction between pelets).
            es (float): Void fraction in solid.
            dp (float): Pellet diameter.
            tao (float): Pelet tortuosity (mass diffusion).
            inner_R (float): Reactor inner radius.
            z (float): Bed lenght.
            
            **options:
                Pmin (float, optional): Lowest pressure allowed. Defaults to 0.5.
                Pterm (float, optional): Lowest pressure allowed in intermediate solutions. Defaults to 0.05.
                heterogeneous (bool, optional): Either to consider mass diffusion or not. Defaults to False.
                n_points (int, optional): Points used in orthogonal collocation problem. Defaults to 6.
                terminal (bool, optional): Set solution as terminal when reaching Pterm. Defaults to True.
                components (list, optional): Components. Defaults to components.
                ivp_rtol (float, optional): Relative tolerance of IVP solution. Defaults to 1e-6.
        """
        
        self.n_beds = self.n_beds + 1
        self.beds['bed_'+str(self.n_beds)] = RadialBed(W, rhos, rhob, es, dp, tao, inner_R, z, **options)
    
    def set_inlet_plant(self, Feb, Fst, Fh2, Fbz, Fme, Fto, Fee, Fh2o, T, P):
        """
        Sets the inlet of the first catalyst bed.

        Args:
            Feb (float): Molar flow rate of Ethylbenzene [kmol/h]
            Fst (float): Molar flow rate of Styrene [kmol/h]
            Fh2 (float): Molar flow rate of H2 [kmol/h]
            Fbz (float): Molar flow rate of Benzene [kmol/h]
            Fme (float): Molar flow rate of Methane [kmol/h]
            Fto (float): Molar flow rate of Toluene [kmol/h]
            Fee (float): Molar flow rate of Ethylene [kmol/h]
            Fh2o (float): Molar flow rate of Steam [kmol/h]
            T (float): Inlet temperature [K]
            P (float): Inlet absolute pressure [bar]

        Raises:
            IndexError: When no bed was created yet.
        """
        params = np.array([Feb, Fst, Fh2, Fbz, Fme, Fto, Fee, Fh2o, T, P])
        keys = ['Feb', 'Fst', 'Fh2', 'Fbz', 'Fme', 'Fto', 'Fee', 'Fh2o', 'T', 'P']
        self.inlet = dict(zip(keys, params))
        self._inlet_values = params
        self._keys = keys
        if self.n_beds == 0:
            raise IndexError('Please add a bed before the inlet')
        self.beds['bed_1'].set_inlet(Feb, Fst, Fh2, Fbz, Fme, Fto, Fee, Fh2o, T, P)
    
    def add_resets(self, bed_number, **resets):
        "key=value"
        for key in resets.keys():
            if key not in self._keys:
                raise KeyError(key + ' is not an available key to reset')
        if 'bed_' + str(bed_number) not in self.beds.keys():
            raise KeyError('bed ' + str(bed_number) + " was not added yet")
        if 'bed_' + str(bed_number) in self.resets.keys():
            self.resets._add_resets(bed_number, **resets)
        else:
            self.resets._set_resets(bed_number, **resets)
    
    def set_bed_steam_ratio(self, bed_number, sOeb):
        if 'bed_' + str(bed_number) not in self.beds.keys():
            raise KeyError('bed_' + str(bed_number) + " was not added yet")
        self.steam_ratios['bed_' + str(bed_number)] = sOeb
    
    def solve(self, **options):
        "points_eval=None"
        
        if 'bed_1' in self.resets.keys():
            self.beds['bed_1'].reset_inlet(**self.resets['bed_1'])
            self.inlet = self.beds['bed_1'].inlet
            self._inlet_values = self.beds['bed_1']._inlet_values
            
        if 'bed_1' in self.steam_ratios.keys():
            Feb0 = self.beds['bed_1'].inlet['Feb']
            sOeb = self.steam_ratios['bed_1']
            self.beds['bed_1'].reset_inlet(Fh2o=Feb0*sOeb)
        self.beds['bed_1'].solve(**options)
        
        if self.n_beds == 0:
            raise IndexError('Please add a bed before the inlet')
        
        elif self.n_beds == 1:
            self.outlet = self.beds['bed_1'].outlet
            
        elif self.n_beds > 1:
            for i in range(2, self.n_beds + 1):
                self.beds['bed_' + str(i)].set_inlet(*self.beds['bed_' + str(i-1)]._outlet_values)
                if 'bed_'+str(i) in self.resets.keys():
                    self.beds['bed_' + str(i)].reset_inlet(**self.resets['bed_' + str(i)])
                if 'bed_'+str(i) in self.steam_ratios.keys():
                    Feb0 = self.beds['bed_1'].inlet['Feb']
                    sOeb = self.steam_ratios['bed_' + str(i)]
                    self.beds['bed_' + str(i)].reset_inlet(Fh2o=Feb0*sOeb)
                self.beds['bed_' + str(i)].solve(**options)
            self.outlet = self.beds['bed_' + str(self.n_beds)].outlet
    
    def get_outlet(self, **options):
        "at solve: points_eval=None"
        try: return self.outlet
        except:
            self.solve(**options)
            return self.outlet
    
    def get_heat_consumed(self, initial_T=786.61):
        if self.n_beds == 0:
            raise IndexError('Please add a bed before the inlet')
        elif self.n_beds == 1:
            return self.beds['bed_1'].get_pre_heating(initial_T)
        elif self.n_beds > 1:
            heat_consumed = pd.Series(name='Q')
            heat_consumed['bed_1'] = self.beds['bed_1'].get_pre_heating(initial_T)
            for i in range(2, self.n_beds + 1):
                prev_T = self.beds['bed_' + str(i-1)].outlet['T']
                heat_consumed['bed_' + str(i)] = self.beds['bed_' + str(i)].get_pre_heating(prev_T)
        heat_consumed['total'] = heat_consumed.values.sum()
        return heat_consumed
    
    @property
    def conversion(self):
        Xeb = 1 - (self.outlet['Feb'] / self.inlet['Feb'])
        Xst = (self.outlet['Fst'] - self.inlet['Fst']) / self.inlet['Feb']
        Xbz = (self.outlet['Fbz'] - self.inlet['Fbz']) / self.inlet['Feb']
        Xto = (self.outlet['Fto'] - self.inlet['Fto']) / self.inlet['Feb']
        return {'Xeb':Xeb, 'Xst':Xst, 'Xbz':Xbz, 'Xto':Xto}
    
    def get_bed_dataframe(self, bed_number, **options):
        "at solve: points_eval=None"
        try: return self.beds['bed_' + str(bed_number)].get_dataframe()
        except:
            self.solve(**options)
            return self.beds['bed_' + str(bed_number)].get_dataframe()
    
    def get_dataframe(self, **options):
        "at solve: points_eval=None"
        df = self.get_bed_dataframe(1, **options)
        W = df.index.values
        
        for n in range(2, self.n_beds + 1):
            df_bed = self.get_bed_dataframe(n, **options)
            df = df.append(df_bed, ignore_index=True)
            W = np.append(W, df_bed.index.values + W[-1])
            
        df['W'] = W
        df.set_index('W',inplace=True)
        
        return df
    
    def to_excel(self, filename, **options):
        "at solve: points_eval=None"
        path = filename + ".xlsx"
        with pd.ExcelWriter(path) as writer:
            self.get_dataframe(**options).to_excel(writer, sheet_name='Multibed')
            for n in range(1, self.n_beds + 1):
                self.get_bed_dataframe(n, **options).to_excel(writer, sheet_name='bed_'+str(n))
    
    def set_opt_params(self, **params):
        """"param=[(bed_number,guess,lb,ub)]
        T(K), P(bar), W(kg*1e3), sOeb(int)
        Each param is a list of tuples"""
        for item in params:
            if item not in self._opt_keys:
                raise KeyError(item + ' is not an available key to optimize\nUse:', self._opt_keys)
        self.opt_params._set_params(**params)
    
    def set_opt_params_multipliers(self, **params):
        """"**param=multiplier
        T(K), P(bar), W(kg*1e3), sOeb(int)
        Each param is a list of tuples"""
        for item in params:
            if item not in self._opt_keys:
                raise KeyError(item + ' is not an available key to optimize\nUse:',self._opt_keys)
        self.opt_params_mult._set_params(**params)
    
    def _guess_bound(self):
        self._guess_0 = []
        self._opt_bounds = []
        self._opt_key_labels = []
        self._opt_params_types = []
        for key in self.opt_params.keys():
            for item in self.opt_params[key]:
                self._guess_0.append(item[1] / self.opt_params_mult[key])
                self._opt_bounds.append((item[2] / self.opt_params_mult[key],
                                         item[3] / self.opt_params_mult[key]))
                self._opt_key_labels.append(key + '_' + str(item[0]))
                self._opt_params_types.append(key)
    
    def _prep_opt(self,params):
        i = 0
        for item in self.opt_params['T']:
            self.add_resets(item[0], T=params[i]*self.opt_params_mult['T'])
            i = i + 1
        for item in self.opt_params['P']:
            self.add_resets(item[0], P=params[i]*self.opt_params_mult['P'])
            i = i + 1
        for item in self.opt_params['W']:
            self.beds['bed_'+str(item[0])].W = params[i]*self.opt_params_mult['W']*1e3
            i = i + 1
        for item in self.opt_params['Feb']:
            if item[0] == 1:
                self.beds['bed_'+str(item[0])].reset_inlet(Feb=params[i]*self.opt_params_mult['Feb'])
            else:
                self.add_resets(item[0], Feb=params[i]*self.opt_params_mult['Feb'])
            i = i + 1
        for item in self.opt_params['sOeb']:
            self.set_bed_steam_ratio(item[0], params[i])
            #self.add_resets(item[0],Fh2o=params[i]*Feb0*self.opt_params_mult['sOeb'])
            i = i + 1
    
    def _opt_pareto(self, params, w_select, w_conv):
        self._prep_opt(params)
        self.solve()
        conv = self.conversion
        Sst = conv['Xst'] / conv['Xeb']
        print('weights', w_select, w_conv)
        print('Params:', params, '\n')
        print('Outlet:\n', self.outlet, '\n')
        return -(w_select*Sst + w_conv*conv['Xst'])
    
    def _constraint_P(self, params):
        self._prep_opt(params)
        self.solve()
        print('Constraint:')
        print('Params:',params,'\n')
        print('Outlet:\n',self.outlet,'\n')
        #return self.outlet['constraint']
        return self.outlet['P'] - self._p_cons
    
    def optimize(self, w_select, w_conv, method=None, tol=1e-8,
                 constraint_P=None, constraint_W=None, options={}):
        """[summary]

        Args:
            w_select (float): Minimum weight for selectivity.
            w_conv (float): Minimum weight for conversion.
            method (str, optional): Optimization method. For constrained problems use 'SLSQP'. Defaults to None.
            tol (float, optional): Optimization tolerance. Defaults to 1e-8.
            constraint_P (float or None, optional): Value of minimum pressure for constraint. Defaults to None.
            constraint_W (float of None, optional): Value for total catalyst weight. Defaults to None.
            options (dict, optional): Options passed to scipy minimize. Defaults to {}.

        Returns:
            [type]: [description]
        """
        
        self._guess_bound()
        cons = []
        
        if not (constraint_P is None):
            for bed in self.beds:
                self.beds[bed].terminal = False
                self._p_cons = constraint_P
            cons_P = NonlinearConstraint(self._constraint_P, 0, np.inf)
            cons.append(cons_P)
        
        if not (constraint_W is None):
            A = np.zeros([1,len(self._opt_params_types)])
            for index,key in enumerate(self._opt_params_types):
                if key == 'W':
                    A[0,index] = 1
            bds = constraint_W/self.opt_params_mult['W']
            cons_W = LinearConstraint(A,bds,bds)
            cons.append(cons_W)
            
        return minimize(self._opt_pareto, self._guess_0, args=(w_select, w_conv), method=method,
                        bounds=self._opt_bounds, constraints=cons, tol=tol, options=options)
    
    def _gen_pareto(self, min_w_select, min_w_conv, n_points):
        "min in range 0:1"
        rng = 1 - min_w_select - min_w_conv
        step = rng / (n_points - 1)
        w_pareto = [(round(min_w_select + i*step, 6),
                     round(1 - min_w_select - i*step, 6)) for i in range(0, n_points)]
        self._w_pareto = w_pareto
        return w_pareto
    
    def optimize_pareto(self, min_w_select, min_w_conv, n_points, **kwargs):
        """[summary]

        Args:
            min_w_select (float): Minimum weight for selectivity.
            min_w_conv (float): Minimum weight for conversion.
            n_points (int): Points in the pareto front
            
            **options:
                method (str, optional): Optimization method. For constrained problems use 'SLSQP'. Defaults to None.
                tol (float, optional): Optimization tolerance. Defaults to 1e-8.
                constraint_P (float or None, optional): Value of minimum pressure for constraint. Defaults to None.
                constraint_W (float of None, optional): Value for total catalyst weight. Defaults to None.
                options (dict, optional): Options passed to scipy minimize. Defaults to {}.


        Returns:
            pandas DataFrame: Pareto solutions
        """
        self._gen_pareto(min_w_select, min_w_conv, n_points)
        
        self._pareto_solutions = []
        self._Sst_pareto = []
        self._Xst_pareto = []
        self._params_pareto = []
        self._last_w_pareto = []
        
        for weights in self._w_pareto:
            
            sol_step = self.optimize(*weights, **kwargs)
            params_multipliers = np.array([self.opt_params_mult[key] for key in self._opt_params_types])
            self._pareto_solutions.append(sol_step)
            
            conv = self.conversion
            self._Sst_pareto.append(conv['Xst'] / conv['Xeb'])
            self._Xst_pareto.append(conv['Xst'])
            
            self._params_pareto.append(sol_step.x * params_multipliers)
            self._last_w_pareto.append(self.beds['bed_'+str(self.n_beds)].ivp_solution.t[-1]*1e-3)
            
        df = pd.DataFrame(self._w_pareto, columns=['w1', 'w2'])
        df['S(ST)'] = self._Sst_pareto
        df['X(ST)'] = self._Xst_pareto
        df_params = pd.DataFrame(self._params_pareto, columns=self._opt_key_labels)
        df_params['W_' + str(self.n_beds)] = self._last_w_pareto
        df = pd.concat([df, df_params], axis=1)
        self.pareto_dataframe = df
        
        return self.pareto_dataframe
    



