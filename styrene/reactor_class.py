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
from collocation.bvp import OrthogonalCollocation
from styrene.mass_transfer import fnu, fuller_ab_mat, wilke_mist, effective_diff
from styrene.thermodynamics import get_Cp, get_vHr, get_HfT
from styrene.fluid_dynamics import fpressure_drop

from scipy.integrate import solve_ivp
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

class CatalystBed(object):
    
    def __init__(
        self, W,
        rhos=2500.0, rhob=1422.0, es=0.4,
        dp=0.0055, tao=3.0, inner_R=3.5, Pmin=0.5, Pterm=None,
        heterogeneous=False, n_points=6,
        terminal=True, components=components,
        ivp_rtol=1e-6):
        """Class for catalyst bed.

        Parameters
        ----------
        W : float
            Catalyst loading [kg].
        rhos : float, optional
            Catalyst solid density [kg/m3], by default 2500.0
        rhob : float, optional
            Catalyst bulk density [kg/m3], by default 1422.0
        es : float, optional
            Catalyst solid void fraction, by default 0.4
        dp : float, optional
            Pellet equivalent diameter [m], by default 0.0055
        tao : float, optional
            Pellet tortuosity, by default 3.0
        inner_R : float, optional
            Catalyst bed inner radius [m], by default 3.5
        Pmin : float, optional
            Minimum pressure allowed [bar]. Useful in optimization. By default 0.5
        Pterm : _type_, optional
            Terminal pressure to interrupt ODE system [bar], by default None
        heterogeneous : bool, optional
            Either of not to account diffusional limitations, by default False
        n_points : int, optional
            Number of internal collocation points in heterogeneous transport equations, by default 6
        terminal : bool, optional
            Either or not to use terminal events on ODE solutions, by default True
        components : _type_, optional
            List of components labels, by default components
        ivp_rtol : _type_, optional
            Relative tolerance for ODE system, by default 1e-6
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
            self.collocation = OrthogonalCollocation(
                CatalystBed._transport_eq,
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
        _nu = np.array(
            [fnu(8, 10, 0, 1),
             fnu(8, 8, 0, 1),
             7.07,
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
    
    def set_inlet(
        self,
        Feb=707.0,
        Fst=7.104,
        Fh2=0.0,
        Fbz=0.293,
        Fme=0.0,
        Fto=4.968,
        Fee=0.0,
        Fh2o=11*707.0,
        T=900,
        P=1.5):
        """Set inlet conditions of catalyst bed.

        Parameters
        ----------
        Feb : float, optional
            Ethylbenzene feed ratio [kmol/h], by default 707.0
        Fst : float, optional
            Styrene feed ratio [kmol/h], by default 7.104
        Fh2 : float, optional
            H2 feed ratio [kmol/h], by default 0.0
        Fbz : float, optional
            Benzene feed ratio [kmol/h], by default 0.293
        Fme : float, optional
            Methane feed ratio [kmol/h], by default 0.0
        Fto : float, optional
            Toluene feed ratio [kmol/h], by default 4.968
        Fee : float, optional
            Ethylene feed ratio [kmol/h], by default 0.0
        Fh2o : float, optional
            Steam feed ratio [kmol/h], by default 707 * 11
        T : int, optional
            Temperature [K], by default 900
        P : float, optional
            Pressure [bar], by default 1.5
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
        """Solve ODE system.

        Parameters
        ----------
        points_eval : int or None, optional
            Number of points to eval in ivp solution, by default None
        **kwargs : any
            Additional keyword arguments passed to scipy.integrate solve_ivp.
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
    
    def __init__(
        self, W,
        rhos=2500.0, rhob=1422.0, es=0.4,
        dp=0.0055, tao=3.0, inner_R=3.5,
        **options):
        """Class for axial-flow catalyst bed.

        Parameters
        ----------
        W : float
            Catalyst loading [kg].
        rhos : float, optional
            Catalyst solid density [kg/m3], by default 2500.0
        rhob : float, optional
            Catalyst bulk density [kg/m3], by default 1422.0
        es : float, optional
            Catalyst solid void fraction, by default 0.4
        dp : float, optional
            Pellet equivalent diameter [m], by default 0.0055
        tao : float, optional
            Pellet tortuosity, by default 3.0
        inner_R : float, optional
            Catalyst bed inner radius [m], by default 3.5
        Pmin : float, optional
            Minimum pressure allowed [bar]. Useful in optimization. By default 0.5
        Pterm : _type_, optional
            Terminal pressure to interrupt ODE system [bar], by default None
        heterogeneous : bool, optional
            Either of not to account diffusional limitations, by default False
        n_points : int, optional
            Number of internal collocation points in heterogeneous transport equations, by default 6
        terminal : bool, optional
            Either or not to use terminal events on ODE solutions, by default True
        components : _type_, optional
            List of components labels, by default components
        ivp_rtol : _type_, optional
            Relative tolerance for ODE system, by default 1e-6
        """
        
        super(AxialBed, self).__init__(
            W, rhos=rhos, rhob=rhob, es=es, dp=dp, tao=tao, inner_R=inner_R, **options
            )
        
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
    
    def __init__(
        self, W,
        rhos=2500.0, rhob=1422.0, es=0.4,
        dp=0.0055, tao=3.0, inner_R=3.5, z=7.0,
        **options):
        """Class for radial-flow catalyst bed.

        Parameters
        ----------
        W : float
            Catalyst loading [kg].
        rhos : float, optional
            Catalyst solid density [kg/m3], by default 2500.0
        rhob : float, optional
            Catalyst bulk density [kg/m3], by default 1422.0
        es : float, optional
            Catalyst solid void fraction, by default 0.4
        dp : float, optional
            Pellet equivalent diameter [m], by default 0.0055
        tao : float, optional
            Pellet tortuosity, by default 3.0
        inner_R : float, optional
            Catalyst bed inner radius [m], by default 1.5
        z : float, optional
            Catalyst bed lenght [m], by default 7.0
        Pmin : float, optional
            Minimum pressure allowed [bar]. Useful in optimization. By default 0.5
        Pterm : _type_, optional
            Terminal pressure to interrupt ODE system [bar], by default None
        heterogeneous : bool, optional
            Either of not to account diffusional limitations, by default False
        n_points : int, optional
            Number of internal collocation points in heterogeneous transport equations, by default 6
        terminal : bool, optional
            Either or not to use terminal events on ODE solutions, by default True
        components : _type_, optional
            List of components labels, by default components
        ivp_rtol : _type_, optional
            Relative tolerance for ODE system, by default 1e-6
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
    
    def __init__(
        self, W,
        rhos=2500.0, rhob=1422.0, es=0.4, **options):
        """Class for axial-flow catalyst bed.

        Parameters
        ----------
        W : float
            Catalyst loading [kg].
        rhos : float, optional
            Catalyst solid density [kg/m3], by default 2500.0
        rhob : float, optional
            Catalyst bulk density [kg/m3], by default 1422.0
        es : float, optional
            Catalyst solid void fraction, by default 0.4
        """

        super(IsoBed, self).__init__(W, rhos=rhos, rhob=rhob, es=es, **options)
    
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

        return np.append(dF,[0, 0])


class BedResets(dict):
    
    def _set_resets(self, bed_number, **resets):
        self[bed_number] = resets
    
    def _add_resets(self, bed_number, **resets):
        if bed_number not in self.keys():
            raise KeyError('Please use set_resets before adding new')
        else:
            for key, value in resets.items():
                self[bed_number][key] = value


class MultiBed(object):
    
    def __init__(self):
        """
        Creates multibed reactor.
        """
        self.n_beds = 0
        self.beds = {}
        self.resets = BedResets()
        self.steam_ratios = {}
        self._keys = ['Feb', 'Fst', 'Fh2', 'Fbz', 'Fme', 'Fto', 'Fee', 'Fh2o', 'T', 'P']
    
    def add_axial_bed(
        self, W,
        rhos=2500.0, rhob=1422.0, es=0.4,
        dp=0.0055, tao=3.0, inner_R=3.5,
        **options):
        """Class for axial-flow catalyst bed.

        Parameters
        ----------
        W : float
            Catalyst loading [kg].
        rhos : float, optional
            Catalyst solid density [kg/m3], by default 2500.0
        rhob : float, optional
            Catalyst bulk density [kg/m3], by default 1422.0
        es : float, optional
            Catalyst solid void fraction, by default 0.4
        dp : float, optional
            Pellet equivalent diameter [m], by default 0.0055
        tao : float, optional
            Pellet tortuosity, by default 3.0
        inner_R : float, optional
            Catalyst bed inner radius [m], by default 3.5
        Pmin : float, optional
            Minimum pressure allowed [bar]. Useful in optimization. By default 0.5
        Pterm : _type_, optional
            Terminal pressure to interrupt ODE system [bar], by default None
        heterogeneous : bool, optional
            Either of not to account diffusional limitations, by default False
        n_points : int, optional
            Number of internal collocation points in heterogeneous transport equations, by default 6
        terminal : bool, optional
            Either or not to use terminal events on ODE solutions, by default True
        components : _type_, optional
            List of components labels, by default components
        ivp_rtol : _type_, optional
            Relative tolerance for ODE system, by default 1e-6
        """
        
        self.n_beds = self.n_beds + 1
        self.beds[self.n_beds] = AxialBed(
            W, rhos=rhos, rhob=rhob, es=es, dp=dp, tao=tao, inner_R=inner_R, **options
            )
    
    def add_radial_bed(self, W,
        rhos=2500.0, rhob=1422.0, es=0.4,
        dp=0.0055, tao=3.0, inner_R=1.5, z=7.0,
        **options):
        """Class for radial-flow catalyst bed.

        Parameters
        ----------
        W : float
            Catalyst loading [kg].
        rhos : float, optional
            Catalyst solid density [kg/m3], by default 2500.0
        rhob : float, optional
            Catalyst bulk density [kg/m3], by default 1422.0
        es : float, optional
            Catalyst solid void fraction, by default 0.4
        dp : float, optional
            Pellet equivalent diameter [m], by default 0.0055
        tao : float, optional
            Pellet tortuosity, by default 3.0
        inner_R : float, optional
            Catalyst bed inner radius [m], by default 1.5
        z : float, optional
            Catalyst bed lenght [m], by default 7.0
        Pmin : float, optional
            Minimum pressure allowed [bar]. Useful in optimization. By default 0.5
        Pterm : _type_, optional
            Terminal pressure to interrupt ODE system [bar], by default None
        heterogeneous : bool, optional
            Either of not to account diffusional limitations, by default False
        n_points : int, optional
            Number of internal collocation points in heterogeneous transport equations, by default 6
        terminal : bool, optional
            Either or not to use terminal events on ODE solutions, by default True
        components : _type_, optional
            List of components labels, by default components
        ivp_rtol : _type_, optional
            Relative tolerance for ODE system, by default 1e-6
        """
        
        self.n_beds = self.n_beds + 1
        self.beds[self.n_beds] = RadialBed(
            W, rhos=rhos, rhob=rhob, es=es, dp=dp, tao=tao, inner_R=inner_R, z=z, **options)
    
    def set_inlet_plant(
        self,
        Feb=707.0,
        Fst=7.104,
        Fh2=0.0,
        Fbz=0.293,
        Fme=0.0,
        Fto=4.968,
        Fee=0.0,
        SEB=11,
        T=900,
        P=1.5):
        """Set inlet conditions of catalyst bed.

        Parameters
        ----------
        Feb : float, optional
            Ethylbenzene feed ratio [kmol/h], by default 707.0
        Fst : float, optional
            Styrene feed ratio [kmol/h], by default 7.104
        Fh2 : float, optional
            H2 feed ratio [kmol/h], by default 0.0
        Fbz : float, optional
            Benzene feed ratio [kmol/h], by default 0.293
        Fme : float, optional
            Methane feed ratio [kmol/h], by default 0.0
        Fto : float, optional
            Toluene feed ratio [kmol/h], by default 4.968
        Fee : float, optional
            Ethylene feed ratio [kmol/h], by default 0.0
        SEB : int, optional
            Steam-to-Ethylbenzene molar feed ratio, by default 11
        T : int, optional
            Temperature [K], by default 900
        P : float, optional
            Pressure [bar], by default 1.5
        """
        
        Fh2o = SEB * Feb
        
        if self.n_beds == 0:
            raise IndexError('Please add a bed before the inlet')
        self.beds[1].set_inlet(
            Feb=Feb,
            Fst=Fst,
            Fh2=Fh2,
            Fbz=Fbz,
            Fme=Fme,
            Fto=Fto,
            Fee=Fee,
            Fh2o=Fh2o,
            T=T,
            P=P)
        
        self.inlet = self.beds[1].inlet.copy()
    
    def add_resets(self, bed_number, **resets):
        "key=value"
        for key in resets.keys():
            if key not in self._keys:
                raise KeyError(key + ' is not an available key to reset')
        if bed_number not in self.beds.keys():
            raise KeyError('bed ' + str(bed_number) + " was not added yet")
        if bed_number in self.resets.keys():
            self.resets._add_resets(bed_number, **resets)
        else:
            self.resets._set_resets(bed_number, **resets)
    
    def set_bed_steam_ratio(self, bed_number, SEB):
        if bed_number not in self.beds.keys():
            raise KeyError('bed ' + str(bed_number) + " was not added yet")
        self.steam_ratios[bed_number] = SEB
    
    def solve(self, **options):
        "points_eval=None"
        
        if 1 in self.resets.keys():
            self.beds[1].reset_inlet(**self.resets[1])
            self.inlet = self.beds[1].inlet
            self._inlet_values = self.beds[1]._inlet_values
            
        if 1 in self.steam_ratios.keys():
            Feb0 = self.beds[1].inlet['Feb']
            SEB = self.steam_ratios[1]
            self.beds[1].reset_inlet(Fh2o=Feb0*SEB)
        self.beds[1].solve(**options)
        
        if self.n_beds == 0:
            raise IndexError('Please add a bed before the inlet')
        
        elif self.n_beds == 1:
            self.outlet = self.beds[1].outlet
            
        elif self.n_beds > 1:
            for i in range(2, self.n_beds + 1):
                self.beds[i].set_inlet(**self.beds[i - 1].outlet)
                if i in self.resets.keys():
                    self.beds[i].reset_inlet(**self.resets[i])
                if i in self.steam_ratios.keys():
                    Feb0 = self.beds[1].inlet['Feb']
                    SEB = self.steam_ratios[i]
                    self.beds[i].reset_inlet(Fh2o=Feb0*SEB)
                self.beds[i].solve(**options)
            self.outlet = self.beds[self.n_beds].outlet
    
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
            return self.beds[1].get_pre_heating(initial_T)
        elif self.n_beds > 1:
            heat_consumed = pd.Series(name='Q')
            heat_consumed[1] = self.beds[1].get_pre_heating(initial_T)
            for i in range(2, self.n_beds + 1):
                prev_T = self.beds[i - 1].outlet['T']
                heat_consumed[i] = self.beds[i].get_pre_heating(prev_T)
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
        try: return self.beds[bed_number].get_dataframe()
        except:
            self.solve(**options)
            return self.beds[bed_number].get_dataframe()
    
    def get_dataframe(self, **options):
        "at solve: points_eval=None"
        df = self.get_bed_dataframe(1, **options)
        W = df.index.values
        
        for n in range(2, self.n_beds + 1):
            df_bed = self.get_bed_dataframe(n, **options)
            df = pd.concat((df, df_bed), ignore_index=True)
            W = np.append(W, df_bed.index.values + W[-1])
            
        df['W'] = W
        df.set_index('W', inplace=True)
        
        return df
    
    def to_excel(self, filename, **options):
        "at solve: points_eval=None"
        path = filename + ".xlsx"
        with pd.ExcelWriter(path) as writer:
            self.get_dataframe(**options).to_excel(writer, sheet_name='Multibed')
            for n in range(1, self.n_beds + 1):
                self.get_bed_dataframe(n, **options).to_excel(writer, sheet_name='bed_'+str(n))




