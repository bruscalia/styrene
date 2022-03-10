# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:48:59 2020

@author: bruno
"""
import numpy as np
import math
from scipy.optimize import root
#from scipy.special import gamma

'''This program seems to be working perfectly well for internal collocation problems following:
    
    1) lap(y) = f(x,y)
    2) symmetric in x
    3) y(xbound) = previously defined
    4) The problem can be defined by dimensional variables'''

def _product(l):
    p = 1
    for i in l:
        p *= i
    return p

#shape must be s, c or any other = plain
def collocation_points(start, end, n_points, shape='s'):
    """
    Returns the collocation points and their respective coordinates in the original system.

    Parameters
    ----------
    start : float or int
        Starting coordinates of the independent variable in the collocation problem.
    end : float or int
        Ending coordinates of the independent variable in the collocation problem.
    n_points : int
        Number of internal collocation points.
    shape : str, optional
        Geometry of the problem.
        's': spherical
        'c': cilindrical
        other: regular slab
            The default is 's'.

    Returns
    -------
    dict
        'x_points' : 1d array
            Coordinates of the collocation points in original dimensions.
        'w_points' : 1d array
            Dimensionless coordinates in range 0:1.

    """
    if shape == 's': a = 3
    elif shape == 'c': a = 2
    else: a = 1
    
    poly = np.zeros(2 * n_points + 1)
    poly[0] = 1.
    polyx2 = np.zeros(n_points + 1)
    polyx2[0] = 1.
    
    for i in range(1, n_points + 1):
        p1 = _product([-n_points + j for j in range(i)])
        p2 = _product([n_points + (a/2) + j for j in range(1, i+1)])
        d1 = _product([(a/2) + j for j in range(i)])*math.factorial(i)
        poly[2*i] = p1*p2/d1
        polyx2[i] = p1*p2/d1
        
    poly = np.flip(poly)
    all_roots = np.roots(poly)
    positive_roots = np.sort(all_roots[all_roots>0])
    w_points = np.append(positive_roots,[1])
    x_points = w_points*(end-start) + start
    return {'w':w_points,'x':x_points, "poly":poly, "Px2":polyx2}

def collocation_matrices(w_points,shape = 's'):
    """
    Returns the matrices used in the collocation problem.

    Parameters
    ----------
    w_points : 1d array like
        Dimensionless coordinates of the collocation problem.
    shape : str, optional
        Geometry of the problem.
        's': spherical
        'c': cilindrical
        other: regular slab
            The default is 's'.

    Returns
    -------
    dict
        Contains the matrices used in the collocation problem.
        'A', 'B', 'Q', 'R', 'T', 'W'

    """
    if shape == 's': a = 3
    elif shape == 'c': a = 2
    else: a = 1
    
    w_points = np.array(w_points)
    _len = len(w_points)

    Q = np.empty([_len,_len])
    for n in range(0,_len):
        Q[:,n] = w_points**(2*n)
    
    R = np.zeros([_len,_len])
    for n in range(1,_len):
        R[:,n] = 2*n*w_points**(2*n-1)
    
    T = np.zeros([_len,_len])
    for n in range(1,_len):
        T[:,n] = 2*n*(2*n-1)*w_points**(2*n-2) + (a-1)*2*n*w_points**(2*n-2)
    
    A = np.matmul(R,np.linalg.inv(Q))
    B = np.matmul(T,np.linalg.inv(Q))
    
    WF = [(1**(2*n+a))/(2*n+a)-(0**(2*n+a))/(2*n+a) for n in range(_len)]
    W = np.matmul(WF,np.linalg.inv(Q))
    
    return {'A':A, 'B':B, 'Q':Q, 'R':R, 'T':T, 'W':W}

def f_lap_sist(ft, x_points, y, args=(), kwargs={}, scale_array=None, mode='default'):
    """
    Solves the transport equation lap(y) = fy(x,y)

    Parameters
    ----------
    ft : Callable
        Vector function. Takes as arguments x, y, *args, **kwargs.
        Returns an array containing the values of the trasport equations of the variables.
        Note: the solver passes the function as ft(x,y.T,*args,**kwargs).
    x_points : 1d array
        Coordinates of the collocation points in original dimensions.
    y : 1d array of shape[M] or 2d array of shape[MxN]
        Values of the dependent variables of the problem.
        M : number of internal collocation points + 1
        N : numer of independent variables
    args : tuple, optional
        Additional arguments passed to the function. The default is ().
    kwargs : dict, optional
        Additional arguments passed to the function. The default is {}.
    scale_array : float, int, or 1d array like, optional
        Scale factor for the y variable when solving the sistem. The default is None.
    mode : mode : str, optional
        containing either:
            'default'
            'scaled'
            'bound-scaled'
            'integrate'
            The default is 'default'.

    Raises
    ------
    TypeError
        Unavailable method.

    Returns
    -------
    lap : 1d array of shape[M] or 2d array of shape[MxN]
        Matrix MxN containing the trasnport equations for each variable at each point.
        M : number of collocation points + 1
        N : dimension of the array returned by ft. Recomended to be the number of dependent variables.

    """
    
    if not (scale_array is None):
        mode = 'scaled'
    if mode == 'default':
        lap = (ft(x_points,y.T, *args, **kwargs) * x_points[-1]**2).T
    elif mode == 'scaled':
        lap = (ft(x_points, y.T, *args, **kwargs) * x_points[-1]**2).T / scale_array
    elif mode == 'integrate':
        lap = (ft(x_points, y.T, *args, **kwargs)).T
    else:
        raise TypeError("Method unavailable")
    return lap

def _f_res_sist(ft, x_points, y, B, scale_array=None, args=(), kwargs={}, mode='default'):
    
    if not (scale_array is None):
        mode = 'scaled'
        y_scaled = y / scale_array
    else:
        y_scaled = y
        
    lap = f_lap_sist(ft, x_points, y[:-1,:], args, kwargs, scale_array, mode)
    BY = np.matmul(B[:-1, :], y_scaled[:, :])
    
    return BY.flatten() - lap.flatten()

def _f_obj_sist(y_in_flat, ft, x_points, yb, B, scale_array=None,
                args=(), kwargs={}, mode='bound-scaled'):
    
    if mode == 'bound-scaled':
        scale_array = np.array(yb)
        mode = 'scaled'
        
    y = np.append(y_in_flat, yb).reshape(len(B), len(yb))
    
    return _f_res_sist(ft, x_points, y, B, scale_array, args, kwargs, mode)

def get_y_sist(ft, x_points, yb, B, args=(), kwargs={}, scale_array=None,
               mode='bound-scaled', guess_0='bounds', root_kwargs={'method':'lm'}):
    """
    Function to return values of y of a multivariate internal orthogonal collocation problem in the format:
        lap(y) = f(x,y)

    Parameters
    ----------
    ft : Callable
        Vector function. Takes as arguments x,y,*args,**kwargs.
        Returns an array containing the values of the trasport equations of the variables.
        Note: the solver passes the function as ft(x,y.T,*args,**kwargs).
    x_points : 1d array like
        Values of the orthogonal collocation points in the independent variable.
    yb : int, float, or 1d array like
        Values of parameters y in the boundary of the problem.
    B : 2d array like
        B matrix of the orthogonal collocation problem.
    args : tuple, optional
        Additional arguments passed to the function. The default is ().
    kwargs : dict, optional
        Additional arguments passed to the function. The default is {}.
    scale_array : int, float, or 1d array like, optional
        Scale factor for the y variable when solving the sistem. The default is None.
    mode : str, optional
        containing either 'default', 'scaled', or 'bound-scaled'. The default is 'bound-scaled'.
    guess_0 : str or 1d array or 2d array like, optional
        initial guess for the values of y in the internal collocation points. The default is 'bounds'.
    root_kwargs : dict, optional
        Additional arguments passed to the scipy root solver. The default is {'method':'lm'}.

    Returns
    -------
    dict
        y : 1d array of shape[M] or 2d array of shape[MxN].
            Values of the dependent variables of the problem.
            M : number of internal collocation points + 1
            N : numer of independent variables
        y_scaled : 1d array or 2d array
            Values of variables y divided by the scale factor.
        'minimize' : dict like
            solution returned by the scipy root function.

    """
    yb = np.array(yb)
    
    if type(guess_0) == str:
        if guess_0 == 'bounds':
            guess_0 = np.tile(yb, B.shape[0] - 1)
    else:
        guess_0 = guess_0.flatten()
        
    sol = root(_f_obj_sist, guess_0,
               args=(ft, x_points, yb, B, scale_array, args, kwargs, mode),
               **root_kwargs)
    
    y = np.append(sol.x, np.array(yb))
    
    if y.size > B.shape[0]:
        y = y.reshape(B.shape[0],-1)
    if mode == 'bound-scaled':
        scale_array = np.array(yb)
    elif (scale_array is None):#scale_array is None
        scale_array = np.ones(yb.size)
        
    y_scaled = y / scale_array
    return {'y_scaled':y_scaled, 'y':y, 'minimize':sol}

def _f_fy(f, y, args=(), kwargs={}):
    return f(y.T, *args, **kwargs).T

def r_rates_sys(r, y, W, args=(), kwargs={}):
    """
    Returns the reaction rates as function of y over the volume.

    Parameters
    ----------
    r : Callable
        Vector function. Takes as arguments y, *args, **kwargs.
        Note: the solver passes the function as r(y.T,*args,**kwargs).
    y : 1d array of shape[M] or 2d array of shape[MxN]
        Values of the dependent variables of the problem.
        M : number of internal collocation points + 1
        N : numer of independent variables
    W : 1d array
        W array from orthogonal collocation problem.
    args : tuple, optional
        Additional arguments passed to the function. The default is ().
    kwargs : dict, optional
        Additional arguments passed to the function. The default is {}.

    Returns
    -------
    dict
        'IF' : Integral per surface area.
        'surf' : Integral based on rate at surface per surface area.
        'eff' : 'IF'/'surf'

    """
    y = np.array(y)
    W = np.array(W)
    r_rates = _f_fy(r, y, args, kwargs)
    r_rates_surf = np.tile(r_rates[-1], [W.size, 1])
    int_func = W.dot(r_rates)#np.matmul(W, r_rates)
    int_surf = W.dot(r_rates_surf)
    return {'IF':int_func, 'surf':int_surf, 'eff': int_func/int_surf}

def integrate_V(fun, x_points, y, W, args=(), kwargs={}):
    """
    Returns the integral of a function over the volume per surface area.

    Parameters
    ----------
    fun : Callable
        Vector function. Takes as arguments x, y, *args, **kwargs.
        Note: the solver passes the function as fun(x,y.T,*args,**kwargs).
    x_points : 1d array
        Values of the collocation points in the original coordinates.
    y : 1d array of shape[M] or 2d array of shape[MxN]
        Values of the dependent variables of the problem.
        M : number of internal collocation points + 1
        N : numer of independent variables
    W : 1d array
        W array from orthogonal collocation problem.
    args : tuple, optional
        Additional arguments passed to the function. The default is ().
    kwargs : dict, optional
        Additional arguments passed to the function. The default is {}.

    Returns
    -------
    integral : float or 1d array
        Integral of fun over the volume per surface area.

    """
    f = f_lap_sist(fun, x_points, y, args, kwargs, mode='integrate')
    integral = np.matmul(W, f)
    return integral

def solve_sys_transport(ft, start, end, yb, n_points, shape='s', **options):
    """
    Solves a system of transport equations for symmetric problems, based on the laplacian function.
    lap(y) = fun(x,y)
    Internal collocation problem.

    Parameters
    ----------
    ft : Callable
        Vector function. Takes as arguments x,y,*args,**kwargs.
        Returns an array containing the values of the trasport equations of the variables.
        Note: the solver passes the function as ft(x,y.T,*args,**kwargs).
    start : float or int
        Starting coordinates of the independent variable in the collocation problem.
    end : float or int
        Ending coordinates of the independent variable in the collocation problem.
    yb : int, float, or 1d array like
        Values of parameters y in the boundary of the problem.
    n_points : int
        Number of internal collocation points.
    shape : str, optional
        Geometry of the problem.
        's': spherical
        'c': cilindrical
        other: regular slab
            The default is 's'.
    **options : key = value
        Additional arguments passed to the solver:
            scale_array : float or 1d array like, optional
                Scale factor for the y variable when solving the sistem. The default is None.
            mode : str, optional
                containing either 'default', 'scaled', or 'bound-scaled'. The default is 'bound-scaled'.
            guess_0 : str or 2d array like, optional
                initial guess for the values of y in the internal collocation points. The default is 'bounds'.
            root_kwargs : dict, optional
                Additional arguments passed to the scipy root solver. The default is {'method':'lm'}.

    Returns
    -------
    sol : dict
        'matrices','x','w','y_scaled','y','minimize'.

    """
    
    sol = {}
    cpoints = collocation_points(start, end, n_points, shape)
    w_points = cpoints['w']
    x_points = cpoints['x']
    poly = cpoints['poly']
    polyx2 = cpoints['Px2']
    sol['w'] = w_points
    sol['x'] = x_points
    sol['poly'] = poly
    sol['Px2'] = polyx2
    sol['matrices'] = collocation_matrices(w_points, shape)
    y_dict = get_y_sist(ft, x_points, yb, sol['matrices']['B'], **options)
    sol['y_scaled'] = y_dict['y_scaled']
    sol['y'] = y_dict['y']
    sol['minimize'] = y_dict['minimize']
    return sol

