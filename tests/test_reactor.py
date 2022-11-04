import pytest

from styrene.reactor import MultiBed


OUTPUT_RADIAL = {
    'Feb': 118.9548497335193,
    'Fst': 476.3384937994504,
    'Fh2': 362.82247716089336,
    'Fbz': 12.691639828473388,
    'Fme': 106.4120166385571,
    'Fto': 111.38001663855712,
    'Fee': 12.398639828473389,
    'Fh2o': 7777.0,
    'T': 890.9604078444497,
    'P': 1.1662097624638503
}

OUTPUT_AXIAL = {
    'Feb': 106.98343338475202,
    'Fst': 511.81187118960656,
    'Fh2': 420.7647756316431,
    'Fbz': 11.658599867676413,
    'Fme': 83.94309555796386,
    'Fto': 88.91109555796388,
    'Fee': 11.36559986767641,
    'Fh2o': 7777.0,
    'T': 882.6234145119014,
    'P': 0.48054904799882475
}


IVP_RTOL = 1e-6


def test_axial():

    test_reac = MultiBed()
    test_reac.add_axial_bed(72950, Pmin=0.501, ivp_rtol=IVP_RTOL, terminal=True, heterogeneous=True, n_points=6)
    test_reac.set_inlet(T=886, P=1.35)
    test_reac.add_axial_bed(82020, Pmin=0.501, ivp_rtol=IVP_RTOL, terminal=True, heterogeneous=True, n_points=6)
    test_reac.add_axial_bed(78330, Pmin=0.501, ivp_rtol=IVP_RTOL, terminal=True, heterogeneous=True, n_points=6)
    test_reac.add_resets(2, T=898.2)
    test_reac.add_resets(3, T=897.6)
    
    test_reac.solve(method="LSODA")
    sol = test_reac.outlet
    
    for key, val in sol.items():
        assert abs(val - OUTPUT_AXIAL[key]) <= 1e-6


def test_radial():
    
    test_reac = MultiBed()
    test_reac.add_radial_bed(72950, Pmin=0.501, ivp_rtol=IVP_RTOL, terminal=True, heterogeneous=True, n_points=6)
    test_reac.set_inlet(T=886, P=1.25)
    test_reac.add_radial_bed(82020, Pmin=0.50005, ivp_rtol=IVP_RTOL, terminal=True, heterogeneous=True, n_points=6)
    test_reac.add_radial_bed(78330, Pmin=0.50, ivp_rtol=IVP_RTOL, terminal=True, heterogeneous=True, n_points=6)
    test_reac.add_resets(2, T=898.2)
    test_reac.add_resets(3, T=897.6)
    
    test_reac.solve(method="LSODA")
    sol = test_reac.outlet
    
    for key, val in sol.items():
        assert abs(val - OUTPUT_RADIAL[key]) <= 1e-6

