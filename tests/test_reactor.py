import pytest

from styrene.reactor import MultiBed


OUTPUT_RADIAL = {
    'Feb': 118.96732835279548,
    'Fst': 476.3628757874466,
    'Fh2': 362.8817916067565,
    'Fbz': 12.689711679068326,
    'Fme': 106.37708418068966,
    'Fto': 111.3450841806897,
    'Fee': 12.396711679068327,
    'Fh2o': 7777.0,
    'T': 890.9534920114189,
    'P': 1.1662099639142283
}

OUTPUT_AXIAL = {
    'Feb': 106.98975460107222,
    'Fst': 511.82452049620645,
    'Fh2': 420.7954330142763,
    'Fbz': 11.657637420790909,
    'Fme': 83.92508748193008,
    'Fto': 88.8930874819301,
    'Fee': 11.364637420790906,
    'Fh2o': 7777.0,
    'T': 882.620671637914,
    'P': 0.4805513834190388
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
