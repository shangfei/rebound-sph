import rebound
import unittest
import datetime
import numpy as np
from ctypes import byref, POINTER, c_double

class TestDerivatives(unittest.TestCase):
    paramkeys = ["m","a","l","k","h","ix","iy"]
    paramlist = [ 
            [1e-3,      1.,    0.,         0.,      0.,     0.0,     0.0],
            [1e-3,      1.,    0.3,        0.02,    0.1,    0.0,     0.0],
            [1e-6,      2.,    0.33,       0.0132,  0.02,   0.126,   0.14],
            [234.3e-6,  1.7567,0.573,      0.572,   0.561,  0.056,   0.0091354],
            [1e-2,      1.7567,0.24573,    0.15472, 0.1561, 0.0056,  0.0013],
            [1e-7,      3.7567,0.523473,   0.23572, 0.00061,0.47256, 0.000024],
            [1e-7,      3.7567,0.523473,   0.23572, 0.00061,1.97,    0.0],
            [1e-7,      3.7567,0.523473,   0.23572, 0.00061,0.0,     1.97],
            ]

    def test_all_1st_order_full(self):
        for params in self.paramlist:
            for i,v in enumerate(self.paramkeys):

                param = dict(zip(self.paramkeys, params))
                simvp = rebound.Simulation()
                simvp.add(m=1.)
                simvp.add(**param)
                p0 = simvp.particles[0].copy()
                p1 = simvp.particles[1].copy()
                var_i = simvp.add_variation()
                var_i.vary(1,v)
                #simvp.move_to_com()
                dt = 0.
                simvp.integrate(dt)
                vec = np.zeros(7,dtype="float64")
                rebound.clibrebound.reb_derivatives_vx(vec.ctypes.data_as(POINTER(c_double)), c_double(dt),c_double(1.),p0,p1) 
                print(vec[i])
                print("vp" , var_i.particles[1].vx)

                #self.assertLess(abs(dp.m ),prec)
            print("---")
    
if __name__ == "__main__":
    unittest.main()
