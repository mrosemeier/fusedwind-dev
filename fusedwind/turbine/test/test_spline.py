
import unittest
import numpy as np
from openmdao.api import Problem, Group
from fusedwind.turbine.geometry import FFDSpline

expected_a = np.array([ 0.        ,  0.16474038,  0.32586582,  0.47988383,  0.62354352,
        0.75394814,  0.86865795,  0.96578062,  1.04404672,  1.10286821,
        1.14237834,  1.16345188,  1.1677051 ,  1.15747556,  1.13578223,
        1.10626695,  1.07311899,  1.04098464,  1.01486427,  1.        ])
expected_b = np.array([  0.00000000e+00,   1.64594590e-01,   3.24699469e-01,
         4.75947393e-01,   1.57524836e+00,   1.51708781e+00,
         1.46277016e+00,   1.40782756e+00,   1.34840732e+00,
         1.28133810e+00,   1.20416987e+00,   1.11519411e+00,
         1.01344381e+00,   8.98673258e-01,   7.71318112e-01,
         6.32436944e-01,   4.83635740e-01,   3.26977498e-01,
         1.64879344e-01,   1.22464680e-16])
expected_c = np.array([  0.00000000e+00,   1.64594590e-01,   3.24699469e-01,
         4.75947393e-01,   1.56250027e+00,   1.45317543e+00,
         1.36994579e+00,   1.31004849e+00,   1.27133932e+00,
         1.25235550e+00,   1.25235550e+00,   1.27133932e+00,
         1.31004849e+00,   1.36994579e+00,   1.45317543e+00,
         1.56250027e+00,   4.75947393e-01,   3.24699469e-01,
         1.64594590e-01,   1.22464680e-16])


def configure():

    p = Problem(root=Group())
    s = np.linspace(0, 1, 20)
    P = np.sin(np.linspace(0, 1, 20)*np.pi)
    a = p.root.add('spla', FFDSpline('a', s, P, np.linspace(0, 1, 4)), promotes=['*'])
    b = p.root.add('splb', FFDSpline('b', s, P, np.linspace(0.2, 1, 4)), promotes=['*'])
    c = p.root.add('splc', FFDSpline('c', s, P, np.linspace(0.2, 0.8, 4)), promotes=['*'])
    p.setup()
    return p

class TestFFDSpline(unittest.TestCase):


    def test_it(self):
        p = configure()
        p['a_C'] = np.array([0, 0, 0, 1.])
        p['b_C'] = np.array([1, 0, 0, 0.])
        p['c_C'] = np.array([1, 0, 0, 1.])
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['a'], expected_a, decimal=6), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['b'], expected_b, decimal=6), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['c'], expected_c, decimal=6), None)


if __name__ == '__main__':

    unittest.main()
    # import matplotlib.pylab as plt
    # p = configure()
    # p['a_C'] = np.array([0, 0, 0, 1.])
    # p['b_C'] = np.array([1, 0, 0, 0.])
    # p['c_C'] = np.array([1, 0, 0, 1.])
    # p.run()
    # plt.plot(p.root.spla.s, p.root.spla.Pinit)
    # plt.plot(p.root.spla.s, p['a'])
    # plt.plot(p.root.splb.s, p['b'])
    # plt.plot(p.root.splc.s, p['c'])
    # plt.show()
