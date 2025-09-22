import unittest
import numpy as np

import systems.linear as linear

class TestLinearSystems(unittest.TestCase):
    def __init__(self):
        self.a0 = np.array(
            [
                [1, -3, 2],
                [-2, 8, -3],
                [4, -6, 5]
            ],
            dtype=float
        )
        self.b0 = np.array([11, -15, 29], dtype=float)

        self.af = np.array(
            [
                [1, -3, 2],
                [0, 2, 1],
                [0, 0, -6]
            ],
            dtype=float
        )

        self.bf = np.array([29, 3.5, 1], dtype=float)

        self.cf = np.array([2, -1, 1], dtype=float)

        self.xf = np.array([0.5, 0.5, 6], dtype=float)

    def test_escalate_to_upper_triangular(self):
        a, b, c = linear.escalate_to_upper_triangular(self.a0, self.b0)
        del(b)
        del(c)
        self.assertTrue(np.allclose(a, np.triu(a))) # test shape
        self.assertTrue(np.allclose(a, self.af)) # test values
    
    def test_solve_upper_triangular(self):
        a, b, c = linear.escalate_to_upper_triangular(self.a0, self.b0)
        del(c)  # c não sera usado
        x = linear.solve_upper_triangular(a, b)
        self.assertTrue(np.allclose(x, self.xf)) # test values
    
    def test_gauss_elimination(self):
        x = linear.gauss_elimination(self.a0, self.b0)
        self.assertTrue(np.allclose(x, self.xf)) # test values
    
    def test_partial_pivot(self):
        x = linear.partial_pivot(self.a0, self.b0)
        self.assertTrue(np.allclose(x, self.xf)) # test values
    
    
if __name__ == "__main__":
    unittest.main()