import unittest
import numpy as np
from run_experiments import predict
from temperature_analysis import run_analysis as a
class TemperatureTests(unittest.TestCase):
    def test_coverage(self):
        d=a.load_data(); self.assertEqual(d.YEAR.tolist(),list(range(1901,2022))); self.assertFalse(d.isna().any().any())
    def test_split(self):
        d=a.load_data(); self.assertEqual(sum(d.YEAR<=2001),101); self.assertEqual(sum(d.YEAR>=2002),20)
    def test_folds(self):
        for start,end in a.FOLDS: self.assertLessEqual(end,2001); self.assertLess(start,end)
    def test_rmse(self): self.assertAlmostEqual(a.rmse([1,3],[1,1]),np.sqrt(2))
    def test_hinge(self): np.testing.assert_array_equal(a.features([1980,1981,1982],'Piecewise Ridge (1981)')[:,1],[0,0,1])
    def test_mean(self): np.testing.assert_allclose(predict('Training mean',None,[1,2],[2,4],[3,4]),[3,3])
    def test_linear(self): np.testing.assert_allclose(predict('Linear Regression',None,[1,2,3],[2,4,6],[4,5]),[8,10])
if __name__=='__main__': unittest.main()
