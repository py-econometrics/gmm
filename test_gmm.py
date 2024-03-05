import unittest

from gmm import GMMEstimator, GMMEstimatorTorch, GMMEstimatorScipy


class TestGMMEstimator(unittest.TestCase):
    def test_backend_initialization(self):
        with self.subTest("Scipy backend"):
            estimator = GMMEstimator(None, backend="scipy")
            self.assertIsInstance(estimator, GMMEstimatorScipy)

        with self.subTest("Torch backend"):
            estimator = GMMEstimator(None, backend="torch")
            self.assertIsInstance(estimator, GMMEstimatorTorch)

        with self.subTest("Nonexistent backend"):
            with self.assertRaises(ValueError):
                GMMEstimator(None, backend="NONEXISTENT")


class TestGMMEstimatorScipy(unittest.TestCase):
    # TODO
    pass


class TestGMMEstimatorTorch(unittest.TestCase):
    # TODO
    pass



if __name__ == '__main__':
    unittest.main()
