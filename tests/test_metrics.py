import itertools
import unittest
import numpy as np
from audioseal_st.metrics import any_operating_point, operating_point


class MetricsTest(unittest.TestCase):
    def test_joint_matches_exhaustive_with_ties(self):
        rng = np.random.default_rng(14)
        for _ in range(50):
            pos = rng.integers(0, 5, (9, 2))
            neg = rng.integers(0, 5, (11, 2))
            alpha = .19
            candidates = [np.r_[np.inf, np.unique(np.r_[pos[:, i], neg[:, i]])] for i in range(2)]
            best = 0
            for a, p in itertools.product(*candidates):
                if ((neg[:, 0] >= a) | (neg[:, 1] >= p)).mean() <= alpha:
                    best = max(best, ((pos[:, 0] >= a) | (pos[:, 1] >= p)).mean())
            got = any_operating_point(pos, neg, alpha)
            self.assertAlmostEqual(got['tpr'], best)
            self.assertLessEqual(got['fpr'], alpha)

    def test_can_disable_bad_detector(self):
        neg = np.c_[np.ones(100), np.ones(100)*10]
        pos = np.c_[np.ones(50)*2, np.zeros(50)]
        joint = any_operating_point(pos, neg)
        self.assertEqual(joint['tpr'], 1)
        self.assertEqual(joint['fpr'], 0)
        self.assertGreaterEqual(joint['tpr'], operating_point(pos[:, 0], neg[:, 0])['tpr'])

    def test_reject_nan(self):
        with self.assertRaises(ValueError):
            any_operating_point([[1, np.nan]], [[0, 0]])


if __name__ == '__main__':
    unittest.main()
