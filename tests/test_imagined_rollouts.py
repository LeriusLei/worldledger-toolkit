import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from sbcc.imagined_rollouts import categorical_cem, imagine_plans, lambda_return


class ImaginationTests(unittest.TestCase):

    def test_terminal_blocks_future_reward_and_boundary_bootstraps(self):
        z = np.zeros((1, 4))
        term = np.array([[0., 0., 1., 0.]])
        reward = np.array([[0., 1., 2., 100.]])
        actual = lambda_return(z, term, reward, z, z, .5, 1.)
        self.assertEqual(actual[0, 0], 2.)
        last = np.array([[0., 1., 0., 0.]])
        boot = np.array([[0., 8., 0., 0.]])
        self.assertEqual(lambda_return(last, z, reward, z, boot, .5, 1.)[0, 0], 5.)

    def test_rollouts_keep_plan_shared_and_stop_on_terminal(self):
        calls = []
        def transition(s, a):
            calls.append(a)
            return s + a
        model = SimpleNamespace(components={'transition_model': transition,
            'reward_model': lambda s, a, n: (float(n >= 2), n >= 2)})
        result = imagine_plans(model, [0, 1], [[1, 1, 1]], gamma=.5)
        np.testing.assert_array_equal(result['plan_particle_returns'], [[.5, 1.]])
        self.assertEqual(result['model_transition_calls'], 3)
        self.assertEqual(calls, [1, 1, 1])

    def test_categorical_cem_finds_discrete_plan_and_is_reproducible(self):
        def score(plans):
            return (plans == np.array([2, 1, 2, 1])).sum(axis=1)
        kwargs = dict(actions=range(3), horizon=4, population=96,
                      iterations=4, elites=12, smoothing=.25,
                      probability_floor=.05, seed=17)
        first = categorical_cem(score, **kwargs)
        second = categorical_cem(score, **kwargs)
        self.assertEqual(first['best_plan'], [2, 1, 2, 1])
        self.assertEqual(first['best_plan'], second['best_plan'])
        self.assertEqual(first['evaluated_plans'], 384)
        np.testing.assert_array_equal(first['action_probabilities'],
                                      second['action_probabilities'])

    def test_categorical_cem_rejects_invalid_floor(self):
        with self.assertRaises(ValueError):
            categorical_cem(lambda p: np.zeros(len(p)), actions=range(6),
                            horizon=2, probability_floor=1/6)


if __name__ == '__main__':
    unittest.main()
