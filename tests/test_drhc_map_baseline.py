"""MAP is candidate-only and can disagree with robust minimax on one pool."""

import unittest

from worldledger.evaluator.baselines.map_policy import choose_map, evaluate_candidate


def source_for(action: str, reward_value: int) -> str:
    return ("def transition(state, action):\n return state\n"
            "def observation(state, action, next_state):\n return ('still',)\n"
            f"def reward(state, action, next_state):\n return {reward_value} if action == '{action}' else 0\n")


class MapBaselineTests(unittest.TestCase):
    def test_candidate_only_horizon_five_and_minimax_disagreement(self):
        actions = ("left", "right")
        left_world = evaluate_candidate("left-world", source_for("left", 10), (0, 0), actions,
                                        (), ((0, 0),), 5)
        right_world = evaluate_candidate("right-world", source_for("right", 9), (0, 0), actions,
                                         (), ((0, 0),), 5)
        pool = (left_world, right_world)
        map_result = choose_map(pool, actions)
        robust_index = min(range(2), key=lambda index: (
            max(max(view.action_values) - view.action_values[index] for view in pool), index))
        self.assertEqual(map_result.action, "right")
        self.assertEqual(actions[robust_index], "left")
        self.assertEqual(map_result.posterior_class_count, 2)

    def test_duplicate_is_one_class_and_empty_pool_fails(self):
        actions = ("left", "right")
        source = source_for("left", 10)
        first = evaluate_candidate("a", source, (0, 0), actions, (), ((0, 0),), 5)
        duplicate = evaluate_candidate("b", "# distinct source\n" + source,
                                       (0, 0), actions, (), ((0, 0),), 5)
        selection = choose_map((duplicate, first), actions)
        self.assertEqual(selection.posterior_class_count, 1)
        self.assertEqual(selection.selected_candidate_id, "a")
        with self.assertRaises(ValueError):
            choose_map((), actions)


if __name__ == "__main__":
    unittest.main()
