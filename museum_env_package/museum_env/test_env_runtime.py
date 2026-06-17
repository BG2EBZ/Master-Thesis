import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from museum_env import env_runtime
from museum_env.env_state import RuntimeCache
from museum_env.metrics import VectorizedRollingWindow


class EnvRuntimeTests(unittest.TestCase):
    def _make_metrics(self, n_humans: int):
        return (
            VectorizedRollingWindow(window_steps=1, n_entities=n_humans),
            VectorizedRollingWindow(window_steps=1, n_entities=n_humans),
        )

    def _make_data(self, robot_pose, human_poses):
        xpos = np.zeros((1 + len(human_poses), 3), dtype=np.float32)
        qpos = np.zeros((3 + 3 * len(human_poses),), dtype=np.float32)
        xpos[0, 0] = float(robot_pose[0])
        xpos[0, 1] = float(robot_pose[1])
        qpos[2] = float(robot_pose[2])

        humans = []
        human_body_ids = []
        for idx, (x, y, yaw) in enumerate(human_poses):
            body_id = idx + 1
            qpos_idx = 3 + 3 * idx
            xpos[body_id, 0] = float(x)
            xpos[body_id, 1] = float(y)
            qpos[qpos_idx + 2] = float(yaw)
            humans.append(SimpleNamespace(qpos_idx=qpos_idx))
            human_body_ids.append(body_id)

        data = SimpleNamespace(xpos=xpos, qpos=qpos)
        return data, humans, human_body_ids

    def test_refresh_observation_snapshot_uses_precomputed_pairwise_distances(self):
        human_xy = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )
        robot_xy = np.array([0.0, 1.0], dtype=np.float32)
        cache = RuntimeCache()
        hh_metric, hr_metric = self._make_metrics(n_humans=3)
        _, pairwise_distances = env_runtime.compute_human_pairwise_geometry(human_xy)

        with patch(
            "museum_env.env_runtime.compute_human_pairwise_geometry",
            side_effect=AssertionError("unexpected pairwise recomputation"),
        ):
            observations = env_runtime.refresh_observation_snapshot(
                cache=cache,
                hh_distance_metric=hh_metric,
                hr_distance_metric=hr_metric,
                human_xy=human_xy,
                robot_xy=robot_xy,
                observation_update_period_steps=1,
                force=True,
                pairwise_distances=pairwise_distances,
            )

        np.testing.assert_allclose(
            observations.nearest_human_distance,
            np.array([0.5, 0.5, 1.5], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            observations.local_crowding_count_1m,
            np.array([1, 1, 0], dtype=np.int32),
        )
        np.testing.assert_allclose(
            observations.human_robot_distance,
            np.array([1.0, np.sqrt(1.25), np.sqrt(5.0)], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            observations.nearest_human_distance_mean_1s,
            observations.nearest_human_distance,
        )
        np.testing.assert_allclose(
            observations.human_robot_distance_mean_1s,
            observations.human_robot_distance,
        )

    def test_precomputed_social_repulsion_matches_expected_push(self):
        human_xy = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )
        pairwise_diff, pairwise_dist = env_runtime.compute_human_pairwise_geometry(human_xy)

        repulsion_from_pairwise = env_runtime._compute_social_repulsion_from_pairwise(
            pairwise_diff=pairwise_diff,
            pairwise_dist=pairwise_dist,
            social_distance=1.0,
            repulsion_gain=2.0,
        )
        repulsion_from_wrapper = env_runtime.compute_social_repulsion(
            human_xy=human_xy,
            social_distance=1.0,
            repulsion_gain=2.0,
        )

        expected = np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(repulsion_from_pairwise, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            repulsion_from_wrapper,
            repulsion_from_pairwise,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_build_world_frame_computes_pairwise_geometry_once(self):
        data, humans, human_body_ids = self._make_data(
            robot_pose=(1.0, -1.0, 0.25),
            human_poses=[
                (0.0, 0.0, 0.10),
                (0.5, 0.0, -0.20),
            ],
        )
        cache = RuntimeCache()
        hh_metric, hr_metric = self._make_metrics(n_humans=2)

        with patch(
            "museum_env.env_runtime.compute_human_pairwise_geometry",
            wraps=env_runtime.compute_human_pairwise_geometry,
        ) as geometry_helper:
            frame = env_runtime.build_world_frame(
                data=data,
                robot_body_id=0,
                humans=humans,
                human_body_ids=human_body_ids,
                cache=cache,
                hh_distance_metric=hh_metric,
                hr_distance_metric=hr_metric,
                observation_update_period_steps=1,
                social_distance=1.0,
                repulsion_gain=2.0,
                force_observations=True,
            )

        self.assertEqual(geometry_helper.call_count, 1)
        self.assertEqual(frame.robot_pose, (1.0, -1.0, 0.25))
        np.testing.assert_allclose(
            frame.pairwise_distances,
            np.array(
                [
                    [np.inf, 0.5],
                    [0.5, np.inf],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            frame.repulsion_vectors,
            np.array(
                [
                    [-1.0, 0.0],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
