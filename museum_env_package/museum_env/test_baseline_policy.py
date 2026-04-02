from importlib import resources
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from museum_env.env import (
    HUMAN_SPAWN_MIN_DISTANCE,
    HUMAN_SPAWN_MIN_ROBOT_DISTANCE,
    INACTIVE_HUMAN_PARK_X,
    MOVE_BACK_SPEED,
    MuseumEnv,
    POST_EXPLANATION_HOLD_RESUME_DISTANCE,
    POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD,
    ROBOT_EXPLANATION_LABEL_GROUP,
    ROBOT_FOLLOWME_LABEL_GROUP,
    ROBOT_NEED_SPACE_LABEL_GROUP,
)
from museum_env.human import (
    HUMAN_WALL_FOOTPRINT_RADIUS,
    Human,
    HumanMode,
    HumanProfile,
)
from museum_env.map_layouts import AxisAlignedRect, DEFAULT_MUSEUM_LAYOUT, MapLayout
from museum_env.robot import RobotCallbackPhase, RobotEmotion


class _FixedRandom:
    def __init__(self, value):
        self.value = float(value)

    def random(self):
        return self.value


class TestSimplifiedTriggerProbabilities(unittest.TestCase):
    def _make_env(self, **kwargs):
        defaults = {
            "distracted_lambda_max_nd_per_sec": 0.0,
            "distracted_lambda_max_normal_per_sec": 0.0,
        }
        defaults.update(kwargs)
        return MuseumEnv(
            render_mode=None,
            enable_event_logs=False,
            strict_action_validation=True,
            **defaults,
        )

    def test_robot_default_speed_and_move_back_speed(self):
        env = self._make_env()
        try:
            self.assertEqual(env.robot.v_max, 1.0)
            self.assertEqual(MOVE_BACK_SPEED, 0.6)
        finally:
            env.close()

    def _assert_in_walkable(self, human, xy, margin=HUMAN_WALL_FOOTPRINT_RADIUS, tol=1e-5):
        xy_arr = np.array(xy, dtype=np.float32)
        projected = human.map_layout.project_point(xy_arr, margin)
        self.assertLessEqual(float(np.linalg.norm(xy_arr - projected)), float(tol))

    def _set_human_pose(self, env, human, x, y, yaw):
        env.data.qpos[human.qpos_idx : human.qpos_idx + 3] = np.array([x, y, yaw], dtype=np.float32)
        mujoco.mj_forward(env.model, env.data)
        if human.body_id is None:
            human.body_id = env.model.body(human.body_name).id

    def _set_robot_pose(self, env, x, y, yaw):
        current_world_x, current_world_y, _ = env._get_robot_pose()
        current_qpos_x = float(env.data.qpos[0])
        current_qpos_y = float(env.data.qpos[1])
        base_offset = np.array(
            [current_world_x - current_qpos_x, current_world_y - current_qpos_y],
            dtype=np.float32,
        )
        env.data.qpos[0:2] = np.array([x, y], dtype=np.float32) - base_offset
        env.data.qpos[2] = float(yaw)
        mujoco.mj_forward(env.model, env.data)

    def _assert_pairwise_distance(self, xy_points, min_distance, tol=1e-6):
        for idx in range(len(xy_points)):
            for jdx in range(idx + 1, len(xy_points)):
                dist = float(np.linalg.norm(xy_points[idx] - xy_points[jdx]))
                self.assertGreaterEqual(dist + tol, float(min_distance))

    @staticmethod
    def _dense_social_repulsion_reference(human_xy, social_distance: float, repulsion_gain: float):
        human_xy = np.asarray(human_xy, dtype=np.float32)
        if not human_xy.size or social_distance <= 1e-6:
            return np.zeros((human_xy.shape[0], 2), dtype=np.float32)

        repulsion_vectors = np.zeros((human_xy.shape[0], 2), dtype=np.float32)
        for idx in range(human_xy.shape[0]):
            diff = human_xy[idx] - human_xy
            neighbor_dist = np.linalg.norm(diff, axis=1)
            mask = (neighbor_dist > 1e-6) & (neighbor_dist < social_distance)
            if np.any(mask):
                directions = diff[mask] / neighbor_dist[mask][:, None]
                strengths = (social_distance - neighbor_dist[mask]) / social_distance
                repulsion_vectors[idx] = repulsion_gain * (directions * strengths[:, None]).sum(axis=0)
        return repulsion_vectors

    @staticmethod
    def _dense_nearest_human_distance_reference(human_xy):
        human_xy = np.asarray(human_xy, dtype=np.float32)
        n_humans = int(human_xy.shape[0])
        if n_humans == 0:
            return np.zeros((0,), dtype=np.float32)
        if n_humans == 1:
            return np.array([np.nan], dtype=np.float32)

        pairwise_dist = np.linalg.norm(human_xy[:, None, :] - human_xy[None, :, :], axis=2)
        np.fill_diagonal(pairwise_dist, np.inf)
        return np.min(pairwise_dist, axis=1).astype(np.float32)

    @staticmethod
    def _dense_local_crowding_count_1m_reference(human_xy):
        human_xy = np.asarray(human_xy, dtype=np.float32)
        n_humans = int(human_xy.shape[0])
        if n_humans == 0:
            return np.zeros((0,), dtype=np.int32)

        pairwise_dist = np.linalg.norm(human_xy[:, None, :] - human_xy[None, :, :], axis=2)
        np.fill_diagonal(pairwise_dist, np.inf)
        return np.count_nonzero(pairwise_dist < 1.0, axis=1).astype(np.int32)

    @staticmethod
    def _dense_human_robot_distance_reference(human_xy, robot_xy):
        human_xy = np.asarray(human_xy, dtype=np.float32)
        if human_xy.size == 0:
            return np.zeros((0,), dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        return np.linalg.norm(human_xy - robot_xy[None, :], axis=1).astype(np.float32)

    def _assert_in_room_a(self, xy, margin=HUMAN_WALL_FOOTPRINT_RADIUS, tol=1e-6):
        room_a_rect = DEFAULT_MUSEUM_LAYOUT.get_spawn_rects(margin)[0]
        x = float(xy[0])
        y = float(xy[1])
        self.assertGreaterEqual(x + tol, float(room_a_rect.xmin))
        self.assertLessEqual(x - tol, float(room_a_rect.xmax))
        self.assertGreaterEqual(y + tol, float(room_a_rect.ymin))
        self.assertLessEqual(y - tol, float(room_a_rect.ymax))

    def _set_callback_state(
        self,
        env,
        target_idx,
        *,
        phase,
        attempt_index=1,
        cue_elapsed_steps=0,
        cue_total_steps=None,
        response_sampled=False,
        cue_completed_this_step=False,
    ):
        if cue_total_steps is None:
            cue_total_steps = env._get_callback_cue_steps()
        env.callback_active_target_idx = int(target_idx)
        env.robot.callback_active = True
        env.robot.callback_target_idx = int(target_idx)
        env.robot.callback_target_xy = np.array([2.4, 0.0], dtype=np.float32)
        env.robot.callback_attempt_index = int(attempt_index)
        env.robot.callback_phase = str(phase)
        env.robot.callback_cue_total_steps = int(cue_total_steps)
        env.robot.callback_cue_elapsed_steps = int(cue_elapsed_steps)
        env.robot.callback_response_sampled = bool(response_sampled)
        env.robot.callback_cue_completed_this_step = bool(cue_completed_this_step)
        env.robot.callback_turn_done = phase != RobotCallbackPhase.TURN
        env.robot.mode = "callback"

    def test_person1_defaults_to_neurodivergent_profile(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=1)
            self.assertEqual(env.humans[0].profile, HumanProfile.NEURODIVERGENT)
            for human in env.humans[1:]:
                self.assertEqual(human.profile, HumanProfile.NORMAL)
        finally:
            env.close()

    def test_active_human_counts_supported_for_5_10_and_15(self):
        for n_humans in (5, 10, 15):
            env = self._make_env(
                n_humans=n_humans,
                impatient_prob=0.0,
                overwhelmed_wait_trigger_prob=0.0,
                attack_wait_trigger_prob=0.0,
            )
            try:
                obs, info = env.reset(seed=100 + n_humans)
                self.assertEqual(obs.shape, (4,))
                self.assertEqual(info, {})
                self.assertEqual(len(env.humans), n_humans)
                self.assertEqual(env.nu, 3 + (3 * n_humans))
                self.assertEqual(env.action_space.shape, (3 + (3 * n_humans),))
                step_out = env.step(None)
                self.assertEqual(len(step_out), 5)
            finally:
                env.close()

    def test_n_humans_validation_rejects_values_outside_supported_capacity(self):
        with self.assertRaises(ValueError):
            self._make_env(n_humans=0)
        with self.assertRaises(ValueError):
            self._make_env(n_humans=16)

    def test_random_reset_spawns_active_humans_in_room_a_and_separated(self):
        env = self._make_env(
            n_humans=15,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=123)
            robot_xy = np.array(env._get_robot_pose()[:2], dtype=np.float32)
            human_xy = env._get_human_poses()[:, :2]
            self.assertEqual(human_xy.shape, (15, 2))
            for human, xy in zip(env.humans, human_xy):
                self._assert_in_walkable(human, xy)
                self._assert_in_room_a(xy)
                self.assertGreaterEqual(
                    float(np.linalg.norm(xy - robot_xy)) + 1e-6,
                    float(HUMAN_SPAWN_MIN_ROBOT_DISTANCE),
                )
                self._assert_in_room_a(human.current_waypoint)
            self._assert_pairwise_distance(human_xy, HUMAN_SPAWN_MIN_DISTANCE)
        finally:
            env.close()

    def test_setting_human_qpos_updates_world_xy_without_xml_base_offset(self):
        env = self._make_env(
            n_humans=5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=321)
            targets = [
                np.array([1.25, 1.75, 0.0], dtype=np.float32),
                np.array([8.5, 9.0, 0.25], dtype=np.float32),
                np.array([5.5, 4.0, -0.5], dtype=np.float32),
            ]
            for human, target in zip(env.humans[:3], targets):
                env.data.qpos[human.qpos_idx : human.qpos_idx + 3] = target
            mujoco.mj_forward(env.model, env.data)

            for human, target in zip(env.humans[:3], targets):
                world_xy = env.data.xpos[human.body_id, :2]
                np.testing.assert_allclose(world_xy, target[:2], atol=1e-6)
        finally:
            env.close()

    def test_human_walkable_sampling_api_returns_walkable_points(self):
        for _ in range(200):
            sampled_xy = DEFAULT_MUSEUM_LAYOUT.sample_walkable_point(HUMAN_WALL_FOOTPRINT_RADIUS, rng=np.random)
            probe_human = Human("probe", "person1", qpos_idx=3, max_speed=1.0)
            self._assert_in_walkable(probe_human, sampled_xy)

    def test_human_room_a_sampling_api_uses_env_rng_reproducibly(self):
        env_a = self._make_env(n_humans=5)
        env_b = self._make_env(n_humans=5)
        try:
            env_a.reset(seed=555)
            env_b.reset(seed=555)
            samples_a = [
                DEFAULT_MUSEUM_LAYOUT.sample_spawn_point(HUMAN_WALL_FOOTPRINT_RADIUS, rng=env_a.np_random)
                for _ in range(10)
            ]
            samples_b = [
                DEFAULT_MUSEUM_LAYOUT.sample_spawn_point(HUMAN_WALL_FOOTPRINT_RADIUS, rng=env_b.np_random)
                for _ in range(10)
            ]
            for a, b in zip(samples_a, samples_b):
                np.testing.assert_allclose(a, b, atol=1e-7)
        finally:
            env_a.close()
            env_b.close()

    def test_custom_map_layout_injection_overrides_waypoints_and_spawn_geometry(self):
        custom_layout = MapLayout(
            name="test_custom_layout",
            default_xml_asset="museum_scene.xml",
            walkable_rects=(AxisAlignedRect(0.0, 4.0, 0.0, 4.0),),
            spawn_rects=(AxisAlignedRect(1.0, 2.0, 1.0, 2.0),),
            robot_waypoints=((1.5, 1.5), (3.0, 3.0)),
            metadata={"rooms": ("test_room",)},
        )
        with resources.path("museum_env.assets", "museum_scene.xml") as xml_file:
            env = self._make_env(
                map_layout=custom_layout,
                xml_path=str(xml_file),
                n_humans=1,
                impatient_prob=0.0,
                overwhelmed_wait_trigger_prob=0.0,
                attack_wait_trigger_prob=0.0,
            )
        try:
            env.reset(seed=556)
            self.assertIs(env.map_layout, custom_layout)
            self.assertEqual(tuple(env.robot.waypoints), custom_layout.robot_waypoints)
            spawn_rect = custom_layout.get_spawn_rects(HUMAN_WALL_FOOTPRINT_RADIUS)[0]
            for human in env.humans:
                self.assertIs(human.map_layout, custom_layout)
                pose_xy = np.array(env.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
                self.assertGreaterEqual(float(pose_xy[0]), float(spawn_rect.xmin))
                self.assertLessEqual(float(pose_xy[0]), float(spawn_rect.xmax))
                self.assertGreaterEqual(float(pose_xy[1]), float(spawn_rect.ymin))
                self.assertLessEqual(float(pose_xy[1]), float(spawn_rect.ymax))
        finally:
            env.close()

    def test_segment_rect_interval_matches_expected_on_single_rect(self):
        rect = AxisAlignedRect(0.0, 1.0, 0.0, 1.0)
        interval = DEFAULT_MUSEUM_LAYOUT._segment_rect_interval(
            start_xy=np.array([-1.0, 0.5], dtype=np.float32),
            end_xy=np.array([2.0, 0.5], dtype=np.float32),
            rect=rect,
        )
        self.assertIsNotNone(interval)
        self.assertAlmostEqual(interval[0], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(interval[1], 2.0 / 3.0, places=6)

    def test_segment_walkable_accepts_motion_across_touching_rectangles(self):
        layout = MapLayout(
            name="touching_rects",
            default_xml_asset="museum_scene.xml",
            walkable_rects=(
                AxisAlignedRect(0.0, 1.0, 0.0, 1.0),
                AxisAlignedRect(1.0, 2.0, 0.0, 1.0),
            ),
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.5, 0.5),),
        )
        self.assertTrue(
            layout.is_segment_walkable(
                start_xy=np.array([0.25, 0.5], dtype=np.float32),
                end_xy=np.array([1.75, 0.5], dtype=np.float32),
                margin=0.0,
            )
        )

    def test_segment_walkable_accepts_default_map_doorway_boundary_path(self):
        self.assertTrue(
            DEFAULT_MUSEUM_LAYOUT.is_segment_walkable(
                start_xy=np.array([8.5, 0.0], dtype=np.float32),
                end_xy=np.array([8.5, -10.0], dtype=np.float32),
                margin=HUMAN_WALL_FOOTPRINT_RADIUS,
            )
        )

    def test_farthest_walkable_point_on_segment_returns_exact_boundary_point(self):
        layout = MapLayout(
            name="single_rect",
            default_xml_asset="museum_scene.xml",
            walkable_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.5, 0.5),),
        )
        farthest_xy = layout.find_farthest_walkable_point_on_segment(
            start_xy=np.array([0.2, 0.5], dtype=np.float32),
            end_xy=np.array([1.8, 0.5], dtype=np.float32),
            margin=0.0,
        )
        np.testing.assert_allclose(farthest_xy, np.array([1.0, 0.5], dtype=np.float32), atol=1e-7)

    def test_degenerate_segment_uses_point_containment_semantics(self):
        layout = MapLayout(
            name="degenerate_segment_layout",
            default_xml_asset="museum_scene.xml",
            walkable_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.5, 0.5),),
        )
        inside_xy = np.array([0.5, 0.5], dtype=np.float32)
        outside_xy = np.array([1.5, 0.5], dtype=np.float32)

        self.assertTrue(layout.is_segment_walkable(inside_xy, inside_xy, margin=0.0))
        self.assertFalse(layout.is_segment_walkable(outside_xy, outside_xy, margin=0.0))
        np.testing.assert_allclose(
            layout.find_farthest_walkable_point_on_segment(outside_xy, outside_xy, margin=0.0),
            np.array([1.0, 0.5], dtype=np.float32),
            atol=1e-7,
        )

    def test_human_prefetched_wall_rects_skip_margin_lookup_for_default_margin(self):
        layout = MapLayout(
            name="prefetch_layout",
            default_xml_asset="museum_scene.xml",
            walkable_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.5, 0.5),),
        )
        human = Human("prefetch_probe", "person1", qpos_idx=3, max_speed=1.0, map_layout=layout)

        inside_xy = np.array([0.5, 0.5], dtype=np.float32)
        outside_xy = np.array([1.5, 0.5], dtype=np.float32)
        segment_start = np.array([0.3, 0.5], dtype=np.float32)
        segment_end = np.array([0.9, 0.5], dtype=np.float32)
        default_margin = HUMAN_WALL_FOOTPRINT_RADIUS
        non_default_margin = 0.1

        original_get_walkable_rects = MapLayout.get_walkable_rects
        get_walkable_rect_calls = []

        def tracked_get_walkable_rects(self, margin):
            if self is layout:
                get_walkable_rect_calls.append(float(margin))
            return original_get_walkable_rects(self, margin)

        with patch.object(MapLayout, "get_walkable_rects", new=tracked_get_walkable_rects):
            self.assertTrue(human._is_point_in_walkable(inside_xy, default_margin))
            np.testing.assert_allclose(
                human._project_point_to_walkable(outside_xy, default_margin),
                np.array([0.75, 0.5], dtype=np.float32),
                atol=1e-7,
            )
            self.assertTrue(human._is_segment_walkable(segment_start, np.array([0.7, 0.5], dtype=np.float32), default_margin))
            np.testing.assert_allclose(
                human._find_farthest_walkable_point_on_segment(segment_start, segment_end, default_margin),
                np.array([0.75, 0.5], dtype=np.float32),
                atol=1e-7,
            )
            self.assertEqual(get_walkable_rect_calls, [])

            self.assertTrue(human._is_point_in_walkable(inside_xy, non_default_margin))
            np.testing.assert_allclose(
                human._project_point_to_walkable(outside_xy, non_default_margin),
                np.array([0.9, 0.5], dtype=np.float32),
                atol=1e-7,
            )
            self.assertTrue(human._is_segment_walkable(segment_start, segment_end, non_default_margin))
            np.testing.assert_allclose(
                human._find_farthest_walkable_point_on_segment(segment_start, segment_end, non_default_margin),
                np.array([0.9, 0.5], dtype=np.float32),
                atol=1e-7,
            )
            self.assertEqual(get_walkable_rect_calls, [non_default_margin] * 4)

    def test_assign_target_from_context_matches_expected_follow_and_impatient_geometry(self):
        human = Human("probe", "person1", qpos_idx=3, max_speed=1.0)
        robot_pose = (4.0, 3.0, 0.25)
        human.set_context(
            index=1,
            n_humans=3,
            robot_pose=robot_pose,
            follow_radius=1.5,
            listen_radius=1.2,
            fan_half_angle=np.deg2rad(60.0),
            impatient_front_offset=2.0,
        )

        human._assign_target_from_context(mode=HumanMode.FOLLOWING)
        expected_follow = Human._compute_fan_target(
            robot_pose=robot_pose,
            radius=1.5,
            relative_angle=0.0,
            base_angle_offset=np.pi,
        )
        np.testing.assert_allclose(human.current_waypoint, expected_follow, atol=1e-7)

        human._assign_target_from_context(mode=HumanMode.IMPATIENT)
        expected_impatient = Human._compute_fan_target(
            robot_pose=robot_pose,
            radius=2.0,
            relative_angle=0.0,
            base_angle_offset=0.0,
        )
        np.testing.assert_allclose(human.current_waypoint, expected_impatient, atol=1e-7)

    def test_assign_target_from_context_uses_reachable_point_for_follow_and_impatient_when_current_xy_given(self):
        layout = MapLayout(
            name="narrow_room",
            default_xml_asset="museum_scene.xml",
            walkable_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.5, 0.5),),
        )
        human = Human("probe", "person1", qpos_idx=3, max_speed=1.0, map_layout=layout)
        current_xy = np.array([0.75, 0.5], dtype=np.float32)

        follow_robot_pose = (0.1, 0.5, np.pi)
        human.set_context(
            index=0,
            n_humans=1,
            robot_pose=follow_robot_pose,
            follow_radius=1.0,
            fan_half_angle=np.deg2rad(60.0),
            impatient_front_offset=1.0,
        )
        raw_follow = Human._compute_fan_target(
            robot_pose=follow_robot_pose,
            radius=1.0,
            relative_angle=0.0,
            base_angle_offset=np.pi,
        )
        expected_follow = human._find_farthest_walkable_point_on_segment(
            current_xy,
            raw_follow,
            HUMAN_WALL_FOOTPRINT_RADIUS,
        )
        human._assign_target_from_context(mode=HumanMode.FOLLOWING, current_xy=current_xy)
        np.testing.assert_allclose(human.current_waypoint, expected_follow, atol=1e-7)
        self.assertFalse(np.allclose(human.current_waypoint, raw_follow))

        impatient_robot_pose = (0.9, 0.5, np.pi)
        human.set_context(
            index=0,
            n_humans=1,
            robot_pose=impatient_robot_pose,
            follow_radius=1.0,
            fan_half_angle=np.deg2rad(60.0),
            impatient_front_offset=1.0,
        )
        raw_impatient = Human._compute_fan_target(
            robot_pose=impatient_robot_pose,
            radius=1.0,
            relative_angle=0.0,
            base_angle_offset=0.0,
        )
        expected_impatient = human._find_farthest_walkable_point_on_segment(
            current_xy,
            raw_impatient,
            HUMAN_WALL_FOOTPRINT_RADIUS,
        )
        human._assign_target_from_context(mode=HumanMode.IMPATIENT, current_xy=current_xy)
        np.testing.assert_allclose(human.current_waypoint, expected_impatient, atol=1e-7)
        self.assertFalse(np.allclose(human.current_waypoint, raw_impatient))

    def test_following_target_near_wall_stays_on_reachable_segment(self):
        layout = MapLayout(
            name="right_wall_room",
            default_xml_asset="museum_scene.xml",
            walkable_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.5, 0.5),),
        )
        human = Human("probe", "person1", qpos_idx=3, max_speed=1.0, map_layout=layout)
        current_xy = np.array([0.7, 0.5], dtype=np.float32)
        robot_pose = (0.1, 0.5, np.pi)
        human.set_context(
            index=0,
            n_humans=1,
            robot_pose=robot_pose,
            follow_radius=1.0,
            fan_half_angle=np.deg2rad(60.0),
        )

        raw_follow = Human._compute_fan_target(
            robot_pose=robot_pose,
            radius=1.0,
            relative_angle=0.0,
            base_angle_offset=np.pi,
        )
        expected_follow = human._find_farthest_walkable_point_on_segment(
            current_xy,
            raw_follow,
            HUMAN_WALL_FOOTPRINT_RADIUS,
        )

        human._assign_target_from_context(mode=HumanMode.FOLLOWING, current_xy=current_xy)

        np.testing.assert_allclose(human.current_waypoint, expected_follow, atol=1e-7)
        self.assertTrue(human._is_point_in_walkable(human.current_waypoint, HUMAN_WALL_FOOTPRINT_RADIUS))
        self.assertFalse(np.allclose(human.current_waypoint, raw_follow))

    def test_following_step_uses_pose_xy_for_reachable_target_without_nameerror(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=902)
            human = env.humans[0]
            human.transition_to(HumanMode.FOLLOWING, reason="test_force_following")
            self._set_human_pose(env, human, x=9.7, y=2.0, yaw=0.0)

            robot_pose = (8.7, 2.0, np.pi)
            robot_xy = np.array(robot_pose[:2], dtype=np.float32)
            human.set_context(
                index=0,
                n_humans=1,
                robot_pose=robot_pose,
                follow_radius=1.0,
                fan_half_angle=np.deg2rad(60.0),
                robot_xy=robot_xy,
                robot_yaw=robot_pose[2],
            )
            human.set_following_distracted_window_active(False)

            ctx = {
                "robot_xy": robot_xy,
                "robot_yaw": robot_pose[2],
                "robot_speed": 0.5,
                "repulsion": np.zeros(2, dtype=np.float32),
                "dt": float(env.timestep),
            }

            action = human.step(env.model, env.data, ctx)

            self.assertEqual(action.shape, (3,))
            self.assertTrue(np.all(np.isfinite(action)))
            self.assertTrue(
                human._is_point_in_walkable(
                    human.current_waypoint,
                    HUMAN_WALL_FOOTPRINT_RADIUS,
                )
            )
        finally:
            env.close()

    def test_listening_moves_toward_robot_when_outside_distance_band(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=901)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening")
            self._set_human_pose(env, human, x=3.0, y=1.0, yaw=0.0)
            ctx = {
                "robot_xy": np.array([0.0, 1.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "listen_radius": env.listen_fan_radius,
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }
            action = human.step(env.model, env.data, ctx)
            self.assertLess(float(action[0]), 0.0)
            self.assertAlmostEqual(float(action[1]), 0.0, delta=1e-6)
        finally:
            env.close()

    def test_listening_moves_outward_when_inside_distance_band_core(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=902)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening")
            self._set_human_pose(env, human, x=0.5, y=1.0, yaw=0.0)
            ctx = {
                "robot_xy": np.array([0.0, 1.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "listen_radius": env.listen_fan_radius,
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }
            action = human.step(env.model, env.data, ctx)
            self.assertGreater(float(action[0]), 0.0)
            self.assertAlmostEqual(float(action[1]), 0.0, delta=1e-6)
        finally:
            env.close()

    def test_listening_on_ring_rotates_toward_robot_without_translation(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=903)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening")
            self._set_human_pose(env, human, x=1.0, y=1.0, yaw=np.pi / 2.0)
            ctx = {
                "robot_xy": np.array([0.0, 1.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "listen_radius": env.listen_fan_radius,
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }
            rotate_action = human.step(env.model, env.data, ctx)
            np.testing.assert_allclose(rotate_action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
            self.assertGreater(abs(float(rotate_action[2])), 0.0)

            self._set_human_pose(env, human, x=1.0, y=1.0, yaw=np.pi)
            stop_action = human.step(env.model, env.data, ctx)
            np.testing.assert_allclose(stop_action, np.zeros(3, dtype=np.float32), atol=1e-6)
        finally:
            env.close()

    def test_listening_force_scales_with_radius_error(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=904)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening")
            ctx = {
                "robot_xy": np.array([0.0, 1.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "listen_radius": env.listen_fan_radius,
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }
            self._set_human_pose(env, human, x=1.2, y=1.0, yaw=0.0)
            near_action = human.step(env.model, env.data, ctx)
            self._set_human_pose(env, human, x=3.0, y=1.0, yaw=0.0)
            far_action = human.step(env.model, env.data, ctx)
            self.assertGreater(abs(float(far_action[0])), abs(float(near_action[0])))
        finally:
            env.close()

    def test_listening_sector_target_helper_clamps_to_boundary_angle(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=905)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening")
            theta = env.listen_front_sector_half_angle - np.deg2rad(1.0)
            pos = env.listen_fan_radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            self._set_human_pose(env, human, x=float(pos[0]), y=float(pos[1]), yaw=0.0)
            tangent = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
            ctx = {
                "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": 4.0 * tangent,
                "listen_radius": env.listen_fan_radius,
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }
            target_xy = human._compute_listening_sector_target_point(
                current_xy=pos,
                robot_xy=ctx["robot_xy"],
                robot_yaw=ctx["robot_yaw"],
                listen_radius=ctx["listen_radius"],
                sector_half_angle=ctx["listening_sector_half_angle"],
            )
            self.assertTrue(
                human.is_within_listening_front_sector(
                    point_xy=target_xy,
                    robot_xy=np.array([0.0, 0.0], dtype=np.float32),
                    robot_yaw=0.0,
                    sector_half_angle=env.listen_front_sector_half_angle,
                )
            )
            target_angle = float(np.arctan2(target_xy[1], target_xy[0]))
            self.assertAlmostEqual(
                target_angle,
                float(env.listen_front_sector_half_angle),
                delta=np.deg2rad(1.5),
            )
        finally:
            env.close()

    def test_listening_repulsion_can_pull_next_step_slightly_past_edge_but_target_stays_front(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=906)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening")
            theta = env.listen_front_sector_half_angle - np.deg2rad(1.0)
            pos = env.listen_fan_radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            self._set_human_pose(env, human, x=float(pos[0]), y=float(pos[1]), yaw=0.0)
            tangent = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
            ctx = {
                "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": 4.0 * tangent,
                "listen_radius": env.listen_fan_radius,
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }
            action = human.step(env.model, env.data, ctx)
            next_xy = pos + float(env.timestep) * action[:2]
            target_xy = human._compute_listening_sector_target_point(
                current_xy=pos,
                robot_xy=ctx["robot_xy"],
                robot_yaw=ctx["robot_yaw"],
                listen_radius=ctx["listen_radius"],
                sector_half_angle=ctx["listening_sector_half_angle"],
            )
            self.assertTrue(
                human.is_within_listening_front_sector(
                    point_xy=target_xy,
                    robot_xy=np.array([0.0, 0.0], dtype=np.float32),
                    robot_yaw=0.0,
                    sector_half_angle=env.listen_front_sector_half_angle,
                )
            )
            self.assertGreater(float(np.dot(action[:2], target_xy - pos)), 0.0)
            self.assertGreater(float(np.linalg.norm(next_xy - pos)), 0.0)
        finally:
            env.close()

    def test_listening_backside_pose_generates_front_sector_target_without_pose_projection(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=9061)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening")
            self._set_human_pose(env, human, x=-1.0, y=0.0, yaw=0.0)
            before_xy = np.array(env.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
            ctx = {
                "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "listen_radius": env.listen_fan_radius,
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }
            action = human.step(env.model, env.data, ctx)
            after_xy = np.array(env.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
            target_xy = human._compute_listening_sector_target_point(
                current_xy=before_xy,
                robot_xy=ctx["robot_xy"],
                robot_yaw=ctx["robot_yaw"],
                listen_radius=ctx["listen_radius"],
                sector_half_angle=ctx["listening_sector_half_angle"],
            )
            np.testing.assert_allclose(after_xy, before_xy, atol=1e-6)
            self.assertTrue(
                human.is_within_listening_front_sector(
                    point_xy=target_xy,
                    robot_xy=np.array([0.0, 0.0], dtype=np.float32),
                    robot_yaw=0.0,
                    sector_half_angle=env.listen_front_sector_half_angle,
                )
            )
            self.assertGreater(float(np.dot(action[:2], target_xy - before_xy)), 0.0)
        finally:
            env.close()

    def test_listening_distracted_sampling_clamps_move_direction_to_front_sector(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=9062)
            human = env.humans[0]
            current_xy = np.array([0.8, 0.0], dtype=np.float32)
            with patch("numpy.random.uniform", return_value=40.0), patch("numpy.random.rand", return_value=0.9):
                human._initialize_listening_distracted_target(
                    current_xy=current_xy,
                    current_yaw=0.0,
                    robot_xy=np.array([0.0, 0.0], dtype=np.float32),
                    robot_yaw=0.0,
                    sector_half_angle=env.listen_front_sector_half_angle,
                )
            move_dir = -np.array(
                [np.cos(human.distracted_target_yaw), np.sin(human.distracted_target_yaw)],
                dtype=np.float32,
            )
            move_angle = float(np.arctan2(move_dir[1], move_dir[0]))
            self.assertLessEqual(abs(move_angle), float(env.listen_front_sector_half_angle) + 1e-6)
        finally:
            env.close()

    def test_listening_step_does_not_depend_on_current_waypoint(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=907)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening")
            self._set_human_pose(env, human, x=2.0, y=1.0, yaw=0.0)
            ctx = {
                "robot_xy": np.array([0.0, 1.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "listen_radius": env.listen_fan_radius,
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }
            human.current_waypoint = np.array([9.0, 9.0], dtype=np.float32)
            action_a = human.step(env.model, env.data, ctx)
            human.current_waypoint = np.array([1.0, -9.0], dtype=np.float32)
            action_b = human.step(env.model, env.data, ctx)
            np.testing.assert_allclose(action_a, action_b, atol=1e-6)
        finally:
            env.close()

    def test_hazard_config_helpers_preserve_validation_and_assignment(self):

        human = Human("probe", "person1", qpos_idx=3, max_speed=1.0)
        human.configure_distracted_follow_hazard(0.3, 4.0, 2.0)
        self.assertAlmostEqual(human.following_distracted_lambda_max_per_sec, 0.3)
        self.assertAlmostEqual(human.following_distracted_ramp_start_seconds, 4.0)
        self.assertAlmostEqual(human.following_distracted_rise_seconds, 2.0)

        human.configure_distracted_listening_hazard(0.2, 5.0, 3.0)
        self.assertAlmostEqual(human.listening_distracted_lambda_max_per_sec, 0.2)
        self.assertAlmostEqual(human.listening_distracted_ramp_start_seconds, 5.0)
        self.assertAlmostEqual(human.listening_distracted_rise_seconds, 3.0)

        human.configure_impatient_follow_hazard(0.4, 6.0, 2.5, 0.35)
        self.assertAlmostEqual(human.following_impatient_lambda_max_per_sec, 0.4)
        self.assertAlmostEqual(human.following_impatient_ramp_start_seconds, 6.0)
        self.assertAlmostEqual(human.following_impatient_rise_seconds, 2.5)
        self.assertAlmostEqual(human.following_impatient_robot_speed_threshold, 0.35)

        with self.assertRaises(ValueError):
            human.configure_distracted_follow_hazard(-0.1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            human.configure_distracted_listening_hazard(0.1, -1.0, 1.0)
        with self.assertRaises(ValueError):
            human.configure_impatient_follow_hazard(0.1, 1.0, 0.0, 0.2)
        with self.assertRaises(ValueError):
            human.configure_impatient_follow_hazard(0.1, 1.0, 1.0, 0.0)

    def test_inactive_humans_are_parked_outside_scene_and_excluded_from_info(self):
        env = self._make_env(
            n_humans=10,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=222)
            _, _, _, _, info = env.step(None)
            self.assertEqual(len(env.humans), 10)
            self.assertEqual(info["humans"]["pose_xy"].shape, (10, 2))
            parked_humans = env.all_humans[len(env.humans) :]
            self.assertEqual(len(parked_humans), 5)
            for idx, human in enumerate(parked_humans):
                park_pose = env.data.qpos[human.qpos_idx : human.qpos_idx + 3]
                self.assertGreaterEqual(float(park_pose[0]), INACTIVE_HUMAN_PARK_X + float(idx))
        finally:
            env.close()

    def test_only_one_active_neurodivergent_human_for_any_supported_count(self):
        for n_humans in (5, 10, 15):
            env = self._make_env(
                n_humans=n_humans,
                impatient_prob=0.0,
                overwhelmed_wait_trigger_prob=0.0,
                attack_wait_trigger_prob=0.0,
            )
            try:
                env.reset(seed=300 + n_humans)
                profiles = [human.profile for human in env.humans]
                self.assertEqual(profiles.count(HumanProfile.NEURODIVERGENT), 1)
                self.assertEqual(profiles[0], HumanProfile.NEURODIVERGENT)
                self.assertTrue(all(profile == HumanProfile.NORMAL for profile in profiles[1:]))
            finally:
                env.close()

    def test_distracted_duration_defaults_to_ten_seconds_for_all_humans(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            expected_steps = max(1, int(round(10.0 / float(env.timestep))))
            for human in env.humans:
                self.assertAlmostEqual(human.max_distracted_duration_seconds, 10.0)
                self.assertEqual(human.distracted_duration, expected_steps)
        finally:
            env.close()

    def test_distracted_duration_stays_fixed_across_resets(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            expected_steps = max(1, int(round(10.0 / float(env.timestep))))
            observed = []
            for seed in (101, 102, 103):
                env.reset(seed=seed)
                durations = tuple(int(human.distracted_duration) for human in env.humans)
                observed.append(durations)
                self.assertEqual(durations, (expected_steps,) * len(env.humans))
            self.assertEqual(len(set(observed)), 1)
        finally:
            env.close()

    def test_distracted_samples_one_local_target_and_stops_after_reaching_it(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=777)
            human = env.humans[1]
            self._set_human_pose(env, human, x=5.0, y=5.0, yaw=0.0)
            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            ctx = {
                "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "stand_threshold": env.listen_stand_threshold,
                "dt": float(env.timestep),
            }

            with patch("numpy.random.uniform", side_effect=[60.0, 1.0]), \
                 patch("numpy.random.rand", return_value=0.75):
                action = human.step(env.model, env.data, ctx)

            expected_target = np.array(
                [5.0 + np.cos(np.deg2rad(60.0)), 5.0 + np.sin(np.deg2rad(60.0))],
                dtype=np.float32,
            )
            self.assertIsNotNone(human.distracted_target_xy)
            self.assertAlmostEqual(human.distracted_target_yaw, np.deg2rad(60.0), places=6)
            np.testing.assert_allclose(human.distracted_target_xy, expected_target, atol=1e-5)
            self._assert_in_walkable(human, human.distracted_target_xy)
            self.assertTrue(
                human._is_segment_walkable(
                    start_xy=np.array([5.0, 5.0], dtype=np.float32),
                    end_xy=human.distracted_target_xy,
                    margin=HUMAN_WALL_FOOTPRINT_RADIUS,
                )
            )
            self.assertFalse(human.distracted_stop_reached)
            self.assertLessEqual(float(np.linalg.norm(action[:2])), 0.5 * human.max_speed + 1e-6)

            self._set_human_pose(
                env,
                human,
                x=float(human.distracted_target_xy[0]),
                y=float(human.distracted_target_xy[1]),
                yaw=float(human.distracted_target_yaw),
            )
            stop_action = human.step(env.model, env.data, ctx)
            np.testing.assert_allclose(stop_action, np.zeros(3, dtype=np.float32), atol=1e-6)
            self.assertEqual(human.mode, HumanMode.DISTRACTED)
            self.assertTrue(human.distracted_stop_reached)
        finally:
            env.close()

    def test_distracted_timeout_recovers_exactly_at_configured_max_steps(self):
        env = self._make_env(
            max_distracted_duration_seconds=0.006,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=888)
            human = env.humans[1]
            expected_steps = max(1, int(round(0.006 / float(env.timestep))))
            self.assertEqual(human.distracted_duration, expected_steps)

            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self._set_human_pose(env, human, x=5.0, y=5.0, yaw=0.0)
            human.distracted_target_xy = np.array([5.0, 5.0], dtype=np.float32)
            human.current_waypoint = np.array([5.0, 5.0], dtype=np.float32)
            human.distracted_target_yaw = 0.0
            human.distracted_stop_reached = True
            ctx = {
                "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "stand_threshold": env.listen_stand_threshold,
                "dt": float(env.timestep),
            }

            for _ in range(max(0, expected_steps - 1)):
                human.step(env.model, env.data, ctx)
                self.assertEqual(human.mode, HumanMode.DISTRACTED)
            human.step(env.model, env.data, ctx)

            self.assertEqual(human.mode, HumanMode.FOLLOWING)
        finally:
            env.close()

    def test_stopped_distracted_human_remains_callback_eligible(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=778)
            target_idx = 1
            human = env.humans[target_idx]
            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            human.distracted_target_xy = np.array([2.6, 0.0], dtype=np.float32)
            human.current_waypoint = np.array([2.6, 0.0], dtype=np.float32)
            human.distracted_target_yaw = 0.0
            human.distracted_stop_reached = True
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.6, 0.0], dtype=np.float32)

            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))

            self.assertIsNotNone(request)
            self.assertEqual(request["target_idx"], target_idx)
        finally:
            env.close()

    def test_callback_response_defaults_are_profile_specific(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            self.assertAlmostEqual(env.callback_rejoin_prob_normal, 0.80)
            self.assertAlmostEqual(env.callback_ignore_prob_normal, 0.20)
            self.assertAlmostEqual(env.callback_rejoin_prob_nd, 0.40)
            self.assertAlmostEqual(env.callback_ignore_prob_nd, 0.60)

            env.np_random = _FixedRandom(0.79)
            self.assertEqual(env._sample_callback_response(HumanProfile.NORMAL), "rejoin")
            env.np_random = _FixedRandom(0.80)
            self.assertEqual(env._sample_callback_response(HumanProfile.NORMAL), "ignore")

            env.np_random = _FixedRandom(0.39)
            self.assertEqual(env._sample_callback_response(HumanProfile.NEURODIVERGENT), "rejoin")
            env.np_random = _FixedRandom(0.40)
            self.assertEqual(env._sample_callback_response(HumanProfile.NEURODIVERGENT), "ignore")
        finally:
            env.close()

    def test_callback_response_can_be_overridden_via_constructor(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
            callback_rejoin_prob_normal=0.10,
            callback_ignore_prob_normal=0.90,
            callback_rejoin_prob_nd=0.80,
            callback_ignore_prob_nd=0.20,
        )
        try:
            self.assertAlmostEqual(env.callback_rejoin_prob_normal, 0.10)
            self.assertAlmostEqual(env.callback_ignore_prob_normal, 0.90)
            self.assertAlmostEqual(env.callback_rejoin_prob_nd, 0.80)
            self.assertAlmostEqual(env.callback_ignore_prob_nd, 0.20)

            env.np_random = _FixedRandom(0.15)
            self.assertEqual(env._sample_callback_response(HumanProfile.NORMAL), "ignore")
            env.np_random = _FixedRandom(0.05)
            self.assertEqual(env._sample_callback_response(HumanProfile.NEURODIVERGENT), "rejoin")
        finally:
            env.close()

    def test_callback_request_not_triggered_when_distance_is_at_or_below_threshold(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=120)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[target_idx].distracted_timer = 0
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.0, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNone(request)
        finally:
            env.close()

    def test_callback_request_triggers_when_distance_exceeds_threshold_even_at_timer_zero(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=121)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[target_idx].distracted_timer = 0
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.01, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNotNone(request)
            self.assertEqual(request["target_idx"], target_idx)
            expected_cue_steps = max(1, int(round(3.0 / float(env.timestep))))
            self.assertEqual(request["cue_steps"], expected_cue_steps)
        finally:
            env.close()

    def test_callback_request_targets_farthest_distracted_under_distance_gate(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=122)
            idx_near = 1
            idx_far = 2
            env.humans[idx_near].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[idx_far].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[idx_near].distracted_timer = 50
            env.humans[idx_far].distracted_timer = 1
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[idx_near] = np.array([2.1, 0.0], dtype=np.float32)
            human_xy[idx_far] = np.array([2.8, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNotNone(request)
            self.assertEqual(request["target_idx"], idx_far)
        finally:
            env.close()

    def test_callback_request_distance_threshold_can_be_overridden_and_is_strict(self):
        env = self._make_env(
            callback_trigger_distance_meters=1.5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=123)
            self.assertAlmostEqual(env.callback_trigger_distance_meters, 1.5)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)

            human_xy[target_idx] = np.array([1.5, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                boundary_request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNone(boundary_request)

            human_xy[target_idx] = np.array([1.5001, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                above_request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNotNone(above_request)
            self.assertEqual(above_request["target_idx"], target_idx)
        finally:
            env.close()

    def test_callback_trigger_distance_constructor_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            self._make_env(
                callback_trigger_distance_meters=0.0,
                impatient_prob=0.0,
                overwhelmed_wait_trigger_prob=0.0,
                attack_wait_trigger_prob=0.0,
            )

    def test_callback_request_is_blocked_when_robot_not_in_move_stage(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=124)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([3.0, 0.0], dtype=np.float32)

            env.robot.listen_mode = True
            request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNone(request)
        finally:
            env.close()

    def test_callback_trigger_distance_exposed_in_info_status(self):
        env = self._make_env(
            callback_trigger_distance_meters=1.75,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=125)
            _, _, _, _, info = env.step(None)
            self.assertAlmostEqual(info["status"]["callback_trigger_distance_meters"], 1.75)
            self.assertEqual(info["status"]["perceived_distracted_indices"], [])
            self.assertFalse(info["status"]["callback_visual_active"])
        finally:
            env.close()

    def test_active_branch_reuses_analysis_for_pre_and_post_step_snapshots(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=140)
            with patch.object(env, "_analyze_human_state", wraps=env._analyze_human_state) as mock_analyze:
                with patch.object(env, "_build_callback_request", wraps=env._build_callback_request) as mock_callback:
                    with patch.object(
                        env,
                        "_update_robot_emotion_and_visual",
                        wraps=env._update_robot_emotion_and_visual,
                    ) as mock_update:
                        env.step(None)

            self.assertEqual(mock_analyze.call_count, 2)
            self.assertIsNotNone(mock_callback.call_args.kwargs["human_analysis"])
            self.assertIsNotNone(mock_update.call_args.kwargs["human_analysis"])
        finally:
            env.close()

    def test_wait_branch_reuses_single_post_step_analysis(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=141)
            env.listen_intro_delay_steps = 1
            env.robot.current_waypoint_idx = 0
            env.robot.listen_mode = False
            env.robot.listen_done = False
            env.robot.turn_done = True
            self._set_robot_pose(env, x=1.0, y=5.0, yaw=0.0)
            self._set_human_pose(env, env.humans[0], x=0.0, y=5.0, yaw=0.0)

            env.step(None)
            self.assertTrue(env.listen_wait_active)

            with patch.object(env, "_analyze_human_state", wraps=env._analyze_human_state) as mock_analyze:
                with patch.object(
                    env,
                    "_update_robot_emotion_and_visual",
                    wraps=env._update_robot_emotion_and_visual,
                ) as mock_update:
                    env.step(None)

            self.assertEqual(mock_analyze.call_count, 1)
            self.assertIsNotNone(mock_update.call_args.kwargs["human_analysis"])
        finally:
            env.close()

    def test_robot_base_color_writes_only_on_emotion_transitions(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=142)

            self.assertFalse(env._apply_robot_base_color_from_robot_emotion())

            env.robot.emotion = RobotEmotion.SAD
            self.assertTrue(env._apply_robot_base_color_from_robot_emotion())
            np.testing.assert_allclose(
                env.model.geom_rgba[env.robot_base_geom_id],
                env._robot_base_rgba_for_emotion(),
                atol=1e-7,
            )
            self.assertFalse(env._apply_robot_base_color_from_robot_emotion())

            env.robot.emotion = RobotEmotion.HAPPY
            self.assertTrue(env._apply_robot_base_color_from_robot_emotion())
            np.testing.assert_allclose(
                env.model.geom_rgba[env.robot_base_geom_id],
                env._robot_base_rgba_for_emotion(),
                atol=1e-7,
            )
            self.assertFalse(env._apply_robot_base_color_from_robot_emotion())
        finally:
            env.close()

    def test_robot_speaking_halo_writes_only_when_speaker_state_toggles(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=143)

            self.assertFalse(env._apply_robot_speaking_halo_visual())

            env.robot.set_speaker_active(True)
            self.assertTrue(env._apply_robot_speaking_halo_visual())
            self.assertGreater(float(env.model.geom_rgba[env.robot_speaking_halo_geom_id][3]), 0.0)
            self.assertFalse(env._apply_robot_speaking_halo_visual())

            env.robot.set_speaker_active(False)
            self.assertTrue(env._apply_robot_speaking_halo_visual())
            self.assertEqual(float(env.model.geom_rgba[env.robot_speaking_halo_geom_id][3]), 0.0)
            self.assertFalse(env._apply_robot_speaking_halo_visual())
        finally:
            env.close()

    def test_robot_text_label_priority_is_preserved_under_dirty_sync(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=144)
            self.assertFalse(env._sync_robot_text_label_visibility())

            env.robot.set_speaker_active(True)
            self.assertTrue(env._sync_robot_text_label_visibility())
            self.assertEqual(env._get_robot_text_label(), "explanation")
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_EXPLANATION_LABEL_GROUP], 1)
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP], 0)
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_NEED_SPACE_LABEL_GROUP], 0)
            self.assertFalse(env._sync_robot_text_label_visibility())

            self._set_callback_state(env, target_idx=1, phase=RobotCallbackPhase.CUE, cue_elapsed_steps=1)
            self.assertTrue(env._sync_robot_text_label_visibility())
            self.assertEqual(env._get_robot_text_label(), "Please_follow_me")
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_EXPLANATION_LABEL_GROUP], 0)
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP], 1)
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_NEED_SPACE_LABEL_GROUP], 0)

            env.fear_active = True
            self.assertTrue(env._sync_robot_text_label_visibility())
            self.assertEqual(env._get_robot_text_label(), "I_need_more_space")
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_EXPLANATION_LABEL_GROUP], 0)
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP], 0)
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_NEED_SPACE_LABEL_GROUP], 1)
        finally:
            env.close()

    def test_analyze_human_state_handles_empty_pose_array(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=145)
            analysis = env._analyze_human_state(
                robot_xy=np.array([0.0, 0.0], dtype=np.float32),
                human_xy=np.zeros((0, 2), dtype=np.float32),
            )
            self.assertEqual(analysis["perceived_distracted_indices"], [])
            self.assertIsNone(analysis["callback_target_idx"])
            self.assertIsNone(analysis["nearest_attack_threat"])
            self.assertEqual(analysis["emotion_modes"], [])
        finally:
            env.close()

    def test_analyze_human_state_selects_nearest_attack_and_farthest_eligible_callback_target(self):
        env = self._make_env(
            callback_trigger_distance_meters=2.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=146)
            env.humans[0].transition_to(HumanMode.ATTACK, reason="test_force_attack")
            env.humans[1].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[2].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[3].transition_to(HumanMode.FOLLOWING, reason="test_force_following")
            env.callback_triggered_for_current_distracted[1] = True

            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[0] = np.array([0.5, 0.0], dtype=np.float32)
            human_xy[1] = np.array([2.5, 0.0], dtype=np.float32)
            human_xy[2] = np.array([3.5, 0.0], dtype=np.float32)
            human_xy[3] = np.array([1.0, 1.0], dtype=np.float32)

            analysis = env._analyze_human_state(
                robot_xy=np.array([0.0, 0.0], dtype=np.float32),
                human_xy=human_xy,
            )

            self.assertEqual(analysis["perceived_distracted_indices"], [1, 2])
            self.assertEqual(analysis["callback_target_idx"], 2)
            self.assertIsNotNone(analysis["nearest_attack_threat"])
            self.assertEqual(analysis["nearest_attack_threat"]["idx"], 0)
            self.assertAlmostEqual(analysis["nearest_attack_threat"]["dist"], 0.5)
            self.assertEqual(
                analysis["emotion_modes"],
                [HumanMode.ATTACK, HumanMode.FOLLOWING] + [human.mode for human in env.humans[4:]],
            )
        finally:
            env.close()

    def test_analyze_human_state_threshold_and_rearm_flags_control_callback_selection(self):
        env = self._make_env(
            callback_trigger_distance_meters=1.5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=147)
            env.humans[0].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[1].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.callback_triggered_for_current_distracted[1] = True

            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[0] = np.array([1.5000, 0.0], dtype=np.float32)
            human_xy[1] = np.array([1.5001, 0.0], dtype=np.float32)

            analysis = env._analyze_human_state(
                robot_xy=np.array([0.0, 0.0], dtype=np.float32),
                human_xy=human_xy,
            )

            self.assertEqual(analysis["perceived_distracted_indices"], [1])
            self.assertIsNone(analysis["callback_target_idx"])
        finally:
            env.close()

    def test_social_repulsion_matches_dense_reference_for_isolated_and_far_humans(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=148)
            human_xy = np.array(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
                dtype=np.float32,
            )
            repulsion = env._compute_social_repulsion(human_xy)
            expected = self._dense_social_repulsion_reference(
                human_xy,
                social_distance=env.social_distance,
                repulsion_gain=env.repulsion_gain,
            )
            np.testing.assert_allclose(repulsion, expected, atol=1e-7)
        finally:
            env.close()

    def test_social_repulsion_matches_dense_reference_for_close_pair_and_boundary_neighbors(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=149)
            human_xy = np.array(
                [
                    [0.79, 0.79],
                    [1.01, 0.79],
                    [0.79, 1.01],
                    [1.01, 1.01],
                ],
                dtype=np.float32,
            )
            repulsion = env._compute_social_repulsion(human_xy)
            expected = self._dense_social_repulsion_reference(
                human_xy,
                social_distance=env.social_distance,
                repulsion_gain=env.repulsion_gain,
            )
            np.testing.assert_allclose(repulsion, expected, atol=1e-7)
        finally:
            env.close()

    def test_social_repulsion_matches_dense_reference_for_mixed_local_cluster(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=150)
            human_xy = np.array(
                [
                    [0.0, 0.0],
                    [0.3, 0.1],
                    [0.55, 0.2],
                    [1.5, 1.5],
                    [1.9, 1.5],
                    [3.5, 3.5],
                ],
                dtype=np.float32,
            )
            repulsion = env._compute_social_repulsion(human_xy)
            expected = self._dense_social_repulsion_reference(
                human_xy,
                social_distance=env.social_distance,
                repulsion_gain=env.repulsion_gain,
            )
            np.testing.assert_allclose(repulsion, expected, atol=1e-7)
        finally:
            env.close()

    def test_nearest_human_distance_helper_matches_dense_reference(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=151)
            human_xy = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.5, 0.5],
                    [4.0, 0.0],
                ],
                dtype=np.float32,
            )
            nearest = env._compute_nearest_human_distances(human_xy)
            expected = self._dense_nearest_human_distance_reference(human_xy)
            np.testing.assert_allclose(nearest, expected, atol=1e-7)
        finally:
            env.close()

    def test_nearest_human_distance_helper_handles_empty_and_singleton_inputs(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=152)
            empty = env._compute_nearest_human_distances(np.zeros((0, 2), dtype=np.float32))
            singleton = env._compute_nearest_human_distances(np.array([[2.0, 3.0]], dtype=np.float32))
            self.assertEqual(empty.shape, (0,))
            self.assertEqual(singleton.shape, (1,))
            self.assertTrue(np.isnan(singleton[0]))
        finally:
            env.close()

    def test_local_crowding_count_1m_helper_matches_dense_reference(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=161)
            human_xy = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.5, 0.5],
                    [4.0, 0.0],
                ],
                dtype=np.float32,
            )
            local_crowding_count_1m = env._compute_local_crowding_count_1m(human_xy)
            expected = self._dense_local_crowding_count_1m_reference(human_xy)
            np.testing.assert_array_equal(local_crowding_count_1m, expected)
        finally:
            env.close()

    def test_local_crowding_count_1m_helper_handles_empty_and_singleton_inputs(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=162)
            empty = env._compute_local_crowding_count_1m(np.zeros((0, 2), dtype=np.float32))
            singleton = env._compute_local_crowding_count_1m(
                np.array([[2.0, 3.0]], dtype=np.float32)
            )
            self.assertEqual(empty.shape, (0,))
            self.assertEqual(singleton.shape, (1,))
            self.assertTrue(np.issubdtype(empty.dtype, np.integer))
            self.assertTrue(np.issubdtype(singleton.dtype, np.integer))
            self.assertEqual(int(singleton[0]), 0)
        finally:
            env.close()

    def test_local_crowding_count_1m_helper_uses_strict_radius_boundary(self):
        env = self._make_env(
            n_humans=3,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=163)
            human_xy = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, 0.0],
                ],
                dtype=np.float32,
            )

            local_crowding_count_1m = env._compute_local_crowding_count_1m(human_xy)

            np.testing.assert_array_equal(
                local_crowding_count_1m,
                np.array([1, 1, 2], dtype=np.int32),
            )
        finally:
            env.close()

    def test_hh_distance_metrics_rolling_mean_supports_warmup_and_eviction(self):
        env = self._make_env(
            n_humans=2,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=153)
            env._reset_hh_distance_metrics_state(window_steps=3)

            avg_1 = env._update_hh_distance_metrics(np.array([1.0, 4.0], dtype=np.float32))
            avg_2 = env._update_hh_distance_metrics(np.array([2.0, 5.0], dtype=np.float32))
            avg_3 = env._update_hh_distance_metrics(np.array([3.0, 6.0], dtype=np.float32))
            avg_4 = env._update_hh_distance_metrics(np.array([10.0, 20.0], dtype=np.float32))

            np.testing.assert_allclose(avg_1, np.array([1.0, 4.0], dtype=np.float32), atol=1e-7)
            np.testing.assert_allclose(avg_2, np.array([1.5, 4.5], dtype=np.float32), atol=1e-7)
            np.testing.assert_allclose(avg_3, np.array([2.0, 5.0], dtype=np.float32), atol=1e-7)
            np.testing.assert_allclose(avg_4, np.array([5.0, 31.0 / 3.0], dtype=np.float32), atol=1e-7)
        finally:
            env.close()

    def test_human_robot_distance_helper_matches_dense_reference(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=156)
            human_xy = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.5, 0.5],
                    [4.0, 0.0],
                ],
                dtype=np.float32,
            )
            robot_xy = np.array([0.5, 1.0], dtype=np.float32)
            dist = env._compute_human_robot_distances(human_xy, robot_xy)
            expected = self._dense_human_robot_distance_reference(human_xy, robot_xy)
            np.testing.assert_allclose(dist, expected, atol=1e-7)
        finally:
            env.close()

    def test_human_robot_distance_helper_handles_empty_input(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=157)
            empty = env._compute_human_robot_distances(
                np.zeros((0, 2), dtype=np.float32),
                np.array([2.0, 3.0], dtype=np.float32),
            )
            self.assertEqual(empty.shape, (0,))
        finally:
            env.close()

    def test_hr_distance_metrics_rolling_mean_supports_warmup_and_eviction(self):
        env = self._make_env(
            n_humans=2,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=158)
            env._reset_hr_distance_metrics_state(window_steps=3)

            avg_1 = env._update_hr_distance_metrics(np.array([1.0, 4.0], dtype=np.float32))
            avg_2 = env._update_hr_distance_metrics(np.array([2.0, 5.0], dtype=np.float32))
            avg_3 = env._update_hr_distance_metrics(np.array([3.0, 6.0], dtype=np.float32))
            avg_4 = env._update_hr_distance_metrics(np.array([10.0, 20.0], dtype=np.float32))

            np.testing.assert_allclose(avg_1, np.array([1.0, 4.0], dtype=np.float32), atol=1e-7)
            np.testing.assert_allclose(avg_2, np.array([1.5, 4.5], dtype=np.float32), atol=1e-7)
            np.testing.assert_allclose(avg_3, np.array([2.0, 5.0], dtype=np.float32), atol=1e-7)
            np.testing.assert_allclose(avg_4, np.array([5.0, 31.0 / 3.0], dtype=np.float32), atol=1e-7)
        finally:
            env.close()

    def test_step_info_metrics_match_pose_xy_reference_in_active_branch(self):
        env = self._make_env(
            n_humans=5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=154)
            _, _, _, _, info = env.step(None)

            self.assertIn("metrics", info)
            self.assertIn("humans", info["metrics"])
            nearest = info["metrics"]["humans"]["nearest_human_distance"]
            mean_1s = info["metrics"]["humans"]["nearest_human_distance_mean_1s"]
            pose_xy = np.array(info["humans"]["pose_xy"], dtype=np.float32)
            expected = self._dense_nearest_human_distance_reference(pose_xy)

            self.assertEqual(nearest.shape, (len(env.humans),))
            self.assertEqual(mean_1s.shape, (len(env.humans),))
            self.assertEqual(info["metrics"]["humans"]["window_seconds"], 1.0)
            self.assertEqual(info["metrics"]["humans"]["window_steps"], env.hh_distance_window_steps)
            np.testing.assert_allclose(nearest, expected, atol=1e-6)
            np.testing.assert_allclose(mean_1s, expected, atol=1e-6)
        finally:
            env.close()

    def test_step_info_metrics_match_pose_xy_reference_in_waiting_branch(self):
        env = self._make_env(
            n_humans=5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=155)
            env.robot.listen_mode = True
            env.listen_wait_active = True
            env.listen_wait_counter = 0
            env.listen_wait_is_final = False
            env.robot.mode = "stop"
            self._set_robot_pose(env, x=2.0, y=2.0, yaw=0.0)

            _, _, _, _, info = env.step(None)

            nearest = info["metrics"]["humans"]["nearest_human_distance"]
            mean_1s = info["metrics"]["humans"]["nearest_human_distance_mean_1s"]
            pose_xy = np.array(info["humans"]["pose_xy"], dtype=np.float32)
            expected = self._dense_nearest_human_distance_reference(pose_xy)

            self.assertEqual(nearest.shape, (len(env.humans),))
            self.assertEqual(mean_1s.shape, (len(env.humans),))
            np.testing.assert_allclose(nearest, expected, atol=1e-6)
            np.testing.assert_allclose(mean_1s, expected, atol=1e-6)
        finally:
            env.close()

    def test_step_info_local_crowding_metrics_match_pose_xy_reference_in_active_branch(self):
        env = self._make_env(
            n_humans=5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=164)
            _, _, _, _, info = env.step(None)

            local_crowding_count_1m = info["metrics"]["humans"]["local_crowding_count_1m"]
            pose_xy = np.array(info["humans"]["pose_xy"], dtype=np.float32)
            expected = self._dense_local_crowding_count_1m_reference(pose_xy)

            self.assertEqual(local_crowding_count_1m.shape, (len(env.humans),))
            self.assertTrue(np.issubdtype(local_crowding_count_1m.dtype, np.integer))
            np.testing.assert_array_equal(local_crowding_count_1m, expected)
        finally:
            env.close()

    def test_step_info_local_crowding_metrics_match_pose_xy_reference_in_waiting_branch(self):
        env = self._make_env(
            n_humans=5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=165)
            env.robot.listen_mode = True
            env.listen_wait_active = True
            env.listen_wait_counter = 0
            env.listen_wait_is_final = False
            env.robot.mode = "stop"
            self._set_robot_pose(env, x=2.0, y=2.0, yaw=0.0)

            _, _, _, _, info = env.step(None)

            local_crowding_count_1m = info["metrics"]["humans"]["local_crowding_count_1m"]
            pose_xy = np.array(info["humans"]["pose_xy"], dtype=np.float32)
            expected = self._dense_local_crowding_count_1m_reference(pose_xy)

            self.assertEqual(local_crowding_count_1m.shape, (len(env.humans),))
            self.assertTrue(np.issubdtype(local_crowding_count_1m.dtype, np.integer))
            np.testing.assert_array_equal(local_crowding_count_1m, expected)
        finally:
            env.close()

    def test_step_info_human_robot_metrics_match_pose_xy_reference_in_active_branch(self):
        env = self._make_env(
            n_humans=5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=159)
            _, _, _, _, info = env.step(None)

            dist = info["metrics"]["humans"]["human_robot_distance"]
            mean_1s = info["metrics"]["humans"]["human_robot_distance_mean_1s"]
            pose_xy = np.array(info["humans"]["pose_xy"], dtype=np.float32)
            robot_xy = np.array(info["robot"]["pose_xy"], dtype=np.float32)
            expected = self._dense_human_robot_distance_reference(pose_xy, robot_xy)

            self.assertEqual(dist.shape, (len(env.humans),))
            self.assertEqual(mean_1s.shape, (len(env.humans),))
            np.testing.assert_allclose(dist, expected, atol=1e-6)
            np.testing.assert_allclose(mean_1s, expected, atol=1e-6)
        finally:
            env.close()

    def test_step_info_human_robot_metrics_match_pose_xy_reference_in_waiting_branch(self):
        env = self._make_env(
            n_humans=5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=160)
            env.robot.listen_mode = True
            env.listen_wait_active = True
            env.listen_wait_counter = 0
            env.listen_wait_is_final = False
            env.robot.mode = "stop"
            self._set_robot_pose(env, x=2.0, y=2.0, yaw=0.0)

            _, _, _, _, info = env.step(None)

            dist = info["metrics"]["humans"]["human_robot_distance"]
            mean_1s = info["metrics"]["humans"]["human_robot_distance_mean_1s"]
            pose_xy = np.array(info["humans"]["pose_xy"], dtype=np.float32)
            robot_xy = np.array(info["robot"]["pose_xy"], dtype=np.float32)
            expected = self._dense_human_robot_distance_reference(pose_xy, robot_xy)

            self.assertEqual(dist.shape, (len(env.humans),))
            self.assertEqual(mean_1s.shape, (len(env.humans),))
            np.testing.assert_allclose(dist, expected, atol=1e-6)
            np.testing.assert_allclose(mean_1s, expected, atol=1e-6)
        finally:
            env.close()

    def test_robot_does_not_perceive_near_distracted_for_label_or_sad(self):
        env = self._make_env(
            callback_trigger_distance_meters=2.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=126)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.0, 0.0], dtype=np.float32)

            events = env._default_events()
            env._update_robot_emotion_and_visual(events=events, robot_xy=robot_xy, human_xy=human_xy)
            env._sync_robot_text_label_visibility()

            self.assertNotIn(target_idx, env.perceived_distracted_indices)
            self.assertNotEqual(env._get_robot_text_label(), "Please_follow_me")
            self.assertEqual(env.robot.emotion, RobotEmotion.NATURAL)
        finally:
            env.close()

    def test_robot_perceives_far_distracted_but_visuals_stay_off_when_callback_not_active(self):
        env = self._make_env(
            callback_trigger_distance_meters=2.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=127)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.01, 0.0], dtype=np.float32)

            events = env._default_events()
            env._update_robot_emotion_and_visual(events=events, robot_xy=robot_xy, human_xy=human_xy)
            env._sync_robot_text_label_visibility()

            self.assertIn(target_idx, env.perceived_distracted_indices)
            self.assertNotEqual(env._get_robot_text_label(), "Please_follow_me")
            self.assertEqual(env.robot.emotion, RobotEmotion.NATURAL)
        finally:
            env.close()

    def test_callback_turn_phase_does_not_show_follow_me_or_distracted_sad(self):
        env = self._make_env(
            callback_trigger_distance_meters=2.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=131)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self._set_callback_state(
                env,
                target_idx,
                phase=RobotCallbackPhase.TURN,
                cue_elapsed_steps=0,
            )

            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.4, 0.0], dtype=np.float32)

            events = env._default_events()
            env._update_robot_emotion_and_visual(events=events, robot_xy=robot_xy, human_xy=human_xy)
            env._sync_robot_text_label_visibility()

            self.assertIn(target_idx, env.perceived_distracted_indices)
            self.assertNotEqual(env._get_robot_text_label(), "Please_follow_me")
            self.assertEqual(env.robot.emotion, RobotEmotion.NATURAL)
        finally:
            env.close()

    def test_callback_visual_phase_shows_follow_me_and_sets_sad(self):
        env = self._make_env(
            callback_trigger_distance_meters=2.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=132)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self._set_callback_state(
                env,
                target_idx,
                phase=RobotCallbackPhase.CUE,
                cue_elapsed_steps=20,
            )

            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.4, 0.0], dtype=np.float32)

            events = env._default_events()
            env._update_robot_emotion_and_visual(events=events, robot_xy=robot_xy, human_xy=human_xy)
            env._sync_robot_text_label_visibility()

            self.assertIn(target_idx, env.perceived_distracted_indices)
            self.assertEqual(env._get_robot_text_label(), "Please_follow_me")
            self.assertEqual(env.robot.emotion, RobotEmotion.SAD)
        finally:
            env.close()

    def test_callback_end_hides_follow_me_and_reverts_distracted_sad_even_if_person_still_far(self):
        env = self._make_env(
            callback_trigger_distance_meters=2.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=133)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.robot.callback_active = False
            env.robot.callback_phase = None
            env.robot.callback_cue_total_steps = 0
            env.robot.callback_cue_elapsed_steps = 0

            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.4, 0.0], dtype=np.float32)

            events = env._default_events()
            env._update_robot_emotion_and_visual(events=events, robot_xy=robot_xy, human_xy=human_xy)
            env._sync_robot_text_label_visibility()

            self.assertIn(target_idx, env.perceived_distracted_indices)
            self.assertNotEqual(env._get_robot_text_label(), "Please_follow_me")
            self.assertEqual(env.robot.emotion, RobotEmotion.NATURAL)
        finally:
            env.close()

    def test_overwhelmed_still_triggers_sad_without_distance_gate(self):
        env = self._make_env(
            callback_trigger_distance_meters=2.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=128)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.OVERWHELMED, reason="test_force_overwhelmed")
            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([0.5, 0.0], dtype=np.float32)

            events = env._default_events()
            env._update_robot_emotion_and_visual(events=events, robot_xy=robot_xy, human_xy=human_xy)
            env._sync_robot_text_label_visibility()

            self.assertEqual(env.robot.emotion, RobotEmotion.SAD)
            self.assertNotEqual(env._get_robot_text_label(), "Please_follow_me")
        finally:
            env.close()

    def test_callback_request_ignores_near_distracted_even_with_larger_timer(self):
        env = self._make_env(
            callback_trigger_distance_meters=2.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=129)
            near_idx = 1
            far_idx = 2
            env.humans[near_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[far_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[near_idx].distracted_timer = 20
            env.humans[far_idx].distracted_timer = 5

            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[near_idx] = np.array([1.0, 0.0], dtype=np.float32)
            human_xy[far_idx] = np.array([2.5, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))

            self.assertIsNotNone(request)
            self.assertEqual(request["target_idx"], far_idx)
        finally:
            env.close()

    def test_perceived_distance_threshold_override_does_not_force_follow_me_or_sad_without_callback_visual(self):
        env = self._make_env(
            callback_trigger_distance_meters=1.5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=130)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)

            human_xy[target_idx] = np.array([1.5, 0.0], dtype=np.float32)
            events = env._default_events()
            env._update_robot_emotion_and_visual(events=events, robot_xy=robot_xy, human_xy=human_xy)
            env._sync_robot_text_label_visibility()
            self.assertNotIn(target_idx, env.perceived_distracted_indices)
            self.assertNotEqual(env._get_robot_text_label(), "Please_follow_me")
            self.assertEqual(env.robot.emotion, RobotEmotion.NATURAL)

            human_xy[target_idx] = np.array([1.5001, 0.0], dtype=np.float32)
            events = env._default_events()
            env._update_robot_emotion_and_visual(events=events, robot_xy=robot_xy, human_xy=human_xy)
            env._sync_robot_text_label_visibility()
            self.assertIn(target_idx, env.perceived_distracted_indices)
            self.assertNotEqual(env._get_robot_text_label(), "Please_follow_me")
            self.assertEqual(env.robot.emotion, RobotEmotion.NATURAL)
        finally:
            env.close()

    def test_callback_response_sampling_uses_target_human_profile_at_two_seconds(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            for idx in (0, 1):
                with self.subTest(target_index=idx):
                    env.reset(seed=200 + idx)
                    env.humans[idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
                    self._set_callback_state(
                        env,
                        idx,
                        phase=RobotCallbackPhase.CUE,
                        cue_elapsed_steps=env._get_callback_response_sample_steps(),
                    )
                    events = env._default_events()
                    with patch.object(env, "_sample_callback_response", return_value="ignore") as callback_sampler:
                        env._maybe_sample_active_callback_response(events)
                        callback_sampler.assert_called_once_with(profile=env.humans[idx].profile)
        finally:
            env.close()

    def test_callback_rejoin_applies_immediately_without_stay_steps(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=300)
            target_idx = 1
            target_human = env.humans[target_idx]
            target_human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self._set_callback_state(
                env,
                target_idx,
                phase=RobotCallbackPhase.CUE,
                cue_elapsed_steps=env._get_callback_response_sample_steps(),
            )
            events = env._default_events()
            with patch.object(env, "_sample_callback_response", return_value="rejoin"), \
                 patch.object(target_human, "apply_callback_response", return_value=True) as apply_response:
                env._maybe_sample_active_callback_response(events)
                apply_response.assert_called_once_with(response="rejoin", stay_steps=0)
        finally:
            env.close()

    def test_callback_response_sampling_happens_only_once(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=301)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self._set_callback_state(
                env,
                target_idx,
                phase=RobotCallbackPhase.CUE,
                cue_elapsed_steps=env._get_callback_response_sample_steps(),
            )
            events = env._default_events()
            with patch.object(env, "_sample_callback_response", return_value="ignore") as callback_sampler:
                env._maybe_sample_active_callback_response(events)
                env._maybe_sample_active_callback_response(events)
                callback_sampler.assert_called_once()
            self.assertTrue(env.robot.callback_response_sampled)
        finally:
            env.close()

    def test_callback_ignore_response_keeps_standard_distracted_motion(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=320)
            human = env.humans[1]
            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            human.apply_callback_response("ignore")
            self._set_human_pose(env, human, x=5.0, y=5.0, yaw=0.0)
            human.distracted_target_xy = np.array([5.5, 5.0], dtype=np.float32)
            human.current_waypoint = np.array([5.5, 5.0], dtype=np.float32)
            human.distracted_target_yaw = 0.0
            human.distracted_stop_reached = False
            ctx = {
                "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "stand_threshold": env.listen_stand_threshold,
                "dt": float(env.timestep),
            }
            action = human.step(env.model, env.data, ctx)
            self.assertEqual(human.mode, HumanMode.DISTRACTED)
            self.assertGreater(float(np.linalg.norm(action[:2])), 0.0)

            human.distracted_stop_reached = True
            stop_action = human.step(env.model, env.data, ctx)
            np.testing.assert_allclose(stop_action, np.zeros(3, dtype=np.float32), atol=1e-6)
        finally:
            env.close()

    def test_callback_first_failed_attempt_starts_second_attempt(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=360)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self._set_callback_state(
                env,
                target_idx,
                phase=RobotCallbackPhase.CUE,
                attempt_index=1,
                cue_elapsed_steps=env._get_callback_cue_steps(),
                cue_completed_this_step=True,
            )
            events = env._default_events()
            env._resolve_completed_callback_cue(events)

            self.assertTrue(events["callback_first_attempt_failed"])
            self.assertTrue(events["callback_attempt_2_started"])
            self.assertFalse(events["callback_completed"])
            self.assertTrue(env.robot.callback_active)
            self.assertEqual(env.robot.callback_attempt_index, 2)
            self.assertEqual(env.robot.callback_phase, RobotCallbackPhase.TURN)
        finally:
            env.close()

    def test_callback_success_at_cue_end_triggers_happy_once(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=361)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.FOLLOWING, reason="test_force_following")
            self._set_callback_state(
                env,
                target_idx,
                phase=RobotCallbackPhase.CUE,
                attempt_index=1,
                cue_elapsed_steps=env._get_callback_cue_steps(),
                cue_completed_this_step=True,
            )
            events = env._default_events()
            env._resolve_completed_callback_cue(events)

            self.assertTrue(events["callback_completed"])
            self.assertTrue(events["callback_success"])
            self.assertTrue(events["happy_triggered"])
            self.assertFalse(env.robot.callback_active)
            self.assertGreater(env.robot.happy_hold_steps_remaining, 0)

            events_second = env._default_events()
            env._resolve_completed_callback_cue(events_second)
            self.assertFalse(events_second["happy_triggered"])
        finally:
            env.close()

    def test_second_callback_completion_always_ends_callback(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=362)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self._set_callback_state(
                env,
                target_idx,
                phase=RobotCallbackPhase.CUE,
                attempt_index=2,
                cue_elapsed_steps=env._get_callback_cue_steps(),
                cue_completed_this_step=True,
            )
            events = env._default_events()
            env._resolve_completed_callback_cue(events)

            self.assertTrue(events["callback_completed"])
            self.assertFalse(events["callback_attempt_2_started"])
            self.assertFalse(env.robot.callback_active)
            self.assertIsNone(env.callback_active_target_idx)
        finally:
            env.close()

    def test_distracted_lambda_zero_never_triggers_after_threshold(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=2)
            dt = float(env.timestep)
            for human in env.humans:
                human.set_following_distracted_window_active(True)
                human.following_steps = int(np.ceil((human.following_distracted_ramp_start_seconds + 120.0) / dt))
                for _ in range(100):
                    self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
        finally:
            env.close()

    def test_distracted_threshold_is_strict_for_nd_and_normal(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=0.15,
            distracted_lambda_max_normal_per_sec=0.08,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=3)
            dt = float(env.timestep)
            for human in (env.humans[0], env.humans[1]):
                human.set_following_distracted_window_active(True)
                human.following_steps = human._get_distracted_follow_threshold_steps(dt=dt)
                self.assertEqual(human._compute_distracted_follow_step_probability(dt=dt), 0.0)
                with patch("numpy.random.rand", return_value=0.0):
                    self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
        finally:
            env.close()

    def test_distracted_probability_ramps_up_after_threshold(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=0.15,
            distracted_lambda_max_normal_per_sec=0.08,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=4)
            dt = float(env.timestep)
            human = env.humans[1]
            self.assertEqual(human.profile, HumanProfile.NORMAL)
            human.set_following_distracted_window_active(True)
            early_seconds = human.following_distracted_ramp_start_seconds + 0.2 * human.following_distracted_rise_seconds
            late_seconds = human.following_distracted_ramp_start_seconds + 0.8 * human.following_distracted_rise_seconds
            human.following_steps = int(np.ceil(early_seconds / dt))
            p_early = human._compute_distracted_follow_step_probability(dt=dt)
            human.following_steps = int(np.ceil(late_seconds / dt))
            p_late = human._compute_distracted_follow_step_probability(dt=dt)
            self.assertGreater(p_late, p_early)
            self.assertGreater(p_early, 0.0)
            probe = 0.5 * (p_early + p_late)
            with patch("numpy.random.rand", return_value=probe):
                human.following_steps = int(np.ceil(early_seconds / dt))
                self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
                human.following_steps = int(np.ceil(late_seconds / dt))
                self.assertEqual(human._maybe_trigger_following_variant(dt=dt), HumanMode.DISTRACTED)
        finally:
            env.close()

    def test_distracted_step_probability_matches_lambda_max_asymptote(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=0.15,
            distracted_lambda_max_normal_per_sec=0.08,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=44)
            dt = float(env.timestep)
            human = env.humans[0]
            human.set_following_distracted_window_active(True)
            long_follow_seconds = human.following_distracted_ramp_start_seconds + 10.0 * human.following_distracted_rise_seconds
            human.following_steps = int(np.ceil(long_follow_seconds / dt))
            p_step = human._compute_distracted_follow_step_probability(dt=dt)
            expected = float(1.0 - np.exp(-human.following_distracted_lambda_max_per_sec * dt))
            self.assertAlmostEqual(p_step, expected, places=12)
        finally:
            env.close()

    def test_impatient_prob_zero_never_triggers(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=5)
            dt = float(env.timestep)
            for human in env.humans:
                for _ in range(100):
                    self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
        finally:
            env.close()

    def test_impatient_prob_one_always_triggers(self):
        env = self._make_env(
            impatient_prob=1.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=6)
            dt = float(env.timestep)
            for human in env.humans:
                self.assertEqual(human._maybe_trigger_following_variant(dt=dt), HumanMode.IMPATIENT)
        finally:
            env.close()

    def test_listening_front_sector_helper_matches_expected_angles(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=908)
            human = env.humans[0]
            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            self.assertTrue(
                human.is_within_listening_front_sector(
                    point_xy=np.array([1.0, 0.0], dtype=np.float32),
                    robot_xy=robot_xy,
                    robot_yaw=0.0,
                    sector_half_angle=env.listen_front_sector_half_angle,
                )
            )
            self.assertFalse(
                human.is_within_listening_front_sector(
                    point_xy=np.array([-1.0, 0.0], dtype=np.float32),
                    robot_xy=robot_xy,
                    robot_yaw=0.0,
                    sector_half_angle=env.listen_front_sector_half_angle,
                )
            )
        finally:
            env.close()

    def test_turn_to_display_uses_fixed_180_degree_target(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=909)
            env.robot.turn_target_yaw = None
            env.robot.turn_done = False
            action = env.robot._turn_to_crowd_action(
                robot_pose=(2.0, 2.0, np.pi / 2.0),
                human_xyz=np.array([[4.0, 2.0, 0.0], [4.0, 3.0, 0.0]], dtype=np.float32),
            )
            expected_target = env.robot._wrap_to_pi((np.pi / 2.0) + np.pi)
            self.assertAlmostEqual(env.robot.turn_target_yaw, expected_target, places=6)
            self.assertLess(float(action[2]), 0.0)
        finally:
            env.close()

    def test_entering_listen_starts_intro_delay_and_defers_speaker(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=907)
            env.robot.current_waypoint_idx = 0
            env.robot.listen_mode = False
            env.robot.listen_done = False
            env.robot.turn_done = True
            self._set_robot_pose(env, x=1.0, y=5.0, yaw=0.0)
            self._set_human_pose(env, env.humans[0], x=0.0, y=5.0, yaw=0.0)

            _, _, _, _, info = env.step(None)

            self.assertTrue(info["events"]["entered_listen"])
            self.assertFalse(info["events"]["started_listen_wait"])
            self.assertTrue(info["status"]["listen_mode"])
            self.assertTrue(info["status"]["listen_intro_delay"]["active"])
            self.assertEqual(info["status"]["listen_intro_delay"]["counter"], 1)
            self.assertEqual(info["status"]["listen_intro_delay"]["steps"], env.listen_intro_delay_steps)
            self.assertEqual(
                info["status"]["listen_intro_delay"]["remaining"],
                env.listen_intro_delay_steps - 1,
            )
            self.assertFalse(info["status"]["listen_wait"]["active"])
            self.assertFalse(info["status"]["speaker_active"])
            self.assertEqual(info["humans"]["mode"][0], HumanMode.LISTENING)
        finally:
            env.close()

    def test_intro_delay_keeps_listening_humans_moving_and_then_starts_wait(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=908)
            env.robot.current_waypoint_idx = 0
            env.robot.listen_mode = False
            env.robot.listen_done = False
            env.robot.turn_done = True
            human = env.humans[0]
            self._set_robot_pose(env, x=1.0, y=5.0, yaw=0.0)
            self._set_human_pose(env, human, x=0.0, y=5.0, yaw=0.0)
            before_xy = np.array(env._get_human_poses()[0, :2], dtype=np.float32)

            _, _, _, _, info = env.step(None)

            after_xy = np.array(info["humans"]["pose_xy"][0], dtype=np.float32)
            self.assertTrue(info["status"]["listen_intro_delay"]["active"])
            self.assertFalse(info["status"]["listen_wait"]["active"])
            self.assertGreater(
                float(np.linalg.norm(after_xy - before_xy)),
                0.0,
            )

            remaining_intro_steps = int(info["status"]["listen_intro_delay"]["remaining"])
            for _ in range(max(0, remaining_intro_steps - 1)):
                _, _, _, _, info = env.step(None)
                self.assertFalse(info["events"]["started_listen_wait"])
                self.assertTrue(info["status"]["listen_intro_delay"]["active"])

            _, _, _, _, final_info = env.step(None)
            self.assertTrue(final_info["events"]["started_listen_wait"])
            self.assertFalse(final_info["status"]["listen_intro_delay"]["active"])
            self.assertTrue(final_info["status"]["listen_wait"]["active"])
            self.assertTrue(final_info["status"]["speaker_active"])
        finally:
            env.close()

    def test_distracted_window_inactive_blocks_trigger_before_first_listen_complete(self):

        env = self._make_env(
            distracted_lambda_max_nd_per_sec=100.0,
            distracted_lambda_max_normal_per_sec=100.0,
            distracted_ramp_start_nd_seconds=0.0,
            distracted_ramp_start_normal_seconds=0.0,
            distracted_rise_nd_seconds=1.0,
            distracted_rise_normal_seconds=1.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=7)
            dt = float(env.timestep)
            human = env.humans[0]
            human.following_steps = int(np.ceil(5.0 / dt))
            human.set_following_distracted_window_active(env._is_distracted_follow_window_active())
            self.assertFalse(env._is_distracted_follow_window_active())
            with patch("numpy.random.rand", return_value=0.0):
                self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
        finally:
            env.close()

    def test_distracted_window_active_during_travel_between_explanations(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=100.0,
            distracted_lambda_max_normal_per_sec=100.0,
            distracted_ramp_start_nd_seconds=0.0,
            distracted_ramp_start_normal_seconds=0.0,
            distracted_rise_nd_seconds=1.0,
            distracted_rise_normal_seconds=1.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=8)
            env.robot.listen_done = True
            env.robot.listen_mode = False
            env.listen_wait_active = False
            dt = float(env.timestep)
            human = env.humans[0]
            human.following_steps = int(np.ceil(5.0 / dt))
            human.set_following_distracted_window_active(env._is_distracted_follow_window_active())
            self.assertTrue(env._is_distracted_follow_window_active())
            with patch("numpy.random.rand", return_value=0.0):
                self.assertEqual(human._maybe_trigger_following_variant(dt=dt), HumanMode.DISTRACTED)
        finally:
            env.close()

    def test_distracted_window_closes_once_final_listening_starts_or_waits(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=9)
            env.robot.listen_done = True
            env.robot.listen_mode = True
            self.assertFalse(env._is_distracted_follow_window_active())
            env.robot.listen_mode = False
            env.listen_wait_active = True
            self.assertFalse(env._is_distracted_follow_window_active())
        finally:
            env.close()

    def test_following_duration_resets_on_env_reset(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=10)
            env.humans[0].following_steps = 123
            env.reset(seed=11)
            self.assertEqual(env.humans[0].following_steps, 0)
        finally:
            env.close()

    def test_following_duration_resets_when_human_leaves_following(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=12)
            human = env.humans[0]
            human.transition_to(HumanMode.FOLLOWING, reason="test_force_following")
            human.following_steps = 123
            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self.assertEqual(human.following_steps, 0)
        finally:
            env.close()

    def test_wait_window_keeps_minimal_listening_mode_without_wait_hazards(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=1.0,
            attack_wait_trigger_prob=1.0,
        )
        try:
            env.reset(seed=15)
            env.robot.listen_mode = True
            env.listen_wait_active = True
            env.listen_wait_counter = 0
            env.listen_wait_is_final = False
            env.robot.mode = "stop"
            self._set_robot_pose(env, x=2.0, y=2.0, yaw=0.0)

            _, _, _, _, info = env.step(None)

            self.assertFalse(info["events"]["overwhelmed_triggered"])
            self.assertFalse(info["events"]["attack_triggered"])
            self.assertFalse(info["events"]["fear_triggered"])
            self.assertFalse(info["events"]["move_back_triggered"])
            self.assertEqual(info["status"]["last_overwhelmed_trigger_indices"], [])
            self.assertEqual(info["status"]["last_attack_trigger_indices"], [])
            self.assertEqual(info["status"]["active_overwhelmed_indices"], [])
            self.assertEqual(info["status"]["active_attack_indices"], [])
            self.assertTrue(all(mode == HumanMode.LISTENING for mode in info["humans"]["mode"]))
        finally:
            env.close()

    def test_nonfinal_listen_wait_completes_without_all_humans_reached(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=17)
            env.robot.current_waypoint_idx = 0
            env.robot.listen_mode = True
            env.robot.listen_done = False
            env.listen_wait_active = True
            env.listen_wait_counter = env.listen_wait_steps - 1
            env.listen_wait_is_final = False
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening_wait")
            self._set_robot_pose(env, x=2.0, y=2.0, yaw=0.0)
            self._set_human_pose(env, human, x=1.0, y=2.0, yaw=0.0)

            _, _, terminated, truncated, info = env.step(None)

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["events"]["completed_listen_wait"])
            self.assertFalse(info["events"]["final_listen_ready"])
            self.assertFalse(info["status"]["listen_mode"])
            self.assertFalse(info["status"]["listen_wait"]["active"])
            self.assertTrue(info["status"]["post_explanation_hold"]["active"])
            self.assertTrue(env.post_explanation_hold_active)
            self.assertNotEqual(info["humans"]["mode"][0], HumanMode.WANDERING)
            self.assertTrue(env.robot.listen_done)
        finally:
            env.close()

    def test_post_explanation_hold_wait_role_does_not_reuse_stale_waypoint(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=117)
            env.robot.current_waypoint_idx = 0
            env.robot.listen_mode = True
            env.robot.listen_done = False
            env.listen_wait_active = True
            env.listen_wait_counter = env.listen_wait_steps - 1
            env.listen_wait_is_final = False
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening_wait")
            stale_waypoint = np.array([0.5, 0.5], dtype=np.float32)
            human.current_waypoint = stale_waypoint.copy()
            self._set_robot_pose(env, x=2.0, y=2.0, yaw=0.0)
            self._set_human_pose(env, human, x=3.5, y=1.0, yaw=0.0)

            env.step(None)
            _, _, terminated, truncated, info = env.step(None)

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["status"]["post_explanation_hold"]["active"])
            self.assertEqual(info["humans"]["yield_role"][0], "wait")
            self.assertEqual(info["humans"]["mode"][0], HumanMode.LISTENING)
            action_xy = np.array(
                [float(info["humans"]["action"]["vx"][0]), float(info["humans"]["action"]["vy"][0])],
                dtype=np.float32,
            )
            self.assertLess(float(np.linalg.norm(action_xy)), 1e-4)
            self.assertTrue(np.allclose(human.current_waypoint, stale_waypoint))
        finally:
            env.close()

    def test_anchor_listening_helper_keeps_hold_target_without_live_robot_pressure(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=120)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_anchor_wait")
            self._set_human_pose(env, human, x=3.5, y=1.0, yaw=0.0)
            pose = human._get_pose(env.data)
            ctx = {
                "robot_xy": np.array([4.5, 1.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "robot_speed": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "listen_radius": float(np.linalg.norm(np.array([3.5, 1.0], dtype=np.float32) - np.array([2.0, 2.0], dtype=np.float32))),
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }

            action = human._step_listening_with_anchor_target_and_live_repulsion(
                env.data,
                ctx,
                pose,
                anchor_robot_xy=np.array([2.0, 2.0], dtype=np.float32),
                anchor_robot_yaw=0.0,
                live_robot_xy=np.array([4.5, 1.0], dtype=np.float32),
            )

            self.assertLess(float(np.linalg.norm(action[:2])), 1e-4)
        finally:
            env.close()

    def test_anchor_listening_helper_uses_live_robot_repulsion_when_robot_approaches(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=121)
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_anchor_wait")
            self._set_human_pose(env, human, x=3.5, y=1.0, yaw=0.0)
            pose = human._get_pose(env.data)
            ctx = {
                "robot_xy": np.array([3.9, 1.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "robot_speed": 0.2,
                "repulsion": np.zeros(2, dtype=np.float32),
                "listen_radius": float(np.linalg.norm(np.array([3.5, 1.0], dtype=np.float32) - np.array([2.0, 2.0], dtype=np.float32))),
                "stand_threshold": env.listen_stand_threshold,
                "listening_sector_half_angle": env.listen_front_sector_half_angle,
                "dt": float(env.timestep),
            }

            action = human._step_listening_with_anchor_target_and_live_repulsion(
                env.data,
                ctx,
                pose,
                anchor_robot_xy=np.array([2.0, 2.0], dtype=np.float32),
                anchor_robot_yaw=0.0,
                live_robot_xy=np.array([3.9, 1.0], dtype=np.float32),
            )

            self.assertLess(float(action[0]), 0.0)
            self.assertGreater(float(np.linalg.norm(action[:2])), 1e-4)
            self.assertLess(float(human.last_v_hr[0]), 0.0)
        finally:
            env.close()

    def test_post_explanation_hold_close_human_gets_walkable_yield_target(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=118)
            env.robot.current_waypoint_idx = 0
            env.robot.listen_mode = True
            env.robot.listen_done = False
            env.listen_wait_active = True
            env.listen_wait_counter = env.listen_wait_steps - 1
            env.listen_wait_is_final = False
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_listening_wait")
            self._set_robot_pose(env, x=2.0, y=2.0, yaw=0.0)
            self._set_human_pose(env, human, x=1.1, y=2.0, yaw=0.0)

            _, _, terminated, truncated, info = env.step(None)

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["humans"]["yield_role"][0], "yield")
            target_xy = np.array(info["humans"]["yield_target_xy"][0], dtype=np.float32)
            self.assertTrue(human._is_point_in_walkable(target_xy, HUMAN_WALL_FOOTPRINT_RADIUS))
            self.assertGreater(
                float(np.linalg.norm(target_xy - np.array([2.0, 2.0], dtype=np.float32))),
                float(np.linalg.norm(np.array([1.1, 2.0], dtype=np.float32) - np.array([2.0, 2.0], dtype=np.float32))),
            )
        finally:
            env.close()

    def test_post_explanation_hold_restores_following_after_robot_restart_threshold(self):
        env = self._make_env(
            n_humans=1,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=119)
            robot_xy = np.array([2.0, 2.0], dtype=np.float32)
            human_xy = np.array([[3.0, 2.0]], dtype=np.float32)
            env.follow_humans = False
            env._start_post_explanation_hold(robot_xy=robot_xy, robot_yaw=0.0, human_xy=human_xy)

            env._maybe_finish_post_explanation_hold(
                robot_xy=np.array([2.0 + POST_EXPLANATION_HOLD_RESUME_DISTANCE + 0.1, 2.0], dtype=np.float32),
                robot_speed=POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD + 0.1,
            )

            self.assertFalse(env.post_explanation_hold_active)
            self.assertTrue(env.follow_humans)
        finally:
            env.close()

    def test_final_listen_wait_terminates_without_all_humans_reached(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=18)
            env.robot.current_waypoint_idx = len(env.robot.waypoints) - 1
            env.robot.listen_mode = True
            env.robot.listen_done = True
            env.listen_wait_active = True
            env.listen_wait_counter = env.listen_wait_steps - 1
            env.listen_wait_is_final = True
            human = env.humans[0]
            human.transition_to(HumanMode.LISTENING, reason="test_force_final_listening_wait")
            self._set_robot_pose(env, x=11.0, y=-12.5, yaw=0.0)
            self._set_human_pose(env, human, x=10.0, y=-12.5, yaw=0.0)

            _, _, terminated, truncated, info = env.step(None)

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["events"]["completed_listen_wait"])
            self.assertTrue(info["events"]["final_listen_ready"])
            self.assertTrue(info["status"]["terminated_reason"] == "final_listen_ready")
        finally:
            env.close()

    def test_reset_step_signature_stable(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            reset_out = env.reset(seed=17)
            self.assertEqual(len(reset_out), 2)
            step_out = env.step(None)
            self.assertEqual(len(step_out), 5)
        finally:
            env.close()

    def test_random_waypoints_stay_inside_walkable_area(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=18)
            for human in env.humans:
                for _ in range(200):
                    waypoint = human._random_waypoint()
                    self._assert_in_walkable(human, waypoint)
        finally:
            env.close()

    def test_velocity_projection_keeps_predicted_position_walkable(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=19)
            human = env.humans[0]
            current_xy = np.array([9.749, 5.0], dtype=np.float32)
            raw_v = np.array([1.5, 0.0], dtype=np.float32)
            safe_v = human._constrain_velocity_with_walkable(
                x=float(current_xy[0]),
                y=float(current_xy[1]),
                v_xy=raw_v,
                dt=float(env.timestep),
                margin=HUMAN_WALL_FOOTPRINT_RADIUS,
            )
            next_xy = current_xy + float(env.timestep) * safe_v
            self._assert_in_walkable(human, next_xy)
            self.assertLessEqual(float(np.linalg.norm(safe_v)), float(human.max_speed) + 1e-6)
        finally:
            env.close()

    def test_segment_walkable_rejects_crossing_room_a_bottom_wall(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=21)
            human = env.humans[0]
            start_xy = np.array([5.0, 0.6], dtype=np.float32)
            end_xy = np.array([5.0, -0.6], dtype=np.float32)
            self.assertFalse(
                human._is_segment_walkable(
                    start_xy=start_xy,
                    end_xy=end_xy,
                    margin=HUMAN_WALL_FOOTPRINT_RADIUS,
                )
            )
        finally:
            env.close()

    def test_segment_walkable_accepts_corridor_doorway_passage(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=22)
            human = env.humans[0]
            start_xy = np.array([8.5, 0.6], dtype=np.float32)
            end_xy = np.array([8.5, -0.6], dtype=np.float32)
            self.assertTrue(
                human._is_segment_walkable(
                    start_xy=start_xy,
                    end_xy=end_xy,
                    margin=HUMAN_WALL_FOOTPRINT_RADIUS,
                )
            )
        finally:
            env.close()

    def test_distracted_target_sampling_clamps_single_sample_to_farthest_walkable_point(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=23)
            human = env.humans[0]
            current_xy = np.array([5.0, 0.6], dtype=np.float32)

            sampled_target_xy = np.array([5.0, -0.4], dtype=np.float32)
            with patch.object(
                human,
                "_sample_distracted_target_candidate",
                return_value=(np.deg2rad(-90.0), sampled_target_xy),
            ) as sample_candidate:
                human._initialize_distracted_target(current_xy=current_xy, current_yaw=0.0)

            sample_candidate.assert_called_once()
            called_kwargs = sample_candidate.call_args.kwargs
            np.testing.assert_allclose(called_kwargs["current_xy"], current_xy, atol=1e-6)
            self.assertEqual(called_kwargs["current_yaw"], 0.0)
            self.assertGreaterEqual(float(human.distracted_target_xy[1]), float(sampled_target_xy[1]))
            self.assertLessEqual(float(human.distracted_target_xy[1]), float(current_xy[1]))
            self.assertTrue(
                human._is_segment_walkable(
                    start_xy=current_xy,
                    end_xy=human.distracted_target_xy,
                    margin=HUMAN_WALL_FOOTPRINT_RADIUS,
                )
            )
        finally:
            env.close()

    def test_high_trigger_run_keeps_all_humans_in_walkable_area(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=0.8,
            distracted_lambda_max_normal_per_sec=0.8,
            distracted_ramp_start_nd_seconds=0.0,
            distracted_ramp_start_normal_seconds=0.0,
            distracted_rise_nd_seconds=1.0,
            distracted_rise_normal_seconds=1.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=1.0,
            attack_wait_trigger_prob=1.0,
        )
        try:
            env.reset(seed=20)
            for _ in range(60):
                _, _, terminated, truncated, info = env.step(None)
                for idx, xy in enumerate(info["humans"]["pose_xy"]):
                    self._assert_in_walkable(env.humans[idx], xy, tol=2e-2)
                if terminated or truncated:
                    break

            env.listen_wait_active = True
            for _ in range(120):
                _, _, terminated, truncated, info = env.step(None)
                for idx, xy in enumerate(info["humans"]["pose_xy"]):
                    self._assert_in_walkable(env.humans[idx], xy, tol=2e-2)
                if terminated or truncated:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
