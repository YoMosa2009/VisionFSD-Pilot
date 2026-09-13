from __future__ import annotations

import pathlib
import sys
import time
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lidar_visualizer import LidarPoint, LivePolarMap
from robot_autonomy import CameraSafety, ArduinoStatus, AutonomousPolicy, SectorClearance, STEER_HEADINGS, corridor_profile
from robot_camera_motion import CameraMotionState, estimate_motion
from robot_slam_lite import LidarSlamLite


class CameraGeometryTests(unittest.TestCase):
    @staticmethod
    def texture():
        noise = np.random.default_rng(42).integers(0, 256, (120, 160), dtype=np.uint8)
        return cv2.GaussianBlur(noise, (3, 3), 0)

    def test_stationary_texture_reduces_false_forward_and_reverse_translation(self):
        frame = self.texture()
        for pwm in (118, -118):
            state = estimate_motion(frame, frame, .1, 62., pwm, pwm, 1.)
            self.assertTrue(state.fresh)
            self.assertGreater(state.confidence, .55)
            self.assertFalse(state.motion_observed)
            self.assertEqual(state.translation_scale, .2)

    def test_frame_pipeline_supplies_approach_cue_to_policy(self):
        safety = object.__new__(CameraSafety)
        safety._fov = 62.
        safety._flow_gray = None
        safety._flow_at = 0.
        policy = AutonomousPolicy(0., 118)
        policy.left_pwm = policy.right_pwm = 118
        now = time.monotonic()
        clear = SectorClearance(2., 2., 2., True, 2., 2.,
                                np.full(STEER_HEADINGS.size, 2., dtype=np.float32))
        texture = self.texture()
        saw_caution = False
        for i in range(10):
            at = now + i * .05
            frame = cv2.warpAffine(texture, cv2.getRotationMatrix2D((80, 60), 0., 1.03 ** i), (160, 120))
            safety._update_motion(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), at, policy.left_pwm, policy.right_pwm)
            command = policy.decide(clear, ArduinoStatus(100., 'F', at), False, at,
                                    camera_motion=safety.motion)
            self.assertEqual(command, 'F')
            saw_caution |= policy.visual_caution
        self.assertTrue(saw_caution)
        self.assertGreater(safety.motion.processing_ms, 0.)

    def test_blank_image_abstains_from_stuck_and_yaw_evidence(self):
        frame = np.zeros((120, 160), dtype=np.uint8)
        state = estimate_motion(frame, frame, .1, 62., -118, 118, 1.)
        self.assertFalse(state.fresh)
        self.assertEqual(state.quality, 'LOW_TEXTURE')
        self.assertIsNone(state.yaw_rate_dps)
        self.assertEqual(state.translation_scale, 1.)

    def test_local_moving_patch_is_not_whole_robot_motion(self):
        previous = np.zeros((120, 160), dtype=np.uint8)
        previous[10:50, 10:55] = self.texture()[:40, :45]
        following = np.roll(previous, 2, axis=1)
        state = estimate_motion(previous, following, .1, 62., 118, 118, 1.)
        self.assertFalse(state.fresh)
        self.assertEqual(state.quality, 'LOCAL_FEATURES')

    def test_expansion_detected_but_turning_and_reverse_abstain(self):
        previous = self.texture()
        transform = cv2.getRotationMatrix2D((80, 60), 0., 1.08)
        following = cv2.warpAffine(previous, transform, (160, 120))
        forward = estimate_motion(previous, following, .1, 62., 118, 118, 1.)
        self.assertGreater(forward.expansion_rate_s, .5)
        for left, right in ((-118, -118), (118, 90)):
            state = estimate_motion(previous, following, .1, 62., left, right, 1.)
            self.assertEqual(state.expansion_rate_s, 0.)

    def test_translation_does_not_look_like_approach_expansion(self):
        previous = self.texture()
        following = cv2.warpAffine(previous, np.float32([[1, 0, -2], [0, 1, 0]]), (160, 120))
        state = estimate_motion(previous, following, .1, 62., 118, 90, 1.)
        self.assertGreater(state.confidence, .35)
        self.assertGreater(state.yaw_rate_dps, 0.)
        self.assertEqual(state.expansion_rate_s, 0.)

    def test_bad_round_trip_matches_abstain(self):
        frame = self.texture()
        features = np.float32([[[x, y]] for y in (20, 50, 80) for x in (20, 50, 80, 110, 140)])
        status = np.ones((len(features), 1), dtype=np.uint8)
        error = np.zeros_like(status, dtype=np.float32)
        with mock.patch('robot_camera_motion.cv2.goodFeaturesToTrack', return_value=features), mock.patch(
            'robot_camera_motion.cv2.calcOpticalFlowPyrLK',
            side_effect=[(features + 2, status, error), (features + 5, status, error)]
        ):
            state = estimate_motion(frame, frame, .1, 62., 118, 118, 1.)
        self.assertFalse(state.fresh)
        self.assertEqual(state.quality, 'INCONSISTENT')

    def test_approach_requires_distinct_sustained_frames_and_only_slows(self):
        policy = AutonomousPolicy(0., 118)
        policy.left_pwm = policy.right_pwm = 118
        now = time.monotonic()
        clear = SectorClearance(2., 2., 2., True, 2., 2.,
                                np.full(STEER_HEADINGS.size, 2., dtype=np.float32))
        first = CameraMotionState(fresh=True, confidence=.9, motion_observed=True,
                                  captured_at=now, expansion_rate_s=.8)
        for offset in (0., .1, .2):
            policy._observe_visual_approach(first, now + offset)
        self.assertFalse(policy.visual_caution)
        outputs = []
        for i in range(1, 7):
            at = now + i * .1
            state = CameraMotionState(fresh=True, confidence=.9, motion_observed=True,
                                      captured_at=at, expansion_rate_s=.8)
            command = policy.decide(clear, ArduinoStatus(100., 'F', at), False, at,
                                    camera_motion=state)
            outputs.append((command, policy.left_pwm, policy.right_pwm))
        self.assertTrue(policy.visual_caution)
        self.assertTrue(all(cmd == 'F' and left > 0 and right > 0 for cmd, left, right in outputs))
        self.assertLess(policy.cruise_pwm, 118)
        self.assertIn('VISION_APPROACH', policy.reason)
        policy._observe_visual_approach(first, now + 2.)
        self.assertFalse(policy.visual_caution)
        command = policy.decide(clear, ArduinoStatus(100., 'F', now + 2.), False, now + 2.,
                                camera_ready=False)
        self.assertEqual(command, 'STOP')


class LidarDetailTests(unittest.TestCase):
    def test_half_degree_map_keeps_nearby_distinct_rays(self):
        live = LivePolarMap(bin_count=720)
        points = [LidarPoint(.1, 500, 200, 1.), LidarPoint(.4, 2000, 200, 1.)]
        live.update(points, 8, 80, 6000)
        self.assertEqual(len(live.fresh(1., .18)), 2)

    def test_older_high_confidence_return_cannot_replace_new_geometry(self):
        live = LivePolarMap(bin_count=720)
        live.update([LidarPoint(10., 500, 90, 2.)], 8, 80, 6000)
        live.update([LidarPoint(10., 2000, 255, 1.)], 8, 80, 6000)
        self.assertEqual(live.fresh(2., .18)[0][1].distance_mm, 500)

    def test_same_timestamp_preserves_nearest_accepted_surface(self):
        live = LivePolarMap(bin_count=720)
        live.update([LidarPoint(10., 500, 90, 1.), LidarPoint(10., 2000, 255, 1.)], 8, 80, 6000)
        self.assertEqual(live.fresh(1., .18)[0][1].distance_mm, 500)

    def test_subdegree_obstacle_at_body_edge_is_not_rounded_away(self):
        point = LidarPoint(16.8, 500, 200, 1.)
        profile = corridor_profile([(0, point)], np.array([0.], dtype=np.float32))
        self.assertLess(profile[0], .45)
        self.assertAlmostEqual(profile[0], .5 * np.cos(np.radians(16.8)) - .075, places=5)

    def test_map_yaw_deskew_aligns_rotating_returns_without_changing_ranges(self):
        points = [(i, LidarPoint((-30. * age) % 360., 1000, 200, 1. - age))
                  for i, age in enumerate(np.linspace(0., .1, 20))]
        corrected = LidarSlamLite.deskew_points(points, 1., 30.)
        self.assertTrue(all(abs((p.angle_deg + 180.) % 360. - 180.) < 1e-5 for _, p in corrected))
        self.assertTrue(all(p.distance_mm == 1000 for _, p in corrected))
        self.assertNotEqual(points[-1][1].angle_deg, corrected[-1][1].angle_deg)
        self.assertIs(LidarSlamLite.deskew_points(points, 1., None), points)
        self.assertIs(LidarSlamLite.deskew_points(points, 1., float('nan')), points)

    def test_map_update_applies_deskew_but_input_remains_raw(self):
        mapper = LidarSlamLite()
        points = [(0, LidarPoint(357., 1000, 200, .9))]
        with mock.patch.object(mapper, '_integrate_points', wraps=mapper._integrate_points) as integrate:
            mapper.update(points, 118, 90, 1., imu_yaw_rate_dps=30.)
        corrected = integrate.call_args.args[0]
        self.assertAlmostEqual(corrected[0][1].angle_deg, 0.)
        self.assertEqual(points[0][1].angle_deg, 357.)
        self.assertEqual(mapper.state().map_updates, 1)

    def test_deskew_rejects_stale_points_and_bounds_correction(self):
        stale = [(0, LidarPoint(0., 1000, 200, .5))]
        self.assertEqual(LidarSlamLite.deskew_points(stale, 1., 60.)[0][1].angle_deg, 0.)
        recent = [(0, LidarPoint(0., 1000, 200, .8))]
        self.assertEqual(LidarSlamLite.deskew_points(recent, 1., 60.)[0][1].angle_deg, 10.)
        self.assertIs(LidarSlamLite.deskew_points(recent, 1., 90.), recent)

    def test_more_rays_do_not_artificially_multiply_cell_confidence(self):
        single, dense = LidarSlamLite(), LidarSlamLite()
        one = [(0, LidarPoint(0., 1000, 200, 1.))]
        single._integrate_points(one)
        dense._integrate_points(one * 100)
        np.testing.assert_array_equal(single.grid, dense.grid)

    def test_all_hit_endpoints_are_observed_even_with_bounded_free_rays(self):
        mapper = LidarSlamLite()
        points = [(i, LidarPoint(float(i) / 2., 2500, 200, 1.)) for i in range(720)]
        with mock.patch('robot_slam_lite.cv2.line', wraps=cv2.line) as line:
            mapper._integrate_points(points)
        self.assertLessEqual(line.call_count, 240)
        rows, cols = mapper._latest_hits.T
        self.assertTrue(np.all(mapper.observed[rows, cols] > 0))
        self.assertGreater(len(np.unique(mapper._latest_hits, axis=0)), 360)


if __name__ == '__main__':
    unittest.main()
