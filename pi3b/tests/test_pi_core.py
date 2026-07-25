from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionfsd_pi import (
    AsyncLaneDetector,
    Detection,
    LaneEstimate,
    LowCostLaneDetector,
    RUNTIME_VERSION,
    Rates,
    SceneObjectTracker,
    TFLiteVehicleDetector,
    TargetSelector,
    _camera_panel,
    _world_panel,
    bearing_deg,
    compose_view,
    estimate_car_range_m,
    is_near_vehicle,
    lane_position,
    nms,
    parse_args,
    touch_action_at,
    touch_buttons,
)


class PiCoreTests(unittest.TestCase):
    def test_range_and_bearing_are_bounded(self) -> None:
        box = (270, 250, 370, 430)
        self.assertGreater(estimate_car_range_m(box, 640, 70), 5.0)
        self.assertLess(abs(bearing_deg(box, 640, 70)), 1.0)

    def test_nms_keeps_the_best_box(self) -> None:
        items = [Detection("car", 0.90, (10, 10, 100, 100)), Detection("car", 0.60, (12, 12, 99, 99))]
        self.assertEqual(nms(items), [items[0]])

    def test_nms_preserves_overlapping_different_scene_classes(self) -> None:
        car = Detection("car", 0.90, (10, 10, 100, 100))
        person = Detection("person", 0.75, (10, 10, 100, 100))
        self.assertEqual(nms([car, person]), [car, person])

    def test_nms_suppresses_overlapping_vehicle_class_hypotheses(self) -> None:
        car = Detection("car", 0.91, (10, 10, 100, 100))
        truck = Detection("truck", 0.72, (10, 10, 100, 100))
        self.assertEqual(nms([truck, car]), [car])

    def test_target_requires_persistent_materially_closer_challenger(self) -> None:
        selector = TargetSelector(70.0)
        now = time.perf_counter()
        incumbent = Detection("car", 0.9, (280, 250, 360, 420))
        self.assertIsNone(selector.update([incumbent], 640, now))
        first = selector.update([incumbent], 640, now + 0.03)
        self.assertIsNotNone(first)
        first_id = first.track_id
        # A closer challenger still needs three detector updates, preventing a
        # single noisy box from replacing a stable lead.
        challenger = Detection("car", 0.9, (210, 100, 430, 460))
        for step in range(3):
            current = selector.update([incumbent, challenger], 640, now + 0.06 + step * 0.03)
            self.assertEqual(current.track_id, first_id)
        current = selector.update([incumbent, challenger], 640, now + 0.15)
        self.assertNotEqual(current.track_id, first_id)

    def test_target_switches_to_a_new_forward_corridor_vehicle(self) -> None:
        selector = TargetSelector(70.0)
        side_vehicle = Detection("car", 0.9, (0, 120, 240, 460))
        forward_vehicle = Detection("car", 0.9, (290, 160, 350, 400))
        self.assertIsNone(selector.update([side_vehicle], 640, 1.0))
        first = selector.update([side_vehicle], 640, 1.05)
        self.assertIsNotNone(first)
        self.assertEqual(selector.update([side_vehicle, forward_vehicle], 640, 1.1).track_id, first.track_id)
        current = selector.update([side_vehicle, forward_vehicle], 640, 1.15)
        self.assertEqual(current.detection.box, forward_vehicle.box)

    def test_one_frame_vehicle_false_positive_never_becomes_lead(self) -> None:
        selector = TargetSelector(70.0)
        false_car = Detection("car", 0.88, (280, 160, 360, 420))
        self.assertIsNone(selector.update([false_car], 640, 1.0, 480))
        self.assertIsNone(selector.update([], 640, 1.1, 480))

    def test_lane_position_classifies_ego_left_and_right_lanes(self) -> None:
        lane = LaneEstimate(0.25, 0.75, 0.50, True, 0.0, 0.44, 0.56, 0.90)
        ego = Detection("car", 0.9, (280, 180, 360, 430))
        left = Detection("car", 0.9, (55, 180, 175, 430))
        right = Detection("car", 0.9, (465, 180, 585, 430))
        self.assertEqual(lane_position(ego, 640, 480, 70.0, lane)[0], 0)
        self.assertEqual(lane_position(left, 640, 480, 70.0, lane)[0], -1)
        self.assertEqual(lane_position(right, 640, 480, 70.0, lane)[0], 1)

    def test_target_prefers_same_lane_over_closer_side_vehicle(self) -> None:
        selector = TargetSelector(70.0)
        lane = LaneEstimate(0.25, 0.75, 0.50, True, 0.0, 0.44, 0.56, 0.90)
        close_left = Detection("car", 0.94, (25, 150, 215, 455))
        farther_ego = Detection("car", 0.82, (285, 190, 355, 410))
        self.assertIsNone(selector.update([close_left, farther_ego], 640, 1.0, 480, lane))
        target = selector.update([close_left, farther_ego], 640, 1.1, 480, lane)
        self.assertIsNotNone(target)
        self.assertEqual(target.detection.box, farther_ego.box)
        self.assertEqual(target.lane_slot, 0)

    def test_lane_slot_resists_short_boundary_jitter(self) -> None:
        selector = TargetSelector(70.0)
        lane = LaneEstimate(0.25, 0.75, 0.50, True, 0.0, 0.44, 0.56, 0.90)
        ego = Detection("car", 0.90, (350, 170, 490, 440))
        noisy_right = Detection("car", 0.90, (430, 170, 570, 440))
        for step in range(3):
            target = selector.update([ego], 640, 1.0 + step * 0.05, 480, lane)
        self.assertIsNotNone(target)
        self.assertEqual(target.lane_slot, 0)
        for step in range(2):
            target = selector.update([noisy_right], 640, 1.2 + step * 0.05, 480, lane)
            self.assertEqual(target.lane_slot, 0)
        target = selector.update([noisy_right], 640, 1.3, 480, lane)
        self.assertEqual(target.lane_slot, 1)

    def test_vehicle_label_voting_resists_one_frame_class_jitter(self) -> None:
        selector = TargetSelector(70.0)
        car = Detection("car", 0.84, (275, 160, 365, 420))
        for step in range(3):
            selector.update([car], 640, 1.0 + step * 0.05, 480)
        noisy_bus = Detection("bus", 0.93, car.box)
        target = selector.update([noisy_bus], 640, 1.2, 480)
        self.assertIsNotNone(target)
        self.assertEqual(target.detection.label, "car")

    def test_vehicle_box_is_smoothed_after_a_jittery_measurement(self) -> None:
        selector = TargetSelector(70.0)
        first = Detection("car", 0.9, (270, 160, 350, 420))
        moved = Detection("car", 0.9, (290, 160, 370, 420))
        selector.update([first], 640, 1.0, 480)
        target = selector.update([moved], 640, 1.1, 480)
        self.assertIsNotNone(target)
        self.assertGreater(target.detection.box[0], first.box[0])
        self.assertLess(target.detection.box[0], moved.box[0])

    def test_world_vehicle_list_contains_only_selected_target(self) -> None:
        selector = TargetSelector(70.0)
        lane = LaneEstimate(0.25, 0.75, 0.50, True, 0.0, 0.44, 0.56, 0.90)
        detections = [
            Detection("car", 0.90, (55, 180, 175, 430)),
            Detection("car", 0.92, (280, 180, 360, 430)),
            Detection("truck", 0.88, (465, 180, 585, 430)),
        ]
        selector.update(detections, 640, 1.0, 480, lane)
        target = selector.update(detections, 640, 1.1, 480, lane)
        vehicles = selector.vehicles(1.1)
        self.assertIsNotNone(target)
        self.assertEqual(len(vehicles), 1)
        self.assertEqual(vehicles[0].track_id, target.track_id)
        self.assertEqual(vehicles[0].lane_slot, 0)

    def test_far_horizon_vehicle_never_becomes_target(self) -> None:
        selector = TargetSelector(70.0)
        far_car = Detection("car", 0.96, (309, 175, 331, 207))
        self.assertFalse(is_near_vehicle(far_car, 640, 480))
        for step in range(6):
            self.assertIsNone(selector.update([far_car], 640, 1.0 + step * 0.1, 480))

    def test_close_target_is_not_replaced_by_far_centre_detection(self) -> None:
        selector = TargetSelector(70.0)
        close_car = Detection("car", 0.88, (260, 170, 380, 440))
        far_car = Detection("truck", 0.99, (309, 175, 331, 207))
        selector.update([close_car], 640, 1.0, 480)
        target = selector.update([close_car], 640, 1.1, 480)
        self.assertIsNotNone(target)
        target_id = target.track_id
        for step in range(5):
            target = selector.update([close_car, far_car], 640, 1.2 + step * 0.1, 480)
            self.assertEqual(target.track_id, target_id)

    def test_target_falls_back_to_nearest_adjacent_lane(self) -> None:
        selector = TargetSelector(70.0)
        lane = LaneEstimate(0.25, 0.75, 0.50, True, 0.0, 0.44, 0.56, 0.90)
        close_left = Detection("car", 0.92, (35, 145, 225, 455))
        farther_right = Detection("car", 0.94, (455, 180, 585, 420))
        selector.update([close_left, farther_right], 640, 1.0, 480, lane)
        target = selector.update([close_left, farther_right], 640, 1.1, 480, lane)
        self.assertIsNotNone(target)
        self.assertEqual(target.lane_slot, -1)

    def test_world_renderer_draws_at_most_one_vehicle(self) -> None:
        lane = LaneEstimate(0.25, 0.75, 0.50, True, 0.0, 0.44, 0.56, 0.90)
        target = TargetSelector(70.0)
        detections = [
            Detection("car", 0.90, (55, 180, 175, 430)),
            Detection("car", 0.92, (280, 180, 360, 430)),
            Detection("truck", 0.88, (465, 180, 585, 430)),
        ]
        target.update(detections, 640, 1.0, 480, lane)
        selected = target.update(detections, 640, 1.1, 480, lane)
        all_tracks = [
            target.current(1.1),
            target.current(1.1),
            target.current(1.1),
        ]
        with patch("visionfsd_pi._draw_world_vehicle") as draw_vehicle:
            _world_panel((640, 480), selected, all_tracks, [], lane, Rates())
        draw_vehicle.assert_called_once()

    def test_raw_yolo_output_decodes_car_without_a_runtime(self) -> None:
        # Decode the actual Pi export layout [1, 84, candidate_count] without
        # requiring the Pi-only tflite_runtime package in desktop CI.
        detector = object.__new__(TFLiteVehicleDetector)
        detector._confidence = 0.35
        detector._input_w = 320
        detector._input_h = 320
        output = np.zeros((1, 84, 2100), dtype=np.float32)
        output[0, 0:4, 0] = (160, 160, 80, 100)
        output[0, 4 + 2, 0] = 0.9
        decoded = detector._decode_yolo(output, 640, 480)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].label, "car")
        self.assertEqual(decoded[0].box, (240, 165, 400, 315))

    def test_standard_tflite_detection_postprocess_decodes_car(self) -> None:
        detector = object.__new__(TFLiteVehicleDetector)
        detector._confidence = 0.35
        boxes = np.zeros((1, 10, 4), dtype=np.float32)
        boxes[0, 0] = (0.2, 0.3, 0.8, 0.7)  # y1, x1, y2, x2
        classes = np.array([[2.0] + [0.0] * 9], dtype=np.float32)
        scores = np.array([[0.92] + [0.0] * 9], dtype=np.float32)
        count = np.array([1.0], dtype=np.float32)
        decoded = detector._decode_detection_postprocess([boxes, classes, scores, count], 640, 480)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded[0].box, (192, 96, 447, 384))

    def test_standard_tflite_detection_postprocess_decodes_world_only_classes(self) -> None:
        detector = object.__new__(TFLiteVehicleDetector)
        detector._confidence = 0.35
        boxes = np.array([[
            (0.2, 0.1, 0.7, 0.3),
            (0.1, 0.4, 0.6, 0.5),
            (0.3, 0.7, 0.8, 0.8),
        ]], dtype=np.float32)
        classes = np.array([[0.0, 9.0, 11.0]], dtype=np.float32)
        scores = np.array([[0.91, 0.82, 0.78]], dtype=np.float32)
        count = np.array([3.0], dtype=np.float32)
        decoded = detector._decode_detection_postprocess([boxes, classes, scores, count], 640, 480)
        self.assertEqual([item.label for item in decoded], ["person", "traffic_light", "stop_sign"])

    def test_world_objects_require_consecutive_evidence_before_display(self) -> None:
        tracker = SceneObjectTracker(70.0)
        person = Detection("person", 0.90, (280, 140, 330, 430))
        self.assertEqual(tracker.update([person], 640, 1.0), [])
        self.assertEqual(tracker.update([person], 640, 1.1), [])
        self.assertEqual(tracker.update([person], 640, 1.2), [])
        confirmed = tracker.update([person], 640, 1.3)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].detection.label, "person")

    def test_scene_hit_miss_hit_does_not_confirm(self) -> None:
        tracker = SceneObjectTracker(70.0)
        person = Detection("person", 0.90, (280, 140, 330, 430))
        self.assertEqual(tracker.update([person], 640, 1.0), [])
        self.assertEqual(tracker.update([], 640, 1.1), [])
        self.assertEqual(tracker.update([person], 640, 1.2), [])

    def test_confirmed_scene_object_hides_after_two_misses(self) -> None:
        tracker = SceneObjectTracker(70.0)
        person = Detection("person", 0.90, (280, 140, 330, 430))
        tracker.update([person], 640, 1.0)
        tracker.update([person], 640, 1.1)
        tracker.update([person], 640, 1.2)
        self.assertEqual(len(tracker.update([person], 640, 1.3)), 1)
        self.assertEqual(len(tracker.update([], 640, 1.38)), 1)
        self.assertEqual(tracker.update([], 640, 1.46), [])

    def test_low_confidence_sign_never_confirms(self) -> None:
        tracker = SceneObjectTracker(70.0)
        sign = Detection("stop_sign", 0.51, (400, 120, 455, 220))
        for step in range(6):
            self.assertEqual(tracker.update([sign], 640, 1.0 + step * 0.05), [])

    def test_moving_person_uses_centre_distance_match(self) -> None:
        tracker = SceneObjectTracker(70.0)
        boxes = (
            (250, 140, 300, 430), (265, 140, 315, 430),
            (280, 140, 330, 430), (295, 140, 345, 430),
        )
        result = []
        for step, box in enumerate(boxes):
            result = tracker.update([Detection("person", 0.90, box)], 640, 1.0 + step * 0.08)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].track_id, 1)

    def test_wide_person_false_positive_never_confirms(self) -> None:
        tracker = SceneObjectTracker(70.0)
        false_person = Detection("person", 0.96, (190, 260, 450, 410))
        for step in range(6):
            self.assertEqual(
                tracker.update([false_person], 640, 1.0 + step * 0.05, 480), []
            )

    def test_person_box_mostly_inside_vehicle_is_suppressed(self) -> None:
        tracker = SceneObjectTracker(70.0)
        vehicle = Detection("car", 0.88, (190, 180, 470, 450))
        false_person = Detection("person", 0.92, (270, 210, 350, 430))
        for step in range(6):
            self.assertEqual(
                tracker.update([vehicle, false_person], 640, 1.0 + step * 0.05, 480), []
            )

    def test_low_cost_lane_detector_requires_and_finds_a_pair(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame, (80, 479), (270, 250), (255, 255, 255), 8, cv2.LINE_AA)
        cv2.line(frame, (560, 479), (370, 250), (255, 255, 255), 8, cv2.LINE_AA)
        lane = LowCostLaneDetector().update(frame, 1.0)
        self.assertTrue(lane.observed)
        self.assertLess(lane.left_bottom_norm, 0.5)
        self.assertGreater(lane.right_bottom_norm, 0.5)

    def test_async_lane_detector_publishes_without_main_loop_work(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame, (80, 479), (270, 250), (255, 255, 255), 8, cv2.LINE_AA)
        cv2.line(frame, (560, 479), (370, 250), (255, 255, 255), 8, cv2.LINE_AA)
        worker = AsyncLaneDetector()
        try:
            worker.submit(1, frame)
            deadline = time.perf_counter() + 1.0
            lane = worker.latest(time.perf_counter())
            while not lane.observed and time.perf_counter() < deadline:
                time.sleep(0.01)
                lane = worker.latest(time.perf_counter())
            self.assertTrue(lane.observed)
        finally:
            worker.close()

    def test_uint8_ssd_input_keeps_raw_rgb_values(self) -> None:
        detector = object.__new__(TFLiteVehicleDetector)
        detector._input = {"dtype": np.uint8, "quantization": (0.0078125, 128)}
        detector._input_h, detector._input_w = 1, 1
        # OpenCV frames are BGR. The detector must receive RGB 10,20,30 -- not
        # a saturated re-quantized value such as 255,255,255.
        bgr = np.array([[[30, 20, 10]]], dtype=np.uint8)
        tensor = detector._input_tensor(bgr)
        np.testing.assert_array_equal(tensor, np.array([[[[10, 20, 30]]]], dtype=np.uint8))

    def test_touch_buttons_select_all_three_views_and_quit(self) -> None:
        buttons = touch_buttons(640, 480)
        self.assertEqual([button.action for button in buttons], ["quit", "world", "camera", "split"])
        for button in buttons:
            x1, y1, x2, y2 = button.rect
            self.assertEqual(touch_action_at(buttons, (x1 + x2) // 2, (y1 + y2) // 2), button.action)
        self.assertIsNone(touch_action_at(buttons, 0, 0))

    def test_split_view_reduces_rendered_pixel_work(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lane = LaneEstimate(0.25, 0.75, 0.50, False, float("inf"))
        rendered = compose_view(frame, None, [], [], lane, Rates(), "split", "")
        self.assertEqual(rendered.shape[:2], (360, 960))

    def test_parser_accepts_20_fps_pi_preset(self) -> None:
        with patch.object(sys, "argv", ["visionfsd_pi.py", "--fps", "20"]):
            self.assertEqual(parse_args().fps, 20)

    def test_parser_defaults_to_25_fps_pi_preset(self) -> None:
        with patch.object(sys, "argv", ["visionfsd_pi.py"]):
            self.assertEqual(parse_args().fps, 25)

    def test_version_is_drawn_in_camera_and_world_huds(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lane = LaneEstimate(0.25, 0.75, 0.50, False, float("inf"))
        with patch("visionfsd_pi.cv2.putText", wraps=cv2.putText) as put_text:
            _camera_panel(frame, None, Rates(), "")
            _world_panel((640, 480), None, [], [], lane, Rates())
        labels = [str(call.args[1]) for call in put_text.call_args_list]
        self.assertTrue(any(RUNTIME_VERSION in label for label in labels))


if __name__ == "__main__":
    unittest.main()
