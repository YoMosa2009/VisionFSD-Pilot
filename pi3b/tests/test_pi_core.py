from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionfsd_pi import (
    Detection,
    LaneEstimate,
    LowCostLaneDetector,
    SceneObjectTracker,
    TFLiteVehicleDetector,
    TargetSelector,
    bearing_deg,
    estimate_car_range_m,
    lane_position,
    nms,
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

    def test_target_requires_persistent_materially_closer_challenger(self) -> None:
        selector = TargetSelector(70.0)
        now = time.perf_counter()
        incumbent = Detection("car", 0.9, (280, 250, 360, 420))
        first = selector.update([incumbent], 640, now)
        self.assertIsNotNone(first)
        first_id = first.track_id
        # A closer challenger still needs three detector updates, preventing a
        # single noisy box from replacing a stable lead.
        challenger = Detection("car", 0.9, (210, 100, 430, 460))
        for step in range(2):
            current = selector.update([incumbent, challenger], 640, now + (step + 1) * 0.03)
            self.assertEqual(current.track_id, first_id)
        current = selector.update([incumbent, challenger], 640, now + 0.09)
        self.assertNotEqual(current.track_id, first_id)

    def test_target_switches_to_a_new_forward_corridor_vehicle(self) -> None:
        selector = TargetSelector(70.0)
        side_vehicle = Detection("car", 0.9, (0, 120, 240, 460))
        forward_vehicle = Detection("car", 0.9, (290, 160, 350, 400))
        first = selector.update([side_vehicle], 640, 1.0)
        self.assertIsNotNone(first)
        current = selector.update([side_vehicle, forward_vehicle], 640, 1.1)
        self.assertEqual(current.detection.box, forward_vehicle.box)

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
        target = selector.update([close_left, farther_ego], 640, 1.0, 480, lane)
        self.assertIsNotNone(target)
        self.assertEqual(target.detection.box, farther_ego.box)
        self.assertEqual(target.lane_slot, 0)

    def test_vehicle_label_voting_resists_one_frame_class_jitter(self) -> None:
        selector = TargetSelector(70.0)
        car = Detection("car", 0.84, (275, 160, 365, 420))
        for step in range(3):
            selector.update([car], 640, 1.0 + step * 0.05, 480)
        noisy_bus = Detection("bus", 0.93, car.box)
        target = selector.update([noisy_bus], 640, 1.2, 480)
        self.assertIsNotNone(target)
        self.assertEqual(target.detection.label, "car")

    def test_world_vehicle_list_keeps_lead_and_adjacent_lanes(self) -> None:
        selector = TargetSelector(70.0)
        lane = LaneEstimate(0.25, 0.75, 0.50, True, 0.0, 0.44, 0.56, 0.90)
        detections = [
            Detection("car", 0.90, (55, 180, 175, 430)),
            Detection("car", 0.92, (280, 180, 360, 430)),
            Detection("truck", 0.88, (465, 180, 585, 430)),
        ]
        selector.update(detections, 640, 1.0, 480, lane)
        selector.update(detections, 640, 1.1, 480, lane)
        vehicles = selector.vehicles(1.1)
        self.assertEqual({item.lane_slot for item in vehicles}, {-1, 0, 1})

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

    def test_world_objects_require_two_detections_before_display(self) -> None:
        tracker = SceneObjectTracker(70.0)
        person = Detection("person", 0.90, (280, 140, 330, 430))
        self.assertEqual(tracker.update([person], 640, 1.0), [])
        confirmed = tracker.update([person], 640, 1.2)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].detection.label, "person")

    def test_low_cost_lane_detector_requires_and_finds_a_pair(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame, (80, 479), (270, 250), (255, 255, 255), 8, cv2.LINE_AA)
        cv2.line(frame, (560, 479), (370, 250), (255, 255, 255), 8, cv2.LINE_AA)
        lane = LowCostLaneDetector().update(frame, 1.0)
        self.assertTrue(lane.observed)
        self.assertLess(lane.left_bottom_norm, 0.5)
        self.assertGreater(lane.right_bottom_norm, 0.5)

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


if __name__ == "__main__":
    unittest.main()
