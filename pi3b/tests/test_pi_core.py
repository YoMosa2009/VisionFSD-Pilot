from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionfsd_pi import (
    Detection,
    TFLiteVehicleDetector,
    TargetSelector,
    bearing_deg,
    estimate_car_range_m,
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

    def test_target_requires_persistent_materially_closer_challenger(self) -> None:
        selector = TargetSelector(70.0)
        now = time.perf_counter()
        incumbent = Detection("car", 0.9, (280, 250, 360, 420))
        first = selector.update([incumbent], 640, now)
        self.assertIsNotNone(first)
        first_id = first.track_id
        # A closer, non-overlapping challenger needs six detector updates.
        challenger = Detection("car", 0.9, (40, 100, 260, 460))
        for step in range(5):
            current = selector.update([incumbent, challenger], 640, now + (step + 1) * 0.03)
            self.assertEqual(current.track_id, first_id)
        current = selector.update([incumbent, challenger], 640, now + 0.20)
        self.assertNotEqual(current.track_id, first_id)

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
