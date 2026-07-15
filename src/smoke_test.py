"""Small offline checks for VisionFSD geometry; does not require a webcam or model."""

import cv2
import numpy as np

from visionfsd import (
    DetectedObject,
    MotionEstimator,
    RoadGeometry,
    RoadGeometryTracker,
    TrackStabilizer,
    _StableTrack,
    build_ego_lane_triangle,
    draw_camera_view,
    draw_world_view,
    estimate_distance_m,
    estimate_monocular_distance_m,
    estimate_lanes,
    estimate_road_geometry,
    estimate_road_geometry_from_masks,
    focal_length_px,
    road_lateral_offset_m,
)


def main() -> None:
    focal = focal_length_px(1280, 70)
    assert 910 < focal < 920
    assert 16.0 < estimate_distance_m("car", 100, focal) < 17.0
    fused_range = estimate_monocular_distance_m("car", (540, 360, 740, 610), focal, 720)
    assert 1.0 < fused_range < 20.0
    tracker = MotionEstimator()
    tracker.update(1, (100.0, 200.0), 20.0, 1.0)
    filtered_distance, closing, velocity = tracker.update(1, (110.0, 205.0), 18.0, 2.0)
    # Heavy range EMA (0.72/0.28) with 0.45 m deadband → stays near 20 on small jump
    assert 18.5 < filtered_distance <= 20.0, filtered_distance
    assert 1.9 < closing < 2.1
    assert np.allclose(velocity, (10.0, 5.0), atol=1e-6)
    sample = DetectedObject(1, "car", 0.95, (240, 180, 400, 300), 10.0, 0.0, 2.0, (5.0, -1.0))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert draw_camera_view(frame, [sample], True).shape == frame.shape
    assert draw_world_view([sample], (640, 480), 20.0).shape == frame.shape
    assert isinstance(estimate_lanes(frame), list)
    lane_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(lane_frame, (90, 479), (290, 270), (255, 255, 255), 8)
    cv2.line(lane_frame, (550, 479), (350, 270), (255, 255, 255), 8)
    assert len(estimate_lanes(lane_frame)) >= 2
    curve_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    left_curve = np.array([[300, 719], [350, 650], [400, 590], [430, 535], [455, 485], [475, 445], [490, 410]], dtype=np.int32)
    right_curve = np.array([[980, 719], [900, 650], [830, 590], [770, 535], [720, 485], [680, 445], [650, 410]], dtype=np.int32)
    cv2.polylines(curve_frame, [left_curve, right_curve], False, (255, 255, 255), 12, cv2.LINE_AA)
    geometry = estimate_road_geometry(curve_frame)
    assert geometry.center_points is not None and geometry.topology == "curve-left"
    assert geometry.frame_width == 1280 and geometry.frame_height == 720
    lateral = road_lateral_offset_m(float(geometry.center_points[0, 0]),
                                    float(geometry.center_points[0, 1]), geometry)
    assert lateral is not None and abs(lateral) < 0.05
    import visionfsd as _vfsd
    _vfsd._GUIDE_EMA = None
    _vfsd._GUIDE_EMA_SIZE = None
    _vfsd._PATH_HOLD = None
    _vfsd._PATH_HOLD_MISS = 0
    _vfsd._ASPHALT_CACHE_MASK = None
    overlay = draw_camera_view(curve_frame, [], True, road_geometry=geometry)
    # Yellow ego-lane triangle should paint something into the frame.
    assert not np.array_equal(overlay, curve_frame)
    assert RoadGeometryTracker().update(curve_frame).center_points is not None
    false_crossroad = curve_frame.copy()
    cv2.line(false_crossroad, (0, 575), (1279, 575), (255, 255, 255), 10, cv2.LINE_AA)
    false_geometry = estimate_road_geometry(false_crossroad)
    assert false_geometry.intersection_y is None and false_geometry.topology != "intersection"
    stop_line = curve_frame.copy()
    cv2.line(stop_line, (395, 605), (850, 605), (255, 255, 255), 12, cv2.LINE_AA)
    candidate = estimate_road_geometry(stop_line)
    assert candidate.intersection_y is not None and candidate.topology != "intersection"
    road_tracker = RoadGeometryTracker()
    for _ in range(3):
        assert road_tracker.update(stop_line).topology != "intersection"
    assert road_tracker.update(stop_line).topology == "intersection"
    stabilizer = TrackStabilizer(max_missing_updates=2)
    low_conf = DetectedObject(5, "car", 0.40, (100, 100, 200, 200), 18.0, 0.0, 0.0, (0.0, 0.0))
    assert stabilizer.update([low_conf], 1.0) == []
    assert len(stabilizer.update([low_conf], 1.1)) == 1
    assert len(stabilizer.update([], 1.2)) == 1
    assert len(stabilizer.update([], 1.3)) == 1
    assert stabilizer.update([], 1.4) == []
    class_stabilizer = TrackStabilizer()
    assert class_stabilizer.update([sample], 2.0)[0].label == "car"
    truck_jitter = DetectedObject(1, "truck", 0.82, (242, 181, 402, 301), 10.1, 0.1, 1.8, (4.0, -1.0))
    stable_vehicle = class_stabilizer.update([truck_jitter], 2.1)[0]
    assert stable_vehicle.label == "car"
    predicted = class_stabilizer.update([], 2.2)[0]
    assert not predicted.observed and predicted.missed_updates == 1
    # Established vehicles coast for many detection cycles (presence hold).
    for i, moment in enumerate((2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2)):
        coasted = class_stabilizer.update([], moment)
        assert len(coasted) == 1, (i, moment, len(coasted))
        assert not coasted[0].observed
    # Eventually expire after the vehicle miss budget.
    for moment in (3.3 + 0.1 * k for k in range(12)):
        class_stabilizer.update([], moment)
    assert class_stabilizer.update([], 5.0) == []
    rebind_stabilizer = TrackStabilizer(max_missing_updates=3)
    rebind_stabilizer._tracks[1] = _StableTrack(
        DetectedObject(1, "car", 0.9, (400, 300, 600, 450), 12.0, 0.0, 0.0, (0.0, 0.0)), 3, 1, 1.0)
    fragmented = DetectedObject(99, "car", 0.9, (405, 305, 605, 455), 12.0, 0.0, 0.0, (0.0, 0.0))
    rebound = rebind_stabilizer.update([fragmented], 1.5)
    assert len(rebound) == 1 and rebound[0].track_id == 1
    assert 99 in rebind_stabilizer._tracks and 1 not in rebind_stabilizer._tracks

    # Two live candidates deliberately built to near-identical IOU (~0.40)
    # against the new observation, so the outcome isn't already decided by
    # the IOU gate alone: candidate 1 keeps the observation's own box shape
    # but is centred well away from it; candidate 2 sits almost exactly on
    # the observation's centre but is a much wider, shorter box (a
    # differently shaped real object, e.g. a bus/truck silhouette). On IOU
    # and centre match alone candidate 2 would win; the size term must
    # still prefer candidate 1.
    flip_stabilizer = TrackStabilizer(max_missing_updates=3)
    flip_stabilizer._tracks[1] = _StableTrack(
        DetectedObject(1, "car", 0.9, (491, 305, 691, 455), 12.0, 0.0, 0.0, (0.0, 0.0)), 3, 1, 1.0)
    flip_stabilizer._tracks[2] = _StableTrack(
        DetectedObject(2, "car", 0.9, (322, 335, 689, 425), 12.0, 0.0, 0.0, (0.0, 0.0)), 3, 1, 1.0)
    flip_observation = DetectedObject(88, "car", 0.9, (405, 305, 605, 455), 12.0, 0.0, 0.0, (0.0, 0.0))
    flip_resolved = flip_stabilizer.update([flip_observation], 1.5)
    flip_rebind = next(obj for obj in flip_resolved if obj.observed)
    assert flip_rebind.track_id == 1

    # A real object's box legitimately shrinks during partial occlusion and
    # grows back on emergence, often with lateral drift (overtaking). The
    # size penalty must not be strong enough to block that reconnect.
    occlusion_stabilizer = TrackStabilizer(max_missing_updates=3)
    occlusion_stabilizer._tracks[1] = _StableTrack(
        DetectedObject(1, "car", 0.9, (445, 333, 555, 418), 14.0, 0.0, 0.0, (0.0, 0.0)), 3, 1, 1.0)
    emerged = DetectedObject(55, "car", 0.9, (440, 300, 640, 450), 12.5, 0.0, 0.0, (0.0, 0.0))
    emerged_resolved = occlusion_stabilizer.update([emerged], 1.5)
    assert len(emerged_resolved) == 1 and emerged_resolved[0].track_id == 1

    drivable = np.zeros((720, 1280), dtype=np.uint8)
    cv2.fillPoly(drivable, [np.array([[210, 719], [530, 360], [750, 360], [1110, 719]], dtype=np.int32)], 1)
    inferred = estimate_road_geometry_from_masks(np.zeros_like(drivable, dtype=np.float32), drivable)
    assert inferred.center_points is not None and inferred.source == "yolopv2-drivable"
    # Ego corridor must stay single-lane-ish even when the drivable mask is wide.
    near_width = float(inferred.right_points[0, 0] - inferred.left_points[0, 0])
    assert near_width < 1280 * 0.42, near_width
    # Far reach should extend well above the old ~50% image cut.
    assert float(np.min(inferred.center_points[:, 1])) < 720 * 0.55

    # A curved laneless road (no paint at all): the curb-fitted corridor must
    # bend with the drivable mask, not stay glued straight under the camera.
    curved_drivable = np.zeros((720, 1280), dtype=np.uint8)
    for y in range(340, 720):
        drift = 0.42 * (719 - y)             # road centre drifts left with depth
        half = 95 + (y - 340) * 0.42
        centre = 640.0 - drift
        x1 = int(np.clip(centre - half, 0, 1279))
        x2 = int(np.clip(centre + half, 0, 1279))
        curved_drivable[y, x1:x2 + 1] = 1
    curved = estimate_road_geometry_from_masks(
        np.zeros_like(curved_drivable, dtype=np.float32), curved_drivable,
    )
    assert curved.center_points is not None and curved.source == "yolopv2-drivable"
    near_x = float(curved.center_points[0, 0])
    far_x = float(curved.center_points[-1, 0])
    assert abs(near_x - 640.0) < 1280 * 0.06, near_x       # anchored under camera
    assert far_x < near_x - 25.0, (near_x, far_x)          # bends left like the road

    # Guidance triangle: base on the ego path, tip banking with the fitted
    # road curvature (left curve -> tip points left of the base).
    import visionfsd as _vfsd2
    _vfsd2._GUIDE_EMA = None
    _vfsd2._GUIDE_EMA_SIZE = None
    triangle = build_ego_lane_triangle(geometry, 1280, 720)
    assert triangle is not None and triangle.shape == (3, 2)
    base_left, tip, base_right = triangle
    base_centre = 0.5 * (float(base_left[0]) + float(base_right[0]))
    assert float(tip[1]) < float(base_left[1]) - 40.0      # tip is the far vertex
    assert float(tip[0]) < base_centre, (float(tip[0]), base_centre)  # banks left

    # Duplicate stacked tracks (coasted + live) must collapse to one display object.
    nms = TrackStabilizer(max_missing_updates=4)
    live = DetectedObject(7, "car", 0.95, (400, 300, 560, 420), 18.0, 0.0, 0.0, (0.0, 0.0))
    ghost = DetectedObject(8, "car", 0.55, (405, 302, 555, 418), 18.4, 0.0, 0.0, (0.0, 0.0),
                           observed=False, missed_updates=2)
    nms._tracks[7] = _StableTrack(live, 4, 0, 3.0)
    nms._tracks[8] = _StableTrack(ghost, 3, 2, 2.8)
    collapsed = nms.update([], 3.1)
    car_tracks = [obj for obj in collapsed if obj.label == "car"]
    assert len(car_tracks) == 1, len(car_tracks)

    # --- Heading / max-car invariants (lightweight regression) -------------
    _test_heading_and_vehicle_cap()
    _test_environment_intel()
    print("Geometry smoke test passed.")


def _test_environment_intel() -> None:
    """Free-space / scene-mode / controlling-light (no neural net)."""
    from environment_intel import (
        FS_BLOCKED,
        FS_FREE,
        FS_UNKNOWN,
        compute_free_space,
        select_controlling_light,
        update_environment_intel,
    )

    # Free-space on a synthetic drivable corridor (full-width lower FOV).
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[360:, 400:880] = 1
    geom = RoadGeometry(
        None, None, np.array([[640, 700], [640, 400]], dtype=np.int32),
        0.8, "straight", None, 1280, 720, mask,
    )
    bands = compute_free_space(geom, [], distances_m=(15.0, 25.0))
    assert len(bands) == 2
    assert bands[0].center in {FS_FREE, FS_UNKNOWN, FS_BLOCKED}

    # Controlling light: nearer + smaller bearing wins.
    lights = [
        DetectedObject(1, "traffic light", 0.6, (900, 100, 940, 200), 40.0, 20.0, 0.0, (0.0, 0.0),
                       None, True, 0, False, "red", 0.5),
        DetectedObject(2, "traffic light", 0.8, (620, 120, 660, 220), 22.0, 3.0, 0.0, (0.0, 0.0),
                       None, True, 0, False, "green", 0.8),
    ]
    control = select_controlling_light(lights)
    assert control is not None and control.track_id == 2, control

    snap = update_environment_intel(lights, geom, oncoming_active=False)
    assert snap.summary.startswith("ENV ")
    assert snap.controlling_light is not None
    assert len(snap.free_space) >= 1


def _test_heading_and_vehicle_cap() -> None:
    """Regression: vehicle mesh yaw is binary; hard cap is 2 cars."""
    import math

    import visionfsd_3d as v3

    # Cap constant locked for the display budget.
    assert v3.RELEVANT_MAX_VEHICLES == 2, v3.RELEVANT_MAX_VEHICLES

    def _reset() -> None:
        v3.WORLD_ONCOMING_LOCK.clear()
        v3.WORLD_POSES.clear()
        v3.WORLD_TRAJECTORY.clear()
        v3.WORLD_TURN_HITS.clear()
        v3.WORLD_CRUISE_YAW.clear()

    def _make(tid: int, closing: float, lat: float, aw: int, ah: int,
              *, bearing: float = 2.0, stationary: bool = False) -> DetectedObject:
        box = (400, 300, 400 + aw, 300 + ah)
        return DetectedObject(
            tid, "car", 0.88, box, 22.0, bearing, closing, (0.0, -5.0),
            None, True, 0, stationary,
        )

    def _is_binary(heading: float) -> bool:
        wrapped = (float(heading) + 180.0) % 360.0 - 180.0
        return abs(wrapped) < 0.5 or abs(abs(wrapped) - 180.0) < 0.5

    # Same-dir / co-speed / merge-like: always 0° (no intermediate spin angles).
    _reset()
    for i in range(12):
        obj = _make(1, 0.3 + 0.2 * math.sin(i), 0.4, 100 + 20 * (i % 3), 72)
        h = v3._visual_vehicle_heading(obj, 20.0 * math.sin(i), False, 0.4, track_id=1)
        _, _, ph = v3._stable_world_pose(
            1, 0.4, 22.0, h, None, False, True, False, obj.closing_mps,
        )
        if 1 in v3.WORLD_POSES:
            lat, dist, hd, t = v3.WORLD_POSES[1]
            v3.WORLD_POSES[1] = (lat, dist, hd, t - 0.05)
        assert _is_binary(ph), ph
        assert abs(ph) < 0.5, ph

    # Merge-like side box + offset: still binary (0), not ±45/±90 spin.
    _reset()
    for i in range(10):
        obj = _make(2, 1.8, 2.4, 145, 68, bearing=16.0)
        h = v3._visual_vehicle_heading(obj, 8.0, False, 2.4, track_id=2)
        _, _, ph = v3._stable_world_pose(
            2, 2.4, 18.0, h, None, False, True, False, 1.8,
        )
        if 2 in v3.WORLD_POSES:
            lat, dist, hd, t = v3.WORLD_POSES[2]
            v3.WORLD_POSES[2] = (lat, dist, hd, t - 0.05)
        assert _is_binary(ph), ph

    # True oncoming opposite: latches to 180° and stays binary.
    _reset()
    last = 0.0
    for i in range(14):
        obj = _make(3, 8.0, 3.0, 90, 85, bearing=3.0)
        h = v3._visual_vehicle_heading(obj, 0.0, True, 3.0, track_id=3)
        _, _, ph = v3._stable_world_pose(
            3, 3.0, 32.0 - i, h, None, True, True, False, 8.0,
        )
        if 3 in v3.WORLD_POSES:
            lat, dist, hd, t = v3.WORLD_POSES[3]
            v3.WORLD_POSES[3] = (lat, dist, hd, t - 0.05)
        assert _is_binary(ph), ph
        last = ph
    assert abs(abs((last + 180.0) % 360.0 - 180.0) - 180.0) < 0.5 or abs(abs(last) - 180.0) < 0.5

    # Mesh filter hard-cap: at most 2 vehicles kept.
    crowd = [
        DetectedObject(
            10 + i, "car", 0.9,
            (300 + i * 40, 300, 380 + i * 40, 400),
            10.0 + i * 3.0, float(i - 2), 0.0, (0.0, 0.0),
            None, True, 0, False, None, 0.8,
        )
        for i in range(8)
    ]
    kept = v3.filter_relevant_traffic(crowd, None)
    vehicles = [o for o in kept if o.label in v3.RELEVANT_VEHICLE_LABELS]
    assert len(vehicles) <= v3.RELEVANT_MAX_VEHICLES, len(vehicles)
    assert len(vehicles) == 2, len(vehicles)

    # Lead picker: closest same-lane ahead; sticky against thrash.
    v3._LEAD_ID = -1
    v3._LEAD_HITS = 0
    v3._LEAD_CHALLENGER = -1
    v3._LEAD_CHALLENGE = 0
    v3._LEAD_DIST = 1e9
    v3.WORLD_ONCOMING_LOCK.clear()
    # Dead-ahead close car vs farther same-lane and a side car (wide bearing).
    same_close = DetectedObject(
        101, "car", 0.9, (580, 320, 700, 480), 12.0, 1.0, 0.5, (0.0, 0.0),
        None, True, 0, False, None, 0.8,
    )
    same_far = DetectedObject(
        102, "car", 0.95, (590, 280, 690, 400), 28.0, 0.5, 0.2, (0.0, 0.0),
        None, True, 0, False, None, 0.95,
    )
    side_car = DetectedObject(
        103, "car", 0.99, (200, 300, 320, 450), 10.0, -22.0, 0.0, (0.0, 0.0),
        None, True, 0, False, None, 0.99,
    )
    lead = v3.pick_lead_vehicle_id([same_far, same_close, side_car], None)
    assert lead == 101, lead  # closest same-lane, not side or far
    # Farther higher-quality car should not steal LEAD without margin+frames.
    for _ in range(5):
        assert v3.pick_lead_vehicle_id([same_far, same_close, side_car], None) == 101
    # Slightly closer challenger (< margin) still loses.
    almost = DetectedObject(
        104, "car", 0.9, (580, 330, 700, 490), 10.5, 0.8, 0.5, (0.0, 0.0),
        None, True, 0, False, None, 0.8,
    )
    for _ in range(8):
        assert v3.pick_lead_vehicle_id([same_close, almost], None) == 101
    # Clearly closer challenger for LEAD_SWAP_FRAMES frames takes over.
    much_closer = DetectedObject(
        105, "car", 0.9, (575, 360, 705, 520), 6.0, 0.5, 0.3, (0.0, 0.0),
        None, True, 0, False, None, 0.85,
    )
    for _ in range(v3.LEAD_SWAP_FRAMES - 1):
        assert v3.pick_lead_vehicle_id([same_close, much_closer], None) == 101
    assert v3.pick_lead_vehicle_id([same_close, much_closer], None) == 105

    # Presence memory reinjects a solid car after a brief full dropout.
    v3._VEHICLE_PRESENCE.clear()
    solid = DetectedObject(
        201, "car", 0.92, (500, 300, 640, 460), 14.0, 1.0, 0.2, (2.0, 0.0),
        None, True, 0, False, None, 0.9,
    )
    for _ in range(4):
        out = v3.merge_vehicle_presence([solid])
        assert any(o.track_id == 201 for o in out)
    # Full dropout: list empty → ghost reappears for a couple of seconds.
    ghosted = v3.merge_vehicle_presence([])
    assert len(ghosted) == 1 and ghosted[0].track_id == 201 and not ghosted[0].observed

    # Depth fusion: synthetic map should pull mono ranges without crashing.
    from depth_perception import fuse_objects_with_depth
    depth_map = np.linspace(1.0, 4.0, 720 * 1280, dtype=np.float32).reshape(720, 1280)
    mono_objs = [
        DetectedObject(301, "car", 0.9, (600, 360, 720, 520), 20.0, 0.0, 0.0, (0.0, 0.0),
                       None, True, 0, False, None, 0.85),
    ]
    fused_objs, scale = fuse_objects_with_depth(
        mono_objs, depth_map, focal_px=910.0, frame_height=720, depth_weight=0.4,
    )
    assert scale > 0.0 and len(fused_objs) == 1
    assert 1.0 <= fused_objs[0].distance_m <= 86.0

    # UFLD decode + geometry merge (synthetic logits, no model load).
    from lane_perception import decode_ufld_output, merge_ufld_into_geometry
    logits = np.full((1, 101, 56, 4), -4.0, dtype=np.float32)
    # Background class high except for lanes 1 and 2 near center columns.
    logits[0, 100, :, :] = 6.0
    for lane, col in ((1, 35), (2, 65)):
        for row in range(56):
            logits[0, :, row, lane] = -4.0
            logits[0, col, row, lane] = 8.0
            logits[0, 100, row, lane] = -8.0
    ufld = decode_ufld_output(logits, frame_width=1280, frame_height=720)
    assert ufld.left is not None and ufld.right is not None and ufld.center is not None
    geom = merge_ufld_into_geometry(None, ufld, frame_width=1280, frame_height=720)
    assert geom is not None and geom.source.startswith("ufld")
    assert geom.center_points is not None and len(geom.center_points) >= 4


if __name__ == "__main__":
    main()
