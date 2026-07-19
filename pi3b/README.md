# VisionFSD Pi 3B runtime

This is a **separate, read-only** Raspberry Pi 3B runtime derived from the
design of the desktop VisionFSD Pilot. It is deliberately not a direct port.

It preserves newest-frame capture, bounded asynchronous inference, a sticky
lane-aware lead target, and a target shown in both camera and world panels.
It intentionally removes OpenVINO GPU, neural road/lane/depth models,
ByteTrack, PyTorch, OpenGL, YouTube, and desktop-scale multi-object rendering.

Use **64-bit** Raspberry Pi OS (`aarch64`). Current LiteRT has an ARM64 wheel
for modern Pi OS/Python 3.13; the obsolete `tflite-runtime` package does not.

## Runtime contract

The included `models/vehicle_ssd_mobilenet_v1.tflite` is a 4.2 MB quantized
SSD MobileNet V1 COCO detector. It keeps only one sticky **lead vehicle**
(car, motorcycle, bus, or truck) in the camera view. Lead selection prefers a
vehicle inside the detected ego lane, and short-term label voting reduces
car/bus/truck classification flicker. The world view can also retain the
nearest confirmed vehicle in each adjacent lane, plus confirmed pedestrians,
traffic lights, and stop signs; those extras never clutter the camera view.
This immediately makes the Pi
folder runnable. The desktop repository's OpenVINO/PyTorch artifacts are not
Pi runtime artifacts. `tools/export_pi_tflite.py` remains available for a
future, Pi-specific YOLO model once a Linux x86/macOS export environment is
available.

The bundled COCO model identifies **stop signs**, not arbitrary traffic-sign
types such as speed-limit signs. Traffic-light colour is not inferred.

The one-command deployment shape is:

```bash
curl -fsSL https://raw.githubusercontent.com/YoMosa2009/VisionFSD-Pilot/main/pi3b/install.sh | bash
```

The installer downloads the same model from TensorFlow's storage and verifies
its SHA-256. A custom HTTPS model can be supplied only with its SHA-256.
It intentionally does not require the optional `libatlas-base-dev` package,
which is unavailable on some current Raspberry Pi OS package sources.

To update an existing installation, preserving the release branch it was
installed from:

```bash
~/visionfsd-pi/pi3b/update.sh
```

## Run from this checkout

```bash
cd pi3b
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python visionfsd_pi.py --camera 0 --model models/vehicle_ssd_mobilenet_v1.tflite
```

The bottom of every visual screen has large touchscreen controls: **Quit**,
**Screen 1** (world), **Screen 2** (camera), and **Screen 3** (split).
They also work with a regular mouse. `1`, `2`, `3` select world, camera,
split; `S` saves a screenshot; `Q`/`Esc` quits. The world panel is a low-cost
OpenCV pseudo-3D view with a centred ego vehicle and two lane boundaries.
The lanes glow bright white only when its low-rate, classical lane pass has a
fresh, geometrically valid left-and-right pair; otherwise they stay dim. The
same perspective lane geometry places vehicles into ego, left, or right lanes
in the world panel. It is not desktop OpenGL and it is not a driving
measurement.

## Performance

`--fps 25`/`30` pace the display. The HUD reports display and detection FPS
separately: a 30 FPS display is not claimed as 30 FPS inference. Acceptance
requires a sustained Pi benchmark with no thermal throttling. If CPU inference
does not sustain the goal after input/model tuning, add a USB accelerator.

The default two LiteRT threads and one OpenCV thread leave headroom for camera
capture and the desktop. All scene classes reuse the same single detector
result, while the lane pass is bounded to 256 px at about 5.5 Hz. Missed
display deadlines reset immediately instead of burst-rendering catch-up frames,
which reduces visible stutter. Do not claim 25 FPS
**detection** unless the HUD's `DETECT` rate reaches it on the physical Pi;
25 FPS display alone is expected.

## Desktop provenance

| Pi component | Desktop reference | Adaptation |
|---|---|---|
| `LatestCamera` | `src/webcam_capture_proc.py` | Linux V4L2, newest frame only |
| `AsyncDetector` | `src/object_perception.py` | LiteRT, one pending frame |
| `TargetSelector` | `src/visionfsd_3d.py` | small tracker, lane-aware lead plus adjacent-lane world vehicles |
| range/bearing | `src/visionfsd.py` | car-width pinhole estimate only |
| split display | `src/visionfsd_3d.py` | OpenCV pseudo-3D, no OpenGL |

## Validation

```bash
python3 -m unittest discover -s tests -v
./run.sh --camera 0 --test-seconds 60 --benchmark-report logs/benchmark.json
```

The synthetic tests cover lane-aware target selection, label stability,
range/bearing, lane extraction, and detector output decoding. The benchmark
records capture, display, detection, and latency.
