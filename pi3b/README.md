# VisionFSD Pi 3B runtime

This is a **separate, read-only** Raspberry Pi 3B runtime derived from the
design of the desktop VisionFSD Pilot. It is deliberately not a direct port.

It preserves newest-frame capture, bounded asynchronous inference, a sticky
single vehicle target, and a target shown in both camera and world panels.
It intentionally removes OpenVINO GPU, road/lane/depth models, ByteTrack,
PyTorch, OpenGL, YouTube, and multi-object rendering.

## Runtime contract

The included `models/vehicle_ssd_mobilenet_v1.tflite` is a 4.2 MB quantized
SSD MobileNet V1 COCO detector. The Pi renderer accepts only its `car` result,
then renders only one sticky target. This immediately makes the Pi folder
runnable. The desktop repository's OpenVINO/PyTorch artifacts are not Pi
runtime artifacts. `tools/export_pi_tflite.py` remains available for a future,
Pi-specific YOLO model once a Linux x86/macOS export environment is available.

The one-command deployment shape is:

```bash
curl -fsSL https://raw.githubusercontent.com/YoMosa2009/VisionFSD-Pilot/main/pi3b/install.sh | bash
```

The installer downloads the same model from TensorFlow's storage and verifies
its SHA-256. A custom HTTPS model can be supplied only with its SHA-256.

## Run from this checkout

```bash
cd pi3b
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python visionfsd_pi.py --camera 0 --model models/vehicle_ssd_mobilenet_v1.tflite
```

`1`, `2`, `3` select world, camera, split; `S` saves a screenshot; `Q`/`Esc`
quits. The world panel is a low-cost OpenCV pseudo-3D view, not desktop OpenGL.

## Performance

`--fps 25`/`30` pace the display. The HUD reports display and detection FPS
separately: a 30 FPS display is not claimed as 30 FPS inference. Acceptance
requires a sustained Pi benchmark with no thermal throttling. If CPU inference
does not sustain the goal after input/model tuning, add a USB accelerator.

## Desktop provenance

| Pi component | Desktop reference | Adaptation |
|---|---|---|
| `LatestCamera` | `src/webcam_capture_proc.py` | Linux V4L2, newest frame only |
| `AsyncDetector` | `src/object_perception.py` | LiteRT, one pending frame |
| `TargetSelector` | `src/visionfsd_3d.py` | small IoU tracker, one visible target |
| range/bearing | `src/visionfsd.py` | car-width pinhole estimate only |
| split display | `src/visionfsd_3d.py` | OpenCV pseudo-3D, no OpenGL |

## Validation

```bash
python3 -m unittest discover -s tests -v
./run.sh --camera 0 --test-seconds 60 --benchmark-report logs/benchmark.json
```

The synthetic tests cover target-lock stability, range/bearing, and detector
output decoding. The benchmark records capture, display, detection, and latency.
