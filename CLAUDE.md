# Perfect Punch — project context

## Vision
A **digital personal boxer**: a webcam-based training partner that calls out punches for the user to throw, throws attacks back that the user must defend against, and grades their technique, timing, and guard.

## Current focus
The two features being built right now:
1. **Prompted hits (offense):** the app calls a punch ("LEFT JAB"), shows a target, and the user hits it on the beat. A CNN checks that the right punch type was thrown.
2. **Dodges (defense):** the app throws attacks at the user's head. The user slips, ducks, steps aside, or blocks with their forearms.

Prioritise work that makes these two loops feel responsive, fair, and accurate. Everything else (more punch classes, combos, coaching, an app frontend) comes later.

## How to run
```powershell
.venv\Scripts\python.exe main_python_analysis.py
```
- Python 3.12 venv in `.venv` (gitignored). Key packages: `mediapipe==0.10.21` (legacy `mp.solutions` API), `opencv-python`, `torch` (CPU build), `numpy<2`.
- There is no `requirements.txt` yet; if you add one, pin `mediapipe` and `numpy` because the code relies on `mp.solutions.pose`.
- Needs a webcam. It opens a normal, resizable OpenCV window. Press `q` to quit. Set `FULLSCREEN=true` for borderless fullscreen (Windows-only `ctypes` calls), or `SHOW_DISPLAY=false` to run headless.
- A round lasts `MAX_RUNTIME` (30 s) after a 3 s countdown and a calibration check. When it ends, stats are written to `session_metrics.json` in the working directory.
- There are no automated tests. Check changes by playing a round.

## Web version (`web/`)
A static, browser-only port of the game (punch prompts + dodges), with no server-side code.
- Run: `cd web; ..\.venv\Scripts\python.exe -m http.server 8000`, then open http://localhost:8000. Webcam access needs `localhost` or HTTPS, so any static host works (GitHub Pages, Netlify, Vercel).
- `game.js`: the game (a port of the main script, `defense.py` and `target_utils.py`). MediaPipe `tasks-vision` and `onnxruntime-web` load from jsdelivr with pinned versions. Calibration, particles and session metrics were not ported. The only additions beyond the Python game are the loading/start panel. When a round ends, the totals go to the browser console and the camera stops, like the Python script.
- `pipeline.js`: port of `normalize_skeleton` / `extract_live_clip` / `mirror_skeleton` / `build_tensor`. **If the Python pipeline or the model changes**, re-run `.venv\Scripts\python.exe web\tools\export_model.py` (rebuilds `punch_model.onnx` and checks it against PyTorch), then `node web/tools/check_pipeline.mjs` (checks the JS pipeline against Python).
- The venv pins `onnx==1.17.0` because newer `onnx` needs protobuf 6 or newer, which breaks `mediapipe==0.10.21`.

## Files
| File | Role |
|---|---|
| `main_python_analysis.py` | Entry point and game loop: camera/model/pose init, calibration, offense flow, model inference, scoring, HUD, metrics output. Runs at module level, not in a `main()`. |
| `defense.py` | Defense mode. `DefenseGame` schedules `Attack`s (jab / hook / uppercut), draws them, and returns `"Dodge"`, `"Block"` or `"Hit"` through `check_collision`. Attack tunables are at the top of the file. |
| `target_utils.py` | Offense helpers and drawing: target placement (`respawn_target`), hit detection (`wrists_hit_circle`), punch type/side choice, callout banner, osu!-style approach rings, particles, floating text, HUD. |
| `extractDataPoints.py` | `PoseTracker`: rolling buffer of elbow and wrist landmarks. |
| `models/best_punch_model.pt` | State dict for `PerfectPunchCNN`, the one the game loads. `models/model_state.pt` is an older copy. |
| `test.py`, `MaxVel.py`, `VelSwitch.py`, `VelSwitch copy.py` | Early experiments (wrist velocity, peak finding, MediaPipe Tasks API with `pose_landmarker_lite.task`). They are not used by the game. |

## How the game loop works
**Offense** (`target_phase`: `idle` → `callout` → `active` → `idle`):
- `idle`: wait for `next_target_spawn_ts`, and never start a punch while an attack's active window is in progress.
- `callout`: show the banner for `CALLOUT_DURATION_S`.
- `active`: place a target relative to the body. An approach ring shrinks over `APPROACH_DURATION_S`, and the moment it reaches the target edge is the ideal hit time. A hit is graded PERFECT (±0.10 s), GOOD (±0.25 s) or MISS.
- On a hit, the world-landmark frames recorded since the target spawned are run through `extract_live_clip` (resample to 60 fps, centre on the velocity peak, 50 frames) and `build_tensor`. `PerfectPunchCNN` then classifies the clip as `jab`, `lead-hook` or `rear-uppercut`. `MODEL_TO_TARGET` maps those to the game's names `jab`, `hook` and `uppercut`. Below `MODEL_CONF_THRESHOLD` (0.55), a prediction is flagged as low confidence.

**Defense** (`DefenseGame.update`, called every frame):
- Each attack has a flashing `warning` phase (0.9 s), then an `active` phase (1.0 s) during which collisions count.
- Jab = a circle at the head. Hook = a horizontal band sweeping across (duck under it). Uppercut = a vertical band (step aside).
- Hit tests use a head box built from the nose and shoulder width. Forearms covering the head count as a block.
- The main loop passes `reserved_windows` so attacks and punch prompts don't overlap.

**Scoring:** PERFECT 300, GOOD 100, a dodge or block +50, getting hit −50 (the score never goes below 0).

## Things to keep in mind
- **Model and pipeline must match training.** `CLIP_FRAMES`, `TARGET_FPS`, `VEL_WINDOW`, `CLASS_NAMES`, the 7 input channels, and the architecture in `PerfectPunchCNN` all have to match how the checkpoint was trained. The training code is not in this repo.
- The model only knows the 3 punch classes above, and only the hand set by `STANCE` (orthodox: left jab/hook, right uppercut). With `ALLOW_RANDOM_SIDES = True`, either hand can be called. A punch thrown with the other hand is passed through `mirror_skeleton` (swap left/right joints, negate x) before inference, so the model sees the hand it was trained on.
- The hit test (`wrists_hit_circle(side=...)`) only counts the called hand. With random sides turned off, the off hand can never score.
- All detection is 2D. Leaning back from an uppercut can't be detected; only a side step or a guard counts (see the TODO in `defense.check_collision`).
- Colors are BGR (OpenCV). In the main loop, pose detection and all drawing and hit tests use the un-mirrored camera frame. The frame is flipped once for display, and text is placed at mirrored positions after that flip. MediaPipe `LEFT_*` landmarks mean the user's own left hand. (The calibration loop flips before running pose detection, but it only measures shoulder width.)
- `target_utils.PUBLIC_ASSET_DIR` points to `../public` (outside this repo) for glove/punch-pad images. Those images aren't present here, so the drawing falls back to shapes.
- Values marked `TODO(tune)` (target and attack sizes, hit radius) are known to need tuning through playtesting.
- Modules use `try: from .x import …` / `except ImportError: from x import …` so they work both as a package and as scripts. Keep that pattern.
- Large binaries (`.pt`, `.task`) are committed directly to git (no LFS).
