# Perfect Punch Web — Codebase Reference

A reference for AI agents working on the **browser version** of Perfect Punch (`web/`). It covers what each file does, how data moves through the game, which constants depend on each other, and known gaps. Read it before changing code.

---

## 1. What it is

A static, browser-only webcam boxing game. No server code: everything runs client-side from a static file host. A round lasts 30 s in solo (60 s in versus) and combines two loops:

1. **Offense (prompted hits):** the game calls a punch ("LEFT JAB") and shows an osu!-style target with a shrinking approach ring. The user hits the target with the called hand when the ring closes. Timing is graded PERFECT / GOOD / MISS, and a CNN (ONNX) classifies the punch type from the skeleton motion.
2. **Defense (dodges):** the game sends telegraphed orange attacks at the user's head: a jab circle, a hook band and an uppercut column. The user slips, ducks, steps aside or blocks with their forearms. Each attack ends as Dodge, Block or Hit.

**Versus** (`GAME_MODE === "versus"`) is a 1v1 fight. Each browser still runs its own camera and pose tracker. Only keypoints and small JSON events cross the network, over PeerJS data channels. A real punch from `PunchDetector` (either hand) becomes one head-circle attack the opponent must dodge. There is no osu target, callout, or punch-type distinction in versus. Solo is the default and is unchanged when `GAME_MODE` stays `"solo"`. The versus picture is the opponent, full screen; your own camera is a thumbnail.

The web game is a port of the repo's Python desktop game (`main_python_analysis.py`, `defense.py`, `target_utils.py`). The Python code is still the **source of truth for the ML pipeline and the model**. Gameplay constants are duplicated by hand.

---

## 2. Files

```
web/
├── index.html          Page shell, start/lobby panel; loads PeerJS then game.js
├── game.js             The game: init, main loop, offense, defense, drawing, HUD, versus hooks
├── net.js              PeerJS rooms, message protocol, clock sync, pose encode (no DOM)
├── versus.js           HP, damage, opponent buffer, HUD/panel/result card, ?bot=1 opponent
├── pipeline.js         Skeleton → model-input pipeline (numerically identical port of the Python)
├── punchDetector.js    Rule-based per-wrist punch detector (port of both_hands_tracking.py + guard relearning)
├── punch_model.onnx    Exported PerfectPunchCNN
├── CODEBASE.md         This reference
└── tools/
    ├── export_model.py     Rebuilds punch_model.onnx from models/best_punch_model.pt; writes fixture.json
    ├── check_pipeline.mjs  Asserts pipeline.js matches the Python pipeline (via fixture.json)
    └── check_net.mjs       Asserts clock sync, pose codec, interpolation, damage, and a fake-peer exchange
```

`tools/fixture.json` is generated and gitignored. There's no bundler, no `package.json`, and no test suite besides the two node checks.

### External dependencies (pinned, loaded from CDNs)
| What | Where |
|---|---|
| `@mediapipe/tasks-vision@0.10.14`: `FilesetResolver`, `PoseLandmarker`, `DrawingUtils` | `cdn.jsdelivr.net` (module + `/wasm`) |
| Pose model `pose_landmarker_lite.task` (float16) | `storage.googleapis.com/mediapipe-models/...` |
| `onnxruntime-web@1.20.1` (wasm backend; `ort.env.wasm.wasmPaths` points to the CDN) | `cdn.jsdelivr.net` |
| `peerjs@1.5.4` (`window.Peer`) | `cdn.jsdelivr.net`, classic script before the module |

Pin any new library to an exact version. Do not add a build step.

---

## 3. Running and verifying

```powershell
cd web; ..\.venv\Scripts\python.exe -m http.server 8000   # then open http://localhost:8000
```
Any static host works (GitHub Pages, Netlify, Vercel). `getUserMedia` needs **`localhost` or HTTPS**.

Open `http://localhost:8000/?bot=1` to fight the in-page bot (no second camera). Create room / Join room uses PeerJS's public signaling server.

There are no automated gameplay tests. Check solo by playing a round with the browser console open, since the end-of-round totals are printed there. Versus prints the same totals and also draws the result card; the camera stays on so a rematch can start.

```powershell
node web/tools/check_net.mjs          # protocol, clock, damage, bot. No browser, no dependencies.
node web/tools/check_pipeline.mjs     # needs web/tools/fixture.json from export_model.py
```

**If the Python pipeline or the model changes**, run from the repo root:
```powershell
.venv\Scripts\python.exe web\tools\export_model.py   # re-export ONNX (opset 17) and assert ONNX logits == PyTorch (atol 1e-4)
node web/tools/check_pipeline.mjs                    # assert pipeline.js tensors == Python tensors (max diff < 1e-4), plain + mirrored
```
- `export_model.py` runs `exec` on `main_python_analysis.py` **only up to the comment `# ── Camera backend helpers`**, which gives it the constants, pipeline functions and the `PerfectPunchCNN` class with no camera side effects. Don't move or rename that marker.
- It builds a synthetic 90-frame, 30 fps world-landmark sequence with one fast arm swing and saves the raw frames, plus the plain and mirrored tensors and logits, to `fixture.json`.
- The venv pins `onnx==1.17.0`, because newer `onnx` needs protobuf ≥ 6, which breaks `mediapipe==0.10.21`.
- Do **not** change `pipeline.js`, `punchDetector.js`, or `punch_model.onnx` for a versus change. Re-export only when the model or the Python pipeline changes.

**TURN.** `net.js` ships with STUN only (`stun.l.google.com:19302` and `stun.global.stun.twilio.com:3478`). Strict NATs often fail to open a data channel without a relay. Append `{urls, username, credential}` objects to the `TURN_SERVERS` array at the top of `net.js`. Nothing in the repo contains TURN credentials.

---

## 4. Conventions (read before touching geometry)

- **Two landmark sets from `detectForVideo`:**
  - `res.landmarks[0]` (**image** landmarks): normalized x, y in [0,1], z, visibility. Used for all on-screen geometry (targets, hit tests, attacks, head box), for `punchDetector.js`, and for the pose packets in versus. **World landmarks are never sent.**
  - `res.worldLandmarks[0]` (**world** landmarks, metres, origin at the hips): used **only** for the CNN, after `normalizeSkeleton`.
- **Left/right:** `"L"`/`"R"` and landmark names mean the user's own hand (MediaPipe convention). Indices: 0 nose; 11/12 shoulders; 13/14 elbows; 15/16 wrists; 17/18 pinky; 19/20 index; 21/22 thumb; 23/24 hips.
- **Mirroring:** all game logic and shape drawing use **camera (un-mirrored) pixel coords**. In `loop()` the canvas is drawn under `ctx.setTransform(-1,0,0,1,W,0)` (video, skeleton, detector peaks, target and attacks), then reset. **Text** is drawn afterwards at `mx(x) = W - 1 - x` so it isn't mirrored. A new shape goes inside the mirrored block; new text goes after it. The opponent panel is the exception: it is drawn **after** the mirror reset, in the sender's raw camera x, so their right hand appears on the screen-left of the panel.
- **Scale:** pixel constants were tuned for a 960×540 camera. `S = W / 960` multiplies `TARGET_RADIUS_PX`, `ARM_PADDING_PX`, `MIN_SHOULDER_PX`, font sizes and so on. `toPx(lm)` gives integer camera pixels.
- **Colors** are `[r,g,b]` arrays (the Python uses BGR). `rgba(c, a, k)` builds the CSS string, where `k` darkens. Jab red, hook green, uppercut blue, attacks orange, PERFECT gold. The opponent skeleton is neon cyan.
- **Time** is seconds: `performance.now() / 1000`. All windows are `[start, end]` arrays in those units. Versus deadlines that must agree across machines use `net.netNow()` (local seconds + `clockOffset`).
- **Shapes** for collision: `["circle", cx, cy, r]` or `["rect", x1, y1, x2, y2]`. Rects are `[x1, y1, x2, y2]`.

---

## 5. `game.js`

### 5.1 Constants
Gameplay values match `main_python_analysis.py` / `defense.py`. Keep both in sync when tuning.

| Constant | Value | Meaning |
|---|---|---|
| `STANCE` | `"orthodox"` | Lead hand is left |
| `ALLOW_RANDOM_SIDES` | `true` | Random L/R per callout; off-hand clips are mirrored before inference |
| `CALLOUT_DURATION_S` | 0.4 | Banner time before the target appears |
| `APPROACH_DURATION_S` | 1.0 | Ring shrink time; ideal hit = spawn + 1.0 s |
| `PERFECT_WINDOW_S` / `GOOD_WINDOW_S` | 0.10 / 0.25 | ± around the ideal hit |
| `TARGET_RADIUS_PX` | 25 (×S) | Logical target radius |
| `TARGET_VISUAL_SCALE` | 2.5 | Drawn radius |
| `HIT_RADIUS_SCALE` | 3.0 | Collision radius = 75 px × S (needs tuning) |
| `MAX_SAME_TARGET_IN_A_ROW` | 2 | Anti-repeat for type+side |
| `SCORE_PERFECT/GOOD/DEFENSE_BONUS/HIT_PENALTY` | 300 / 100 / 50 / 50 | Score is floored at 0 |
| `TARGET_RESPAWN_DELAY_S` | 2.0 | Gap after a target resolves. Solo only; versus has no targets |
| `SCHEDULE_MARGIN_S` / `RESERVED_MARGIN_S` | 0.1 | Spacing between punch moments and attack windows |
| `MAX_RUNTIME` / `PRESTART_COUNTDOWN_S` | 30 / 3 | Solo round length and countdown. Versus length is `VERSUS_ROUND_S` (60) |
| `MODEL_CONF_THRESHOLD` | 0.55 | Below this the prediction is shown as "(low)" |
| `WORLD_BUFFER_FRAMES` | 150 | World-skeleton buffer length |
| `LOOK_BACK_SECONDS` | 2.5 | Fallback clip window |
| **Web-only** | | |
| `SEGMENT_BUFFER_FRAMES` | 3 | Frames added on each side of a detected punch segment |
| `PUNCH_RESOLVE_GRACE_S` | 0.35 | Wait after the window closes for a punch still being thrown |
| `PENDING_CLASSIFY_S` | 0.6 | After an overlap hit, wait this long for the punch segment to finish |
| `SHOW_PUNCH_DEBUG` | `true` | Per-hand detector state text + peak circles |
| **Attacks** | | |
| `ATTACK_WARNING_S` / `ATTACK_ACTIVE_S` | 0.9 / 1.0 | Flashing warning, then the solid dodge window |
| `ATTACK_FLASH_COUNT`, `ATTACK_WARNING_ALPHA`, `ATTACK_SWEEP_FRACTION` | 3, 0.35, 0.5 | Band fill crosses the screen in the first half of active |
| `SPAWN_DELAY_RANGE`, `FIRST_ATTACK_DELAY_S` | [3, 7], 2.0 | Seconds between attacks / before the first attack. Solo only |
| `ATTACK_WEIGHTS` | jab .40, hook .35, uppercut .25 | Solo only. Versus always spawns the jab circle |
| `JAB_RADIUS_SW`, `HOOK_BAND_HEIGHT_SW`, `UPPERCUT_BAND_WIDTH_SW` | 0.45, 0.4, 0.8 | In shoulder widths |
| `HEAD_HALF_WIDTH_SW`, `HEAD_TOP_SW`, `HEAD_BOTTOM_SW` | 0.15, 0.35, 0.15 | Head box relative to the nose |
| `GAME_MODE` | `"solo"` | `"solo"` or `"versus"`. Every versus branch checks this |

`HAND_CONTACT = { L: [15,19,21,17], R: [16,20,22,18] }` (wrist, index, thumb, pinky) are the landmarks that can hit a target.

### 5.2 Lifecycle
`mode`: `loading → ready → countdown → playing → over`.

1. `init()`: `getUserMedia({960×540, 30 fps})` → sets `W`, `H`, `S` from the real video size → in parallel, creates the `PoseLandmarker` (VIDEO mode, 1 pose, GPU delegate with a CPU fallback) and the ONNX `InferenceSession("punch_model.onnx", wasm)` → shows the lobby (Solo / Create room / Join room) → `loop()`. Errors map to friendly messages (`NotAllowedError`, `NotFoundError`). Room buttons stay disabled until `mode === "ready"`. `?bot=1` skips the lobby and opens a versus room against `BotOpponent`.
2. Solo button → `countdown` (3 s) → `newRound(now)` → `playing`. This is the old Start button.
3. `endRound()` once `MAX_RUNTIME` is reached in solo: `console.log`s the score, timing and defense totals, stops the camera tracks, clears the canvas. Nothing is persisted. In versus the same log runs, the camera stays on, and the result card is drawn (see §11).

### 5.3 `loop()` (per new camera frame)
- `requestAnimationFrame`. Repeated frames are skipped via `video.currentTime === lastVideoTime`. Solo returns immediately once `mode === "over"`. Versus keeps looping so the result card and a rematch can paint.
- Run `landmarker.detectForVideo(video, performance.now())`.
- **Always, even outside a round:** push `{t, skel: normalizeSkeleton(worldLandmarks)}` to `worldBuf` (max 150) and call `detector.update(now, lms)` → `punchEvents`. That way the guard and shoulder scale are already learned when play starts.
- **Versus, after `detector.update`:** `versus.tick` (heartbeat, pose-loss, buffer prune) and, when a pose is present, `net.sendPose(netNow(), lms)` with **image** landmarks. This also runs in the lobby and during the countdown.
- Mirrored block: video, skeleton (points with visibility < 0.5 or off-frame are hidden), detector peak circles, and `stepRound` if playing.
- Unmirrored block: detector state text, the countdown number, `drawRoundText` (badge, callout, floating texts, vignette, prediction box, HUD). Versus then draws the opponent panel and, when the round is over, the result card.

### 5.4 Round state `st` (from `newRound`)
`start, score, phase ("idle"|"callout"|"active"), type, side, center, spawnTs, idealTs, calloutTs, nextSpawn, handInsidePrev, recent, reserved, texts, vignetteUntil, prediction, pending, timing {perfect, good, miss}`.

In versus, `st.start` is the local equivalent of the host's `t0` (`t0 - clockOffset`), not "whatever `performance.now()` was when the frame noticed the countdown hit zero".

### 5.5 Offense: `stepRound(lms, now, events)`
1. Prune expired `st.reserved` windows.
2. **idle → callout** when `now ≥ nextSpawn` and a pose is present, **unless** `[now, calloutEnd]` or the planned hit window `[plannedHit ± 0.25]` overlaps `defense.activeWindow()` (± 0.1). It picks `pickNextTarget()` (type from `choosePunchType`: jab 62.5 % / hook 25 % / uppercut 12.5 %, side random) and reserves both windows.
3. **callout → active** after 0.4 s. `respawnTarget` picks a random point on the called hand's half of the body. x spans the shoulders ± one shoulder width; y runs from nose/shoulder height down to mid-chest; it stays away from the frame edge by the visual radius and tries to stay `radius·HIT_RADIUS_SCALE + 20·S` away from both wrists (20 tries, otherwise the farthest candidate). It sets `spawnTs`, `idealTs = now + 1.0`, `handInsidePrev = true` (a hand already resting there doesn't count) and reserves the real hit window.
4. **active, hit test.** A hit is either of:
   - **`overlapHit`**: the called hand's `HAND_CONTACT` points (visibility ≥ 0.35) **enter** the circle (rising edge) while `now ∈ [ideal − 0.25, ideal + 0.25]`. It's graded at `now`. Classification is deferred: `st.pending = {type, side, spawnTs, deadline: now + 0.6}`.
   - **`punchHit`**: a detector `punch` event for the called side (with `peakT ≥ spawnTs`), `|peakT − ideal| ≤ 0.25`, and its `peakPts` inside the circle. It's graded at `peakT` and classified right away from its segment.
   - Grade: PERFECT if `|gradeTs − ideal| ≤ 0.10`, else GOOD.
   - **MISS**: a called-hand punch event that isn't a hit, **or** the window has closed and the called hand isn't still `throwing` (or the 0.35 s grace has run out).
   - Off-hand punches and punches with no active target are ignored.
   - Either result → `idle`, `nextSpawn = now + 2.0`.
   - **Versus does not run this loop.** `stepVersus` listens for a `punch` event from either hand and sends one attack, with `PUNCH_ATTACK_COOLDOWN_S` (0.6 s) so one swing cannot send two.
5. `resolvePending` (each frame while `st.pending` is set): classify on the first matching `punch` event via `segmentClip`; otherwise, after the deadline, via `snapshotClip`.
6. Defense update (see 5.7), then the osu! target is drawn on top.

### 5.6 Classification
- `bufferFps()`: the real fps measured from the `worldBuf` timestamps (falls back to 30).
- `segmentClip(startT, endT)`: takes the buffer frames between the punch start and end ± `SEGMENT_BUFFER_FRAMES`, plus enough earlier lead-in frames (`ceil(49/60 · fps)`) so that `extractLiveClip`'s peak search (which only looks after its first 50 resampled frames) covers the segment.
- `snapshotClip(spawnTs)`: the fallback. Frames since the target spawned, or the last 2.5 s if there are too few.
- `requestPrediction` builds the clip **synchronously**, because `worldBuf` keeps changing. Only `classify` is async.
- `classify(type, side, clip)`: `null` clip → "Warming up (n/needed frames)". If `side !== choosePunchSide(type, false)` (not the stance-default hand: orthodox **left** jab, **left** hook, **right** uppercut), `mirrorSkeleton(clip)` is applied, because the model only learned one hand per class. Then `session.run({input: Tensor([1,7,50,4,3])})` → `out.logits` → softmax → the text "Predicted: Lead-Hook (82%)", shown for 1.8 s. Green means confident; orange means below 0.55.
- **The prediction is display-only in solo.** Versus does not classify punches for attacks. `MODEL_TO_TARGET` is defined but not used yet.

### 5.7 Defense (`defense` object, port of `DefenseGame`)
- `update(lms, now, reserved)`: in **solo**, the first spawn comes 2 s into the round. It spawns when the timer is due and the would-be active window `[now+0.9, now+1.9]` doesn't overlap any `reserved` window (± 0.1). Otherwise it retries next frame. During `active` it runs `checkCollision`. If the active phase ends untouched → `"Dodge"`. It records `lastEvent = {outcome, pos: anchor, meta}`, schedules the next spawn in 3–7 s, and draws.
- **Versus** skips `SPAWN_DELAY_RANGE`, `FIRST_ATTACK_DELAY_S`, and the reserved-window test (there is no punch window to protect). `defense.enqueue(meta)` pushes onto `queue` (max 2). A full queue sends `attackResult {outcome:"dropped", dmg:0}`. The next attack is always the jab circle on the local nose. `"INCOMING"` shows during the warning, and `"ATTACK QUEUED"` while another waits. The circle is not drawn inside `update`; the versus frame maps it onto the screen.
- `spawnAttack`: a jab is a circle on the nose (clamped on-screen); a hook is a full-width band at nose height with a random sweep direction; an uppercut is a full-height column at the body centre, sweeping bottom → top.
- `Attack`: `phase(now)` returns `"warning"`, `"active"`, or `"done"` (there is no `"over"` state). `sweptRect(now)` is the part of the band covered so far, `dangerShape(now)` is the whole circle for a jab or the swept rect for bands (`null` outside active), and `draw(now)`.
- `checkCollision`: `null` if the danger shape misses the head box (`headRect`). **Block** if a forearm box (elbow→wrist, padded 20·S) overlaps both the padded head zone and the danger shape. Otherwise **Hit**. Everything is 2D, so leaning back from an uppercut can't be detected.
- Back in `stepRound`, solo: Hit → −50 (floored at 0), red vignette, "HIT!". Dodge or Block → +50, "DODGED!" / "BLOCKED!".
- Versus outcomes are applied by `applyVersusDefense`: Dodge → 0 damage and +50. Block → half of `ATTACK_DAMAGE` (5) and +25. Hit → `ATTACK_DAMAGE` (10), −50 (floored at 0), vignette, "HIT!". Then `attackResult` and `state` are sent. HP 0 sends `ko` and ends the round. Collision still uses the local camera landmarks and the circle in camera pixels.
- **Scheduling contract (solo):** offense won't schedule across an attack's active window, and defense won't start an active window across a reserved punch window. Versus has no target windows, so it does not consult `reserved`.

### 5.8 Drawing helpers
`fillCircle`, `strokeCircle`, `fillRect`, `fitFont` (sizes `ctx.font` to fit a box), `textCentered` (outlined text), `drawOsuTarget` (glow, body, ring shrinking from 3r to r; the edge pulses after the ideal moment), `drawCallout` (full-width banner that scales in and fades out), `pill`, `drawHud` (solo: time top-left, red at ≤ 5 s; score top-right), `drawVignette`, `drawPunchPeaks` / `drawPunchState` (debug).

Versus HUD, the opponent panel and the result card live in `versus.js` and are called from the unmirrored part of `loop`. Floating texts with `screen: true` (the "LANDED -12" / "DODGED" / "BLOCKED" line by the opponent panel) are not passed through `mx`.

### 5.9 Versus hook points
Each is marked `// VERSUS:` in `game.js`.

| Hook | What it does |
|---|---|
| `GAME_MODE` | Gates every versus branch. Default `"solo"`. |
| Lobby | Solo / Create room / Join room, shown when `mode === "ready"`. |
| `attachVersus` | Builds `Versus` on a `Net` or a `BotOpponent` and listens for the protocol events. |
| `loop` tick + `sendPose` | After `detector.update`, including lobby and countdown. Image landmarks only. |
| Countdown | `ceil(t0 - netNow())`, then `newRound(t0 - clockOffset)`. |
| `stepVersus` | Punch events only. Cooldown, then one attack. No callout or target. |
| `drawVersusFrame` | Opponent full screen, dodge circle, local camera thumbnail. |
| `applyVersusDefense` | Fixed damage, block score, `attackResult`, KO. |
| `spawnAttack("jab")` / queue | Random timer off. Circle only. |
| HUD / banners | HP bars, or the solo `drawHud`. |
| `endRound` | Versus returns before stopping the camera. |
| Result card clicks | Rematch and Leave, hit-tested in canvas pixels. |

---

## 6. `pipeline.js` (must stay numerically identical to the Python)

A skeleton frame is an array of 33 `[x, y, z, visibility]`.

- `normalizeSkeleton(worldLms)`: subtracts the hip centre `(lm23+lm24)/2` from xyz and divides by the shoulder distance `|lm11 − lm12|` (when > 1e-6).
- `extractLiveClip(rawFrames, srcFps)` → 50 frames or `null`:
  1. Resample linearly to 60 fps (`round(n/srcFps·60)` frames; skipped if `|srcFps − 60| < 0.5`). This matches `np.interp` over `linspace`.
  2. `null` if fewer than 50 frames.
  3. Velocity per frame = RMS over joints 11–16 of the xyz displacement, smoothed with a centred 7-wide box filter that divides by 7 at the edges too (`np.convolve(..., 'same')`).
  4. Peak = argmax over indices ≥ 49 (the first maximum wins). The clip is the 50 frames ending at the peak, inclusive.
- `mirrorSkeleton(clip)`: swaps the 16 L/R joint pairs and negates x.
- `buildTensor(clip)` → `Float32Array(7·T·4·3)`, laid out like the Python `(7, T, 4, 3)` tensor: index `((ch·T + t)·4 + j)·3 + k`. The joints are `j` = [L elbow, R elbow, L wrist, R wrist]:
  - ch0 position · ch1 velocity (t=0 is zero) · ch2 bone unit vector (shoulder→elbow, elbow→wrist), with **component x of the elbows overwritten by the elbow-angle cosine** · ch3 y-velocity broadcast · ch4 that side's elbow-angle cosine broadcast · ch5 `vx/√(vx²+vz²)` · ch6 `vy/|v|` (eps 1e-8).
- Model constants **fixed by training**: `CLIP_FRAMES = 50`, `TARGET_FPS = 60`, `VEL_WINDOW = 7`, `CLASS_NAMES = ["jab","lead-hook","rear-uppercut"]` (logit order), 7 channels. Changing any of them needs retraining (the training code is not in this repo) plus a re-export.

---

## 7. `punchDetector.js`

A rule-based detector with no DOM, so it runs under node. It is fed **image** landmarks every frame.

- `PunchDetector({L:{wrist, shoulder, contact}, R:{…}})`. `update(t, lms)` returns that frame's events. The shoulder-width scale is the median of the last 150 samples and is only learned while both hands are idle.
- Per-hand `PunchTracker`:
  - `wristRel = (wrist − shoulder) / scale`; `extension = |wristRel − guardPos|`; `speed` = norm of the least-squares slope over the last 5 samples (shoulder widths/s).
  - The **guard** is the per-axis median of the last ≤ 150 samples taken while idle, `speed < 3` and `ext < 0.4`. It's ready after **30** samples (`GUARD_READY_SAMPLES`); until then the debug overlay shows "learning guard n/30".
  - **Guard relearn:** if the hand rests (`speed < 3`) away from the guard for **1 s**, the guard is wiped and relearned from the new position. This is web-only.
  - States: `idle → throwing` (`ext > 0.8 && speed > 7 && guardReady`) → `recovering` (`ext < 0.9·maxExt && speed < 0.75·maxSpeed` gives a **punch** event; a throw older than 1 s gives a **dropped** event) → `idle` (`ext < 0.6` and 0.1 s after the peak, or a 2 s timeout).
- Events:
  - `{type:"start", side, t}`
  - `{type:"punch", side, startT, peakT, endT, peakPts, peakPos}`: `peakPts` holds the contact landmarks `{x, y, visibility}` at maximum extension.
  - `{type:"dropped", side, t}`
- `game.js` reads `detector.trackers[side].state`, `.guardReady`, `.guardBuf`, `.lastExt`, `.lastSpeed` and `.punchPeaks` (pruned by the debug overlay).

---

## 8. Differences from the Python desktop game

| | Web | Python |
|---|---|---|
| Hit signal | Overlap rising edge **or** a detector punch peaking on the target | Overlap rising edge only |
| Clip selection | Detected punch segment (fallback: since spawn / 2.5 s) | Frames since spawn (fallback: 2.5 s) |
| FPS for resampling | Measured from timestamps | `cap.get(CAP_PROP_FPS)`, fixed |
| Calibration (distance check) | None | Shoulder span 15–30 % of width for 2 s |
| Accuracy / reaction / speed stats, `session_metrics.json` | None (console totals only; versus also draws a card) | Yes |
| Particles, hit pops, miss fade-out | None | Yes |
| Glove art | None | Fallback shapes only (assets missing) |
| Multiplayer | Versus over PeerJS (web only) | None |

---

## 9. Known gaps and good next steps

- Prediction correctness isn't scored or tallied. Comparing `MODEL_TO_TARGET[name]` with `type` and the confidence threshold, then adding accuracy counters, would bring solo level with Python. Versus does not use the classifier.
- No session metrics file. Solo still only prints totals. Versus draws a card but doesn't persist it.
- No calibration step, so a user who is too close or too far gets bad targets and attacks.
- `SHOW_PUNCH_DEBUG` is on by default.
- `HIT_RADIUS_SCALE` and the attack sizes still need playtesting.
- 2D only: leaning back from an uppercut can't be detected.
- With `ALLOW_RANDOM_SIDES = false`, the off hand can never score.
- Versus has no TURN server configured. Strict NATs will fail until `TURN_SERVERS` is filled in.
- The opponent pose is 2D image landmarks. It is a preview, not a hit target.

---

## 10. Change guide

| You want to… | Edit | Also |
|---|---|---|
| Tune timing / scoring / sizes | Constants at the top of `game.js` | Mirror them in the Python files if the two versions should stay matched |
| Tune versus damage, HP, round length, pose rate | Constants at the top of `versus.js` | `node web/tools/check_net.mjs` |
| Change hit logic | `stepRound` step 3, `handHitsCircle` / `pointsHitCircle` | Check the interaction with `st.pending` and the miss/grace rules |
| Tune punch detection | Thresholds in `punchDetector.js` | Check the debug overlay (`ext`, `spd`, state) |
| Add an attack type | `ATTACK_WEIGHTS`, `spawnAttack`, `Attack.sweptRect` / `dangerShape` / `draw`, `checkCollision`, and `DAMAGE` | Both clients must agree on `DAMAGE` or the attack is ignored |
| Change features or the model | Python `build_tensor` / `extract_live_clip` + retrain, then port to `pipeline.js` | `export_model.py` → `check_pipeline.mjs` must both pass |
| Add a punch class | Retrain, then `CLASS_NAMES` in `pipeline.js`, `MODEL_TO_TARGET`, `choosePunchType`, `PUNCH_COLORS`, `LABELS`, `choosePunchSide`, `DAMAGE` | Re-export ONNX |
| Add a dependency | Import from a CDN with an exact pinned version | Keep the app static (no build step) |
| Add a network message | `Net.handle` in `net.js` and the matching listener in `Versus` | Bump `PROTOCOL_VERSION` if old clients must disconnect |

---

## 11. Versus (`net.js`, `versus.js`)

### 11.1 Roles and links
- The host creates `new Peer("pp-" + code, {config})`. `code` is 5 characters from `ABCDEFGHJKLMNPQRSTUVWXYZ23456789` (no I, O, 0, 1). `unavailable-id` retries up to 5 times.
- The guest creates an anonymous `Peer` and opens **two** connections to `pp-` + code:
  - `events`: `{reliable: true, serialization: "json", metadata: {channel: "events"}}`
  - `pose`: `{reliable: false, serialization: "json", metadata: {channel: "pose"}}` so a lost frame cannot block the next one.
- A third peer, once a guest id is set, is sent `{t:"full", v:1}` and closed. The guest shows "Room is full." A missing id (`peer-unavailable`) or a bad code shows "Room not found."
- `hello.v` must equal `PROTOCOL_VERSION` (1). A mismatch shows "Update required" and closes.
- ICE: the two STUN servers above, then whatever is in `TURN_SERVERS`. See the TURN note in §3.
- `Net` is an `EventTarget`. `new Net({PeerCtor, name, W, H})` takes the constructor so node tests can pass a fake. It does not touch `window` or the DOM. `netNow()` is `performance.now()/1000 + clockOffset`. The host's offset is always 0.

### 11.2 Clock sync
On link-up, each side runs 8 ping/pong exchanges 100 ms apart, then 3 more every 5 s. The guest is the only side that adopts an offset; the host uses the same pings so its HUD can show a round-trip too.

For guest send time `c0`, host receive `s1`, host send `s2`, guest receive `c3` (all seconds):

- `rtt = (c3 − c0) − (s2 − s1)`
- `offset = ((s1 − c0) + (s2 − c3)) / 2`

The sample with the smallest rtt wins. A stored best older than 30 s may be replaced by a slower sample. Each completed sample dispatches `latency` with `{rtt}` in **milliseconds**. The HUD shows `ping NN ms`.

### 11.3 Message protocol
Every events-channel message is JSON `{t, v: 1, ...}` where `t` is the type. Unknown `t` and malformed JSON are ignored. `handle` is the single switch for that channel.

The pose channel cannot use `t` for both the type and the sample time. Its packets are `{v: 1, t: <netNow seconds>, k: number[45]}`. The channel is the type. `k` is image-landmark `x, y, visibility` for indices `[0,11,12,13,14,15,16,17,18,19,20,21,22,23,24]`, each rounded to 3 decimals. A missing pose is not sent. Sends are capped at `POSE_HZ` (20).

| `t` | Channel | Fields | Who |
|---|---|---|---|
| `hello` | events | `name, v, W, H` | both, after the events channel opens |
| `ping` | events | `id, c0` | either side |
| `pong` | events | `id, c0, s1, s2` | the receiver of a ping |
| `ready` | events | `ready` | both |
| `start` | events | `t0` | host only. `t0 = netNow() + 3.5` (3 s countdown after a 0.5 s margin) |
| `attack` | events | `id, sentAt` | the attacker. One shape, the head circle. `kind` and `mult` are ignored if present. An attack with no `id` is dropped |
| `attackResult` | events | `id, outcome, dmg` | the defender. `outcome` is Dodge, Block, Hit, or dropped |
| `state` | events | `hp, score` | whoever's totals changed, and at least every 1 s. Each client owns its own HP and score |
| `ko` | events | `loser: "me" \| "you"` | the client whose HP hit 0 sends `"me"` |
| `rematch` | events | (none) | both. When both have sent it, the host sends a new `start` |
| `full` | events | (none) | host, to a third peer |
| (pose) | pose | `t` = netNow, `k` = 45 numbers | whoever has a pose. No video, no world landmarks |

### 11.4 Damage and the round
Constants live at the top of `versus.js`:

| Constant | Value |
|---|---|
| `MAX_HP` | 100 |
| `VERSUS_ROUND_S` | 60 |
| `ATTACK_DAMAGE` | 10 |
| `PUNCH_ATTACK_COOLDOWN_S` | 0.6 |
| `BLOCK_DAMAGE_FACTOR` | 0.5 |
| `POSE_HZ` | 20 |
| `OPP_RENDER_DELAY_S` | 0.10 |
| `POSE_STALE_WARN_S` / `POSE_STALE_DROP_S` | 3 / 10 |
| `MAX_QUEUED_ATTACKS` | 2 |
| `PIP_W_FRAC` | 0.28 |
| `OPP_HOLD_S` / `OPP_FADE_S` | 0.4 / 0.4 |
| `OUTGOING_TTL_S` | 6 |
| `BLOCK_SCORE` / `LANDED_BONUS` | 25 / 100 |

A clean hit removes `ATTACK_DAMAGE` (10). A block removes `round(10 * 0.5)` = 5. There is no PERFECT multiplier and no jab/hook/uppercut table. HP never goes below 0.

The attacker does not score a grade. When the result arrives they get +100 for a Hit, and floating text `LANDED -10`, `DODGED`, or `BLOCKED`. `dropped` and anything unanswered after 6 s are removed quietly.

The round ends at `t0 + 60 s`, or on KO. The displayed clock pauses while the opponent pose is stale (3–10 s) and the round is not ended by the clock during that pause. At 10 s with no pose the link is treated as a disconnect. Higher HP wins; equal HP uses score; otherwise it is a draw.

Closing the link mid-round shows "Opponent left" and a forfeit win for the player still in the round, then returns to the lobby. `init()` is not called again.

Rematch: both clients send `rematch`, HP / score / queues reset, and the host sends a new `start`. The camera and models stay loaded.

### 11.5 Picture
`versus.js` keeps `oppBuf`, the last 30 `{t, k}` samples (older than ~2 s are pruned, and a sample older than the newest is ignored). The opponent is drawn full screen at `renderT = netNow() - 0.10`, interpolating the two samples that straddle `renderT`. Past the newest sample, that sample is held for 0.4 s and then fades out over 0.4 s. Coordinates are the sender's raw camera x, so their right hand is on screen-left, as if they face you. The drawing is head, shoulders, arms, hands, and hips, in neon cyan, thicker when that wrist is moving fast.

The local camera is a thumbnail (`pipRect`, bottom left, `PIP_W_FRAC` wide, camera aspect, mirrored selfie) with the local skeleton on top of it. That thumbnail is how you see yourself slip.

The dodge circle is spawned in **camera pixels** on the local nose, and `checkCollision` is unchanged. `mapCamPoint` / `mapCamRadius` mirror that circle into a rectangle. The full frame and the thumbnail are two rectangles fed to the same function, so the circle sits on the head in the thumbnail and at the same fraction of the main view (aimed at the viewer). A fast opponent wrist, or the `attack` message if it arrives first, starts a flash and a growing glow on that hand, plus the word `INCOMING`, and then the warning/active circle.

### 11.6 Lobby, HUD, `?bot=1`
The start panel has Solo, Create room, and Join room (5-character code, uppercased), plus a name field (default "Player") sent in `hello`. After the link: the code and a Copy button, "Opponent connected: \<name\>", ping, and a Ready control for each side. The host sees Start only when both are ready. Leave closes the peer and returns to the lobby.

The versus HUD replaces the solo score pill: my HP bar, name and score on the top left (green to red as HP falls); the opponent's bar on the top right, filling from the right; the clock and `ping NN ms` in the top center (red at ≤ 5 s).

`?bot=1` sets up a versus room whose peer is `BotOpponent`. It implements the same `EventTarget` surface as `Net` (`sendEvent`, `sendPose`, `netNow`, `close`, plus `setLive`) and does not load PeerJS. Its skeleton fills the main view, sways, and reaches in with the right arm every few seconds. While the round is live it sends an `attack` (`id`, `sentAt` only) every 4–8 s, which you dodge on the head circle. It answers your punches with a 50 % dodge (otherwise block or hit) using the fixed damage, and sends `state`. One tab can play a full versus round.

### 11.7 Deviations from the written spec
These follow the code that was already in the repo.

- `Attack.phase` returns `"done"`, not `"over"`. Resolution is unchanged: Hit or Block as soon as `checkCollision` says so, Dodge when the active window ends.
- Pose packets use `t` as the sample time. Events packets use `t` as the type. One field cannot be both.
- `TARGET_RESPAWN_DELAY_S` and `MAX_RUNTIME` stay at their solo values. Versus does not spawn targets; its round length is `VERSUS_ROUND_S`.
- The host also sends ping/pong, so both HUDs can show a ping. Only the guest writes `clockOffset`.
- `endRound` still `console.log`s the totals in versus, and also draws the card. The camera is not stopped.
- A name field was added so `hello` and the HUD have something to show. The fade after a held pose lasts 0.4 s (the spec says "fade" and not for how long).
- The countdown number is `ceil(t0 - netNow())`, so the 0.5 s margin can display `4` before `3`.
