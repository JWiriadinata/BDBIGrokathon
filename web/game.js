// Browser port of main_python_analysis.py + defense.py + target_utils.py:
// prompted punches (callout → osu!-style target → CNN check) and telegraphed attacks to dodge.
import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs";
import { CLASS_NAMES, CLIP_FRAMES, TARGET_FPS, normalizeSkeleton, extractLiveClip, mirrorSkeleton, buildTensor } from "./pipeline.js";
import { PunchDetector, GUARD_READY_SAMPLES } from "./punchDetector.js";
// VERSUS: networking and round rules. Solo never constructs a Net.
import { Net } from "./net.js";
import {
  Versus, BotOpponent, BLOCK_SCORE, LANDED_BONUS,
  hitDamage, blockDamage, enqueueAttack, judge, verdictTitle,
  mapCamPoint, mapCamRadius, pipRect,
} from "./versus.js";

// ── Game config (same values as main_python_analysis.py) ─────────────────────
const STANCE = "orthodox";
const ALLOW_RANDOM_SIDES = true;
const CALLOUT_DURATION_S = 0.4;
const APPROACH_DURATION_S = 1.0;
const PERFECT_WINDOW_S = 0.10;
const GOOD_WINDOW_S = 0.25;
const TARGET_VISUAL_SCALE = 2.5;
const HIT_RADIUS_SCALE = 3.0;
const MAX_SAME_TARGET_IN_A_ROW = 2;
const SCORE_PERFECT = 300, SCORE_GOOD = 100, SCORE_DEFENSE_BONUS = 50, SCORE_HIT_PENALTY = 50;
const FLOATING_TEXT_S = 0.6, HIT_VIGNETTE_S = 0.3, SCHEDULE_MARGIN_S = 0.1;
const MODEL_CONF_THRESHOLD = 0.55, LOOK_BACK_SECONDS = 2.5, WORLD_BUFFER_FRAMES = 150;
const MAX_RUNTIME = 30, PRESTART_COUNTDOWN_S = 3, PREDICTION_OVERLAY_S = 1.8, TARGET_RESPAWN_DELAY_S = 2.0;
// Punch detection (punchDetector.js). A detected punch is only listened to for the hand of the active target.
const SEGMENT_BUFFER_FRAMES = 3;     // camera frames added either side of a punch's start→end for the model
const PUNCH_RESOLVE_GRACE_S = 0.35;  // after the hit window closes, wait this long for a punch still being thrown
const PENDING_CLASSIFY_S = 0.6;      // after a wrist-overlap hit, wait this long for the punch segment to finish
const SHOW_PUNCH_DEBUG = true;       // per-hand detector state + peak circles (like both_hands_tracking.py)
const PEAK_DISPLAY_S = 0.5;
const MODEL_TO_TARGET ={ jab: "jab", "lead-hook": "hook", "rear-uppercut": "uppercut" };
// Pixel sizes below were tuned for the Python app's 960×540 camera; scaled by W / 960.
const BASE_W = 960, TARGET_RADIUS_PX = 25;

// ── Attack config (same values as defense.py) ────────────────────────────────
const ATTACK_FLASH_COUNT = 3, ATTACK_WARNING_S = 0.9, ATTACK_ACTIVE_S = 1.0;
const ATTACK_WARNING_ALPHA = 0.35, ATTACK_SWEEP_FRACTION = 0.5;
const SPAWN_DELAY_RANGE = [3, 7], FIRST_ATTACK_DELAY_S = 2.0, RESERVED_MARGIN_S = 0.1, ARM_PADDING_PX = 20;
const ATTACK_WEIGHTS = { jab: 0.4, hook: 0.35, uppercut: 0.25 };
const JAB_RADIUS_SW = 0.45, HOOK_BAND_HEIGHT_SW = 0.4, UPPERCUT_BAND_WIDTH_SW = 0.8;
const HEAD_HALF_WIDTH_SW = 0.15, HEAD_TOP_SW = 0.35, HEAD_BOTTOM_SW = 0.15, MIN_SHOULDER_PX = 40;

// Colors as [r, g, b].
const PUNCH_COLORS = { jab: [255, 0, 0], hook: [0, 255, 0], uppercut: [0, 0, 255] };
const GRADE_COLORS = { PERFECT: [255, 230, 0], GOOD: [255, 255, 255], MISS: [160, 160, 160] };
const ATTACK_COLOR = [255, 165, 0];
const WHITE = [255, 255, 255];
const LABELS = { jab: "JAB", hook: "HOOK", uppercut: "UPPERCUT", L: "LEFT", R: "RIGHT" };

// MediaPipe Pose landmark indices.
const NOSE = 0, LS = 11, RS = 12, LE = 13, RE = 14, LW = 15, RW = 16, LHIP = 23, RHIP = 24;
const HAND_CONTACT = { L: [15, 19, 21, 17], R: [16, 20, 22, 18] };   // wrist, index, thumb, pinky
const SIDE_COLORS = { L: [0, 255, 0], R: [255, 165, 0] };             // debug overlay only

const POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const panel = document.getElementById("panel");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start");
// VERSUS: lobby / room controls. Solo still uses the same panel shell.
const lobbyEl = document.getElementById("lobby");
const roomEl = document.getElementById("room");
const lobbyMsg = document.getElementById("lobby-msg");
const botNote = document.getElementById("bot-note");
const soloBtn = document.getElementById("solo");
const createBtn = document.getElementById("create");
const joinBtn = document.getElementById("join");
const fightBotBtn = document.getElementById("fight-bot");
const nameInput = document.getElementById("player-name");
const codeInput = document.getElementById("room-code");
const codeLine = document.getElementById("code-line");
const codeValue = document.getElementById("code-value");
const botLine = document.getElementById("bot-line");
const copyBtn = document.getElementById("copy");
const roomStatus = document.getElementById("room-status");
const roomPing = document.getElementById("room-ping");
const readyYou = document.getElementById("ready-you");
const readyOpp = document.getElementById("ready-opp");
const waitHost = document.getElementById("wait-host");
const readyBtn = document.getElementById("ready");
const startVersusBtn = document.getElementById("start-versus");
const leaveBtn = document.getElementById("leave");
let W = 960, H = 540, S = 1;
let landmarker, session, drawing;
// VERSUS: "solo" keeps the original round. "versus" is the online (or ?bot=1) fight.
let GAME_MODE = "solo";
let net = null, versus = null, connecting = false;
const BOT_MODE = new URLSearchParams(location.search).get("bot") === "1";

// ── Small helpers ────────────────────────────────────────────────────────────
const rgba = (c, a = 1, k = 1) => `rgba(${c[0] * k | 0},${c[1] * k | 0},${c[2] * k | 0},${a})`;
const rand = (a, b) => a + Math.random() * (b - a);
const randint = (a, b) => Math.floor(rand(a, b + 1));
const toPx = (lm) => [Math.trunc(lm.x * W), Math.trunc(lm.y * H)];
const mx = (x) => W - 1 - x;   // camera x → mirrored display x
const callout = (type, side) => `${LABELS[side]} ${LABELS[type]}`;
const windowsOverlap = (a, b, m = 0) => !!a && !!b && a[0] < b[1] + m && a[1] > b[0] - m;
const rectsOverlap = (a, b) => !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3]);

function circleOverlapsRect(cx, cy, r, [x1, y1, x2, y2]) {
  const nx = Math.min(Math.max(cx, x1), x2), ny = Math.min(Math.max(cy, y1), y2);
  return (cx - nx) ** 2 + (cy - ny) ** 2 <= r * r;
}
const shapeOverlapsRect = (s, rect) =>
  !!s && (s[0] === "circle" ? circleOverlapsRect(s[1], s[2], s[3], rect) : rectsOverlap(s.slice(1), rect));

function choosePunchType() {
  const r = Math.random();
  return r < 0.625 ? "jab" : r < 0.875 ? "hook" : "uppercut";
}

function choosePunchSide(type, allowRandom) {
  if (allowRandom) return Math.random() < 0.5 ? "L" : "R";
  const lead = STANCE === "southpaw" ? "R" : "L", rear = lead === "R" ? "L" : "R";
  return type === "uppercut" ? rear : lead;
}

// ── Offense: target placement and hit test (target_utils.py) ─────────────────
function respawnTarget(lms, radius, side, margin, avoidHandsPx) {
  const edge = Math.max(radius, margin);
  const [lsx, lsy] = toPx(lms[LS]), [rsx, rsy] = toPx(lms[RS]);
  const lhy = toPx(lms[LHIP])[1], rhy = toPx(lms[RHIP])[1];
  const ny = toPx(lms[NOSE])[1];
  const sw = Math.abs(rsx - lsx);
  const fullMin = Math.max(edge, Math.min(lsx, rsx) - sw), fullMax = Math.min(W - edge, Math.max(lsx, rsx) + sw);
  let xmin = fullMin, xmax = fullMax;
  // Keep the target on the half of the body of the hand that should hit it.
  const bodyCx = Math.floor((lsx + rsx) / 2), handShoulderX = side === "L" ? lsx : rsx;
  if (handShoulderX >= bodyCx) xmin = Math.max(xmin, bodyCx + radius);
  else xmax = Math.min(xmax, bodyCx - radius);
  if (xmax <= xmin) [xmin, xmax] = [fullMin, fullMax];
  const shoulderY = Math.floor((lsy + rsy) / 2), waistY = Math.floor((lhy + rhy) / 2);
  const ymin = Math.max(edge, Math.min(ny, shoulderY));
  const ymax = Math.min(H - edge, Math.max(ny, shoulderY + Math.floor((waistY - shoulderY) / 2)));
  if (xmax <= xmin || ymax <= ymin) return null;
  const hands = [toPx(lms[LW]), toPx(lms[RW])];
  let best = null, bestD = -1;
  for (let i = 0; i < 20; i++) {
    const c = [randint(xmin, xmax), randint(ymin, ymax)];
    const d = Math.min(...hands.map(([hx, hy]) => Math.hypot(c[0] - hx, c[1] - hy)));
    if (d >= avoidHandsPx) return c;
    if (d > bestD) [best, bestD] = [c, d];
  }
  return best;
}

function pointsHitCircle(points, [cx, cy], radius) {
  const r = Math.max(1, Math.trunc(radius * HIT_RADIUS_SCALE));
  return points.some((p) => {
    if ((p.visibility ?? 1) < 0.35) return false;
    const [x, y] = toPx(p);
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r;
  });
}
const handHitsCircle = (lms, center, radius, side) => pointsHitCircle(HAND_CONTACT[side].map((i) => lms[i]), center, radius);

// ── Defense: attacks (defense.py) ────────────────────────────────────────────
function posePoints(lms) {
  const p = { nose: toPx(lms[NOSE]), ls: toPx(lms[LS]), rs: toPx(lms[RS]), le: toPx(lms[LE]),
              re: toPx(lms[RE]), lw: toPx(lms[LW]), rw: toPx(lms[RW]) };
  p.sw = Math.max(MIN_SHOULDER_PX * S, Math.abs(p.ls[0] - p.rs[0]));
  p.bodyCx = Math.floor((p.ls[0] + p.rs[0]) / 2);
  return p;
}

function headRect({ nose: [nx, ny], sw }) {
  return [nx - HEAD_HALF_WIDTH_SW * sw, ny - HEAD_TOP_SW * sw, nx + HEAD_HALF_WIDTH_SW * sw, ny + HEAD_BOTTOM_SW * sw];
}

// Forearms (not the whole arm) decide a block, so a shoulder next to the head isn't a guard.
function forearmRects(p) {
  const pad = ARM_PADDING_PX * S;
  return [[p.le, p.lw], [p.re, p.rw]].map(([[ex, ey], [wx, wy]]) =>
    [Math.min(ex, wx) - pad, Math.min(ey, wy) - pad, Math.max(ex, wx) + pad, Math.max(ey, wy) + pad]);
}

class Attack {
  constructor(kind, spawnTs, { circle = null, band = null, direction = 1, anchor }) {
    Object.assign(this, { kind, spawnTs, circle, band, direction, anchor });
    this.activeStart = spawnTs + ATTACK_WARNING_S;
    this.activeEnd = this.activeStart + ATTACK_ACTIVE_S;
  }
  phase(now) { return now < this.activeStart ? "warning" : now < this.activeEnd ? "active" : "done"; }
  sweptRect(now) {
    const [x1, y1, x2, y2] = this.band;
    const p = Math.max(0, Math.min(1, (now - this.activeStart) / Math.max(1e-3, ATTACK_ACTIVE_S * ATTACK_SWEEP_FRACTION)));
    if (this.kind === "hook") {
      const span = Math.trunc((x2 - x1) * p);
      return this.direction > 0 ? [x1, y1, x1 + span, y2] : [x2 - span, y1, x2, y2];
    }
    const span = Math.trunc((y2 - y1) * p);   // uppercut sweeps bottom → top
    return [x1, y2 - span, x2, y2];
  }
  dangerShape(now) {
    if (this.phase(now) !== "active") return null;
    if (this.kind === "jab") return ["circle", ...this.circle];
    const r = this.sweptRect(now);
    return r[0] >= r[2] || r[1] >= r[3] ? null : ["rect", ...r];
  }
  flashOn(now) {
    const period = ATTACK_WARNING_S / Math.max(1, ATTACK_FLASH_COUNT);
    return ((now - this.spawnTs) % period) < period * 0.55;
  }
  drawChevrons(alpha) {
    const [x1, y1, x2, y2] = this.band;
    let pts, size, dir;
    if (this.kind === "hook") {
      size = (y2 - y1) * 0.6;
      pts = [0, 1, 2, 3, 4].map((i) => [x1 + (x2 - x1) * (i + 0.5) / 5, (y1 + y2) / 2]);
      dir = [this.direction, 0];
    } else {
      size = (x2 - x1) * 0.45;
      pts = [0, 1, 2, 3].map((i) => [(x1 + x2) / 2, y1 + (y2 - y1) * (i + 0.5) / 4]);
      dir = [0, -1];
    }
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = rgba(WHITE);
    ctx.lineWidth = Math.max(3, Math.trunc(size * 0.14));
    const [dx, dy] = dir, [px, py] = [-dy, dx], half = size * 0.5;
    for (const [cx, cy] of pts) {
      ctx.beginPath();
      ctx.moveTo(cx - dx * half * 0.6 + px * half, cy - dy * half * 0.6 + py * half);
      ctx.lineTo(cx + dx * half * 0.6, cy + dy * half * 0.6);
      ctx.lineTo(cx - dx * half * 0.6 - px * half, cy - dy * half * 0.6 - py * half);
      ctx.stroke();
    }
    ctx.restore();
  }
  draw(now) {
    const phase = this.phase(now);
    if (phase === "done") return;
    ctx.save();
    if (this.kind === "jab") {
      const [cx, cy, r] = this.circle;
      if (phase === "warning") {
        if (this.flashOn(now)) fillCircle(cx, cy, r, rgba(ATTACK_COLOR, ATTACK_WARNING_ALPHA));
        strokeCircle(cx, cy, r, rgba(WHITE, 0.5), 2);
      } else {
        fillCircle(cx, cy, r, rgba(ATTACK_COLOR, 0.9));
        strokeCircle(cx, cy, r * 0.55, rgba(WHITE, 0.9), 3);
        strokeCircle(cx, cy, r, rgba(WHITE, 0.9), 4);
      }
      ctx.restore();
      return;
    }
    const [x1, y1, x2, y2] = this.band;
    if (phase === "warning") {
      const on = this.flashOn(now);
      if (on) fillRect([x1, y1, x2, y2], rgba(ATTACK_COLOR, ATTACK_WARNING_ALPHA));
      this.drawChevrons(on ? 0.85 : 0.35);
    } else {
      fillRect([x1, y1, x2, y2], rgba(ATTACK_COLOR, ATTACK_WARNING_ALPHA));
      fillRect(this.sweptRect(now), rgba(ATTACK_COLOR, 0.95));
      this.drawChevrons(1);
    }
    ctx.strokeStyle = rgba(WHITE);
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (this.kind === "hook") { ctx.moveTo(x1, y1); ctx.lineTo(x2, y1); ctx.moveTo(x1, y2); ctx.lineTo(x2, y2); }
    else { ctx.moveTo(x1, y1); ctx.lineTo(x1, y2); ctx.moveTo(x2, y1); ctx.lineTo(x2, y2); }
    ctx.stroke();
    ctx.restore();
  }
}

function spawnAttack(lms, now, forcedKind = null) {
  const p = posePoints(lms), kinds = Object.keys(ATTACK_WEIGHTS);
  // VERSUS: a queued attack passes its kind. Solo still rolls ATTACK_WEIGHTS; only the sweep direction is random.
  let kind = forcedKind;
  if (!kind) {
    let r = Math.random();
    kind = kinds[kinds.length - 1];
    for (const k of kinds) { if (r < ATTACK_WEIGHTS[k]) { kind = k; break; } r -= ATTACK_WEIGHTS[k]; }
  }
  const { sw } = p, [nx, ny] = p.nose;
  if (kind === "jab") {
    const rad = Math.trunc(JAB_RADIUS_SW * sw);
    const cx = Math.min(Math.max(nx, rad), W - rad), cy = Math.min(Math.max(ny, rad), H - rad);
    return new Attack("jab", now, { anchor: [cx, cy], circle: [cx, cy, rad] });
  }
  if (kind === "hook") {
    const half = Math.trunc(HOOK_BAND_HEIGHT_SW * sw / 2);
    return new Attack("hook", now, { anchor: [nx, ny], band: [0, Math.max(0, ny - half), W - 1, Math.min(H - 1, ny + half)],
                                     direction: Math.random() < 0.5 ? -1 : 1 });
  }
  const half = Math.trunc(UPPERCUT_BAND_WIDTH_SW * sw / 2);
  return new Attack("uppercut", now, { anchor: [p.bodyCx, ny], direction: -1,
                                       band: [Math.max(0, p.bodyCx - half), 0, Math.min(W - 1, p.bodyCx + half), H - 1] });
}

// "Block" | "Hit" | null. A dodge is decided when the active phase ends untouched.
function checkCollision(lms, attack, now) {
  const danger = attack.dangerShape(now);
  if (!danger) return null;
  const p = posePoints(lms), head = headRect(p);
  if (!shapeOverlapsRect(danger, head)) return null;
  const pad = ARM_PADDING_PX * S, guardZone = [head[0] - pad, head[1] - pad, head[2] + pad, head[3] + pad];
  // Leaning back from an uppercut can't be seen in 2D; only a side step or a guard counts.
  return forearmRects(p).some((f) => rectsOverlap(f, guardZone) && shapeOverlapsRect(danger, f)) ? "Block" : "Hit";
}

const defense = {
  queue: [],
  banner: null,
  onDropped: null,
  // keepQueue: a versus countdown can receive attacks that belong to the round about to start.
  reset(opts = {}) {
    this.attack = null; this.nextSpawn = null; this.lastEvent = null; this.banner = null;
    this.stats = { blocked: 0, dodged: 0, hit: 0 };
    if (!opts.keepQueue) this.queue = [];
  },
  activeWindow() { return this.attack ? [this.attack.activeStart, this.attack.activeEnd] : null; },
  // VERSUS: cap the queue at 2. The overflow is reported immediately and discarded.
  enqueue(meta) {
    if (!enqueueAttack(this.queue, { meta })) {
      this.onDropped?.(meta);
      return false;
    }
    return true;
  },
  update(lms, now, reserved) {
    // VERSUS: one head-circle per queued punch. No random timer and no punch-target windows.
    if (GAME_MODE === "versus") {
      if (!this.attack && this.queue.length && lms) {
        const item = this.queue.shift();
        this.attack = spawnAttack(lms, now, "jab");
        this.attack.meta = item.meta;
        this.banner = { text: "INCOMING", until: this.attack.activeStart };
        versus?.armHandCue(now);
      }
    } else {
      if (this.nextSpawn === null) this.nextSpawn = now + FIRST_ATTACK_DELAY_S;
      if (!this.attack && lms && now >= this.nextSpawn) {
        const start = now + ATTACK_WARNING_S, end = start + ATTACK_ACTIVE_S;
        // If this would collide with a punch moment, try again next frame.
        if (!reserved.some(([a, b]) => start < b + RESERVED_MARGIN_S && end > a - RESERVED_MARGIN_S)) this.attack = spawnAttack(lms, now);
      }
    }
    const a = this.attack;
    if (!a) return null;
    let outcome = a.phase(now) === "active" && lms ? checkCollision(lms, a, now) : null;
    if (!outcome && now >= a.activeEnd) outcome = "Dodge";
    if (outcome) {
      this.stats[{ Block: "blocked", Hit: "hit", Dodge: "dodged" }[outcome]] += 1;
      this.lastEvent = { outcome, pos: a.anchor, meta: a.meta || null };
      // VERSUS: the next attack comes from the opponent, not a random delay.
      if (GAME_MODE !== "versus") this.nextSpawn = now + rand(...SPAWN_DELAY_RANGE);
      this.attack = null;
    }
    if (GAME_MODE !== "versus") a.draw(now);
    return outcome;
  },
};

// ── Drawing helpers ──────────────────────────────────────────────────────────
function fillCircle(x, y, r, style) { ctx.fillStyle = style; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); }
function strokeCircle(x, y, r, style, lw) { ctx.strokeStyle = style; ctx.lineWidth = lw; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.stroke(); }
function fillRect([x1, y1, x2, y2], style) { ctx.fillStyle = style; ctx.fillRect(x1, y1, x2 - x1, y2 - y1); }

function fitFont(text, maxW, maxH) {
  ctx.font = "bold 100px system-ui, sans-serif";
  const size = Math.max(10, Math.min(100 * maxW / ctx.measureText(text).width, maxH * 1.35));
  ctx.font = `bold ${size}px system-ui, sans-serif`;
  return size;
}

function textCentered(text, x, y, color, alpha = 1, outline = true) {
  ctx.save();
  ctx.globalAlpha = Math.max(0, alpha);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  if (outline) {
    ctx.lineWidth = Math.max(3, parseFloat(ctx.font.match(/(\d+(\.\d+)?)px/)[1]) / 8);
    ctx.strokeStyle = "rgba(20,20,20,0.9)";
    ctx.lineJoin = "round";
    ctx.strokeText(text, x, y);
  }
  ctx.fillStyle = rgba(color);
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawOsuTarget([cx, cy], r, color, progress) {
  const p = Math.max(0, Math.min(1, progress)), appear = Math.min(1, progress / 0.15);
  ctx.save();
  ctx.globalAlpha = 0.30 * appear;
  fillCircle(cx, cy, r * 1.25, rgba(color));
  ctx.globalAlpha = 0.92 * appear;
  fillCircle(cx, cy, r, rgba(color));
  fillCircle(cx, cy, r * 0.72, rgba(color, 1, 0.75));
  strokeCircle(cx, cy, r, rgba(WHITE), Math.max(3, r / 12));
  ctx.globalAlpha = 1;
  const ringT = Math.max(3, Math.trunc(r / 14));
  strokeCircle(cx, cy, r + (3 * r - r) * (1 - p), rgba(WHITE), ringT);
  if (progress >= 1) strokeCircle(cx, cy, r + ringT + 2, rgba(color), 2);   // hit window past ideal: pulse the edge
  ctx.restore();
}

function drawCallout(text, color, elapsed) {
  const fade = 0.1;
  let k = 1, scale = 1;
  if (elapsed < fade) { k = elapsed / fade; scale = 0.8 + 0.2 * k; }
  else if (elapsed > CALLOUT_DURATION_S - fade) k = (CALLOUT_DURATION_S - elapsed) / fade;
  k = Math.max(0, Math.min(1, k));
  const bandH = Math.max(20, H * 0.22 * scale), y1 = H / 2 - bandH / 2;
  ctx.save();
  ctx.globalAlpha = 0.85 * k;
  fillRect([0, y1, W, y1 + bandH], rgba(color));
  fillRect([0, y1 - 1.5, W, y1 + 1.5], rgba(WHITE));
  fillRect([0, y1 + bandH - 1.5, W, y1 + bandH + 1.5], rgba(WHITE));
  ctx.restore();
  fitFont(text, W * 0.88, bandH * 0.55);
  textCentered(text, W / 2, H / 2, WHITE, k);
}

function pill(x1, y1, x2, y2) {
  const r = (y2 - y1) / 2;
  ctx.fillStyle = "rgba(20,20,20,0.6)";
  ctx.beginPath();
  ctx.roundRect(x1, y1, x2 - x1, y2 - y1, r);
  ctx.fill();
}

function drawHud(score, remaining) {
  const margin = Math.max(12, W / 60), pillH = Math.max(40, H / 12);
  fitFont("SCORE 00000", W * 0.22, pillH * 0.5);
  const s = Math.max(0, Math.ceil(remaining));
  const timeText = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`, scoreText = `SCORE ${score}`;
  const tx2 = margin + ctx.measureText(timeText).width + pillH;
  pill(margin, margin, tx2, margin + pillH);
  textCentered(timeText, (margin + tx2) / 2, margin + pillH / 2, remaining <= 5 ? [255, 80, 80] : WHITE, 1, false);
  const sx1 = W - margin - ctx.measureText(scoreText).width - pillH;
  pill(sx1, margin, W - margin, margin + pillH);
  textCentered(scoreText, (sx1 + W - margin) / 2, margin + pillH / 2, WHITE, 1, false);
}

function drawVignette(strength) {
  const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.hypot(W, H) / 2);
  g.addColorStop(0, "rgba(255,0,0,0)");
  g.addColorStop(1, `rgba(255,0,0,${0.8 * Math.min(1, strength)})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

// ── Round state ──────────────────────────────────────────────────────────────
let mode = "loading", modeStart = 0, lastVideoTime = -1;
let st;   // per-round state, see newRound()
const worldBuf = [];   // {t, skel} normalized world skeletons
// Runs on every camera frame (also before the round) so the guard and scale are learned by the time play starts.
const detector = new PunchDetector({
  L: { wrist: LW, shoulder: LS, contact: HAND_CONTACT.L },
  R: { wrist: RW, shoulder: RS, contact: HAND_CONTACT.R },
});

function newRound(now) {
  // VERSUS: attacks that arrived during the countdown belong to this round.
  defense.reset({ keepQueue: GAME_MODE === "versus" });
  if (GAME_MODE === "versus" && versus) {
    versus.resetRound();
    net?.setLive?.(true);
  }
  st = {
    start: now, score: 0, phase: "idle", type: null, side: null, center: null, spawnTs: 0, idealTs: 0,
    calloutTs: 0, nextSpawn: 0, handInsidePrev: true, recent: [], reserved: [], texts: [], vignetteUntil: 0,
    prediction: null, pending: null, timing: { perfect: 0, good: 0, miss: 0 },
  };
}

function pickNextTarget() {
  let type, side;
  for (let i = 0; i < 25; i++) {
    type = choosePunchType();
    side = choosePunchSide(type, ALLOW_RANDOM_SIDES);
    if (st.recent.length < MAX_SAME_TARGET_IN_A_ROW || st.recent.some((r) => r !== type + side)) break;
  }
  st.recent.push(type + side);
  if (st.recent.length > MAX_SAME_TARGET_IN_A_ROW) st.recent.shift();
  return [type, side];
}

function bufferFps() {
  const n = worldBuf.length;
  return n > 1 ? (n - 1) / (worldBuf[n - 1].t - worldBuf[0].t) : 30;
}

// Fallback when no punch segment was detected: frames since the target appeared, or a fixed look-back.
function snapshotClip(spawnTs) {
  const fps = bufferFps();
  let frames = worldBuf.filter((f) => f.t >= spawnTs);
  const minRaw = Math.max(2, Math.round(LOOK_BACK_SECONDS * fps));
  if (frames.length < minRaw) frames = worldBuf.slice(-minRaw);
  return extractLiveClip(frames.map((f) => f.skel), fps);
}

// A detected punch: the velocity peak is searched only in its start→end frames ± SEGMENT_BUFFER_FRAMES.
// The model still needs CLIP_FRAMES frames ending at that peak, so earlier frames are prepended as lead-in:
// extractLiveClip only searches after its first CLIP_FRAMES (resampled) frames, i.e. from about `lo` onward.
function segmentClip(startT, endT) {
  const fps = bufferFps();
  const first = worldBuf.findIndex((f) => f.t >= startT);
  const last = worldBuf.findLastIndex((f) => f.t <= endT);
  if (first < 0 || last < first) return null;
  const lo = Math.max(0, first - SEGMENT_BUFFER_FRAMES), hi = Math.min(worldBuf.length - 1, last + SEGMENT_BUFFER_FRAMES);
  const from = Math.max(0, lo - Math.ceil((CLIP_FRAMES - 1) / TARGET_FPS * fps));
  return extractLiveClip(worldBuf.slice(from, hi + 1).map((f) => f.skel), fps);
}

// The clip is built synchronously (worldBuf keeps changing); only the model run is async.
function requestPrediction(round, type, side, clip) {
  classify(type, side, clip)
    .then((p) => { round.prediction = { ...p, until: performance.now() / 1000 + PREDICTION_OVERLAY_S }; })
    .catch((e) => console.error("classify failed", e));
}

const isPunchFor = (e, side, sinceTs) => e.type === "punch" && e.side === side && e.peakT >= sinceTs;

// After a wrist-overlap hit, classify the called hand's punch once the detector reports it finished.
function resolvePending(now, events) {
  const p = st.pending, ev = events.find((e) => isPunchFor(e, p.side, p.spawnTs));
  if (ev) requestPrediction(st, p.type, p.side, segmentClip(ev.startT, ev.endT) ?? snapshotClip(p.spawnTs));
  else if (now > p.deadline) requestPrediction(st, p.type, p.side, snapshotClip(p.spawnTs));
  else return;
  st.pending = null;
}

async function classify(type, side, clip) {
  if (!clip) {
    const needed = Math.max(2, Math.round(CLIP_FRAMES / TARGET_FPS * bufferFps()));
    return { text: `Warming up (${worldBuf.length}/${needed} frames)`, color: [220, 220, 220] };
  }
  // The model only learned the stance-default hand per punch type; mirror the other hand.
  if (side !== choosePunchSide(type, false)) clip = mirrorSkeleton(clip);
  const out = await session.run({ input: new ort.Tensor("float32", buildTensor(clip), [1, 7, CLIP_FRAMES, 4, 3]) });
  const logits = Array.from(out.logits.data), m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m)), sum = exps.reduce((a, b) => a + b);
  const probs = exps.map((e) => e / sum), best = probs.indexOf(Math.max(...probs));
  const name = CLASS_NAMES[best], conf = probs[best], confident = conf >= MODEL_CONF_THRESHOLD;
  const label = name.replace(/(^|-)\w/g, (s) => s.toUpperCase());
  return { text: `Predicted: ${label} (${Math.round(conf * 100)}%)${confident ? "" : " (low)"}`,
           color: confident ? [0, 255, 0] : [255, 165, 0] };
}

function stepRound(lms, now, events) {
  st.reserved = st.reserved.filter((w) => w[1] > now);
  const radius = TARGET_RADIUS_PX * S, visualR = radius * TARGET_VISUAL_SCALE;

  // 1. Start a callout when the respawn timer allows and no attack is in its dodge phase.
  if (st.phase === "idle" && lms && now >= st.nextSpawn) {
    const calloutEnd = now + CALLOUT_DURATION_S, plannedHit = calloutEnd + APPROACH_DURATION_S;
    const planned = [plannedHit - GOOD_WINDOW_S, plannedHit + GOOD_WINDOW_S], attackWin = defense.activeWindow();
    if (!windowsOverlap(attackWin, [now, calloutEnd], SCHEDULE_MARGIN_S) && !windowsOverlap(attackWin, planned, SCHEDULE_MARGIN_S)) {
      [st.type, st.side] = pickNextTarget();
      st.phase = "callout";
      st.calloutTs = now;
      st.reserved.push([now, calloutEnd], planned);
    }
  }

  // 2. Callout over → spawn the target.
  if (st.phase === "callout" && lms && now >= st.calloutTs + CALLOUT_DURATION_S) {
    const c = respawnTarget(lms, radius, st.side, visualR + 6, radius * HIT_RADIUS_SCALE + 20 * S);
    if (c) {
      Object.assign(st, { center: c, spawnTs: now, idealTs: now + APPROACH_DURATION_S, phase: "active", handInsidePrev: true });
      st.reserved.push([st.idealTs - GOOD_WINDOW_S, st.idealTs + GOOD_WINDOW_S]);
    }
  }

  // 3. Hit = either signal, inside the timing window:
  //    - overlap: the called hand ENTERING the target (a resting guard doesn't count), graded at that moment;
  //    - punch: a detected punch from the called hand peaking on the target, graded at the peak time.
  //    Any other detected punch from the called hand while the target is up is a miss. Punches from the other
  //    hand, and punches when no target is active, are ignored.
  if (st.pending) resolvePending(now, events);
  if (st.phase === "active") {
    const windowOpen = now >= st.idealTs - GOOD_WINDOW_S, windowClose = st.idealTs + GOOD_WINDOW_S;
    const punch = events.find((e) => isPunchFor(e, st.side, st.spawnTs));
    let overlapHit = false;
    if (lms) {
      const inside = handHitsCircle(lms, st.center, radius, st.side);
      overlapHit = windowOpen && now <= windowClose && inside && !st.handInsidePrev;
      st.handInsidePrev = inside;
    }
    const punchHit = !!punch && Math.abs(punch.peakT - st.idealTs) <= GOOD_WINDOW_S && pointsHitCircle(punch.peakPts, st.center, radius);
    const hit = overlapHit || punchHit;
    // A punch still being thrown when the window closes gets a moment to report its (possibly in-window) peak.
    const settling = now <= windowClose + PUNCH_RESOLVE_GRACE_S && detector.trackers[st.side].state === "throwing";
    const missed = !hit && (!!punch || (now > windowClose && !settling));
    if (hit) {
      const gradeTs = punchHit ? punch.peakT : now;
      const grade = Math.abs(gradeTs - st.idealTs) <= PERFECT_WINDOW_S ? "PERFECT" : "GOOD";
      st.timing[grade.toLowerCase()] += 1;
      st.score += grade === "PERFECT" ? SCORE_PERFECT : SCORE_GOOD;
      st.texts.push({ text: grade, pos: st.center, color: GRADE_COLORS[grade], born: now, scale: grade === "PERFECT" ? 1.4 : 1.2 });
      if (punch) requestPrediction(st, st.type, st.side, segmentClip(punch.startT, punch.endT) ?? snapshotClip(st.spawnTs));
      else st.pending = { type: st.type, side: st.side, spawnTs: st.spawnTs, deadline: now + PENDING_CLASSIFY_S };
    } else if (missed) {
      st.timing.miss += 1;
      st.texts.push({ text: "MISS", pos: st.center, color: GRADE_COLORS.MISS, born: now, scale: 1.1 });
    }
    if (hit || missed) {
      Object.assign(st, { phase: "idle", center: null, type: null, side: null, nextSpawn: now + TARGET_RESPAWN_DELAY_S });
    }
  }

  // 4. Defense (drawn first so the punch target stays on top of an attack band).
  const outcome = defense.update(lms, now, st.reserved);
  if (outcome === "Hit") {
    st.score = Math.max(0, st.score - SCORE_HIT_PENALTY);
    st.vignetteUntil = now + HIT_VIGNETTE_S;
    st.texts.push({ text: "HIT!", pos: defense.lastEvent.pos, color: [255, 60, 60], born: now, scale: 1.2 });
  } else if (outcome) {
    st.score += SCORE_DEFENSE_BONUS;
    st.texts.push({ text: outcome === "Dodge" ? "DODGED!" : "BLOCKED!", pos: defense.lastEvent.pos, color: WHITE, born: now, scale: 1.2 });
  }
  if (st.phase === "active") drawOsuTarget(st.center, visualR, PUNCH_COLORS[st.type], (now - st.spawnTs) / APPROACH_DURATION_S);
}

function drawRoundText(now) {
  const visualR = TARGET_RADIUS_PX * S * TARGET_VISUAL_SCALE;
  if (st.phase === "active") {
    fitFont(st.side, visualR * 1.1, visualR * 0.7);
    textCentered(st.side, mx(st.center[0]), st.center[1], WHITE);
  }
  if (st.phase === "callout") drawCallout(callout(st.type, st.side), PUNCH_COLORS[st.type], now - st.calloutTs);
  st.texts = st.texts.filter((t) => now - t.born <= FLOATING_TEXT_S);
  for (const t of st.texts) {
    const k = (now - t.born) / FLOATING_TEXT_S, pop = 1 + 0.25 * Math.max(0, 1 - k / 0.2);
    ctx.font = `bold ${32 * S * t.scale * pop}px system-ui, sans-serif`;
    textCentered(t.text, mx(t.pos[0]), t.pos[1] - 60 * S * k, t.color, 1 - k * k);
  }
  if (now < st.vignetteUntil) drawVignette((st.vignetteUntil - now) / HIT_VIGNETTE_S);
  if (st.prediction && now <= st.prediction.until) {
    ctx.font = `bold ${22 * S}px system-ui, sans-serif`;
    const tw = ctx.measureText(st.prediction.text).width, pad = 12 * S, y = H * 0.12, th = 22 * S;
    ctx.fillStyle = "rgba(0,0,0,0.85)";
    ctx.fillRect(W / 2 - tw / 2 - pad, y - th / 2 - pad, tw + 2 * pad, th + 2 * pad);
    ctx.strokeStyle = rgba(st.prediction.color);
    ctx.lineWidth = 2;
    ctx.strokeRect(W / 2 - tw / 2 - pad, y - th / 2 - pad, tw + 2 * pad, th + 2 * pad);
    textCentered(st.prediction.text, W / 2, y, st.prediction.color, 1, false);
  }
  drawHud(st.score, MAX_RUNTIME - (now - st.start));
}

// Debug overlay for the punch detector: peak circles in camera coords (call under the mirror transform)…
function drawPunchPeaks(now) {
  for (const [side, tr] of Object.entries(detector.trackers)) {
    tr.punchPeaks = tr.punchPeaks.filter((p) => now - p.t < PEAK_DISPLAY_S);
    for (const p of tr.punchPeaks) { const [x, y] = toPx(p.pos); strokeCircle(x, y, 30 * S, rgba(SIDE_COLORS[side]), 3); }
  }
}

// …and per-hand state text in screen coords.
function drawPunchState() {
  ctx.save();
  ctx.font = `bold ${18 * S}px system-ui, sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  Object.entries(detector.trackers).forEach(([side, tr], i) => {
    const state = tr.guardReady ? tr.state : `learning guard ${tr.guardBuf.length}/${GUARD_READY_SAMPLES}`;
    const text = `${side}: ${state}  ext ${tr.lastExt.toFixed(2)}  spd ${tr.lastSpeed.toFixed(1)}`;
    ctx.fillStyle = rgba(SIDE_COLORS[side]);
    ctx.fillText(text, 20 * S, H * 0.2 + 28 * S * i);
  });
  ctx.restore();
}

// Same as the end of the Python script: print the totals, release the camera, close the view.
function endRound() {
  mode = "over";
  const d = defense.stats, t = st.timing;
  console.log(`Final score: ${st.score}  |  Timing - Perfect: ${t.perfect}, Good: ${t.good}, Miss: ${t.miss}`);
  console.log(`Defense stats - Blocked: ${d.blocked}, Dodged: ${d.dodged}, Hit: ${d.hit}`);
  // VERSUS: the result card takes the place of a cleared canvas, and the camera stays up for a rematch.
  if (GAME_MODE === "versus") return;
  video.srcObject.getTracks().forEach((track) => track.stop());
  ctx.clearRect(0, 0, W, H);
}

// ── Main loop ────────────────────────────────────────────────────────────────
function loop() {
  // VERSUS: keep drawing the result card. Solo still stops the loop when the round ends.
  if (mode === "over" && GAME_MODE !== "versus") return;
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;   // only new camera frames
  lastVideoTime = video.currentTime;
  const now = performance.now() / 1000;
  const res = landmarker.detectForVideo(video, performance.now());
  const lms = res.landmarks[0] ?? null;
  if (res.worldLandmarks[0]) {
    worldBuf.push({ t: now, skel: normalizeSkeleton(res.worldLandmarks[0]) });
    if (worldBuf.length > WORLD_BUFFER_FRAMES) worldBuf.shift();
  }
  const punchEvents = lms ? detector.update(now, lms) : [];
  // VERSUS: the opponent fills the frame. Solo keeps the mirrored camera below.
  if (GAME_MODE === "versus" && versus && net) {
    const liveScore = mode === "countdown" || !st ? 0 : st.score;
    versus.tick(now, liveScore, mode === "playing");
    if (lms && versus.connected) net.sendPose(net.netNow(), lms);
    drawVersusFrame(lms, now, punchEvents);
    return;
  }

  // Shapes are drawn in camera coordinates under a mirror transform; text after, at mirrored x.
  ctx.setTransform(-1, 0, 0, 1, W, 0);
  ctx.drawImage(video, 0, 0, W, H);
  if (lms) {
    // Like mp_drawing.draw_landmarks: skip low-visibility or off-frame points (their guesses jitter).
    const shown = (l) => (l.visibility ?? 1) >= 0.5 && l.x >= 0 && l.x <= 1 && l.y >= 0 && l.y <= 1;
    drawing.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS.filter((c) => shown(lms[c.start]) && shown(lms[c.end])),
                           { color: "#f5f5f5", lineWidth: 2 });
    drawing.drawLandmarks(lms.filter(shown), { color: "#ff3030", radius: 3 });
  }
  if (SHOW_PUNCH_DEBUG) drawPunchPeaks(now);
  if (mode === "playing") stepRound(lms, now, punchEvents);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (SHOW_PUNCH_DEBUG) drawPunchState();

  if (mode === "countdown") {
    const left = PRESTART_COUNTDOWN_S - (now - modeStart);
    if (left <= 0) { mode = "playing"; newRound(now); }
    else { fitFont("3", W * 0.3, H * 0.3); textCentered(String(Math.ceil(left)), W / 2, H / 2, WHITE); }
  }
  if (mode === "playing") {
    drawRoundText(now);
    if (now - st.start >= MAX_RUNTIME) endRound();
  }
}

// VERSUS: a detected punch from either hand is one head-circle attack. No callout or target.
function stepVersus(lms, now, events) {
  if (events.some((e) => e.type === "punch")) versus.notePunch(now);
  const outcome = defense.update(lms, now, []);
  if (outcome) applyVersusDefense(outcome, now);
}

// The circle uses the same selfie mapping on the full frame and in the thumbnail,
// so the picture the player dodges is the circle collision tests on their head.
function drawMappedJab(atk, now, rect) {
  if (!atk?.circle) return;
  const [cx, cy, r] = atk.circle;
  const [x, y] = mapCamPoint(cx, cy, W, H, rect);
  const rr = Math.max(4, mapCamRadius(r, W, rect));
  const phase = atk.phase(now);
  if (phase === "done") return;
  if (phase === "warning") {
    if (atk.flashOn(now)) fillCircle(x, y, rr, rgba(ATTACK_COLOR, ATTACK_WARNING_ALPHA));
    strokeCircle(x, y, rr, rgba(WHITE, 0.75), Math.max(2, rr / 18));
  } else {
    fillCircle(x, y, rr, rgba(ATTACK_COLOR, 0.9));
    strokeCircle(x, y, rr * 0.55, rgba(WHITE, 0.9), Math.max(2, rr / 16));
    strokeCircle(x, y, rr, rgba(WHITE, 0.95), Math.max(3, rr / 12));
  }
}

function drawLocalPip(lms, now) {
  const pip = pipRect(W, H);
  ctx.save();
  ctx.beginPath();
  ctx.rect(pip.x, pip.y, pip.w, pip.h);
  ctx.clip();
  ctx.setTransform(-pip.w / W, 0, 0, pip.h / H, pip.x + pip.w, pip.y);
  ctx.drawImage(video, 0, 0, W, H);
  if (lms) {
    const shown = (l) => (l.visibility ?? 1) >= 0.5 && l.x >= 0 && l.x <= 1 && l.y >= 0 && l.y <= 1;
    drawing.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS.filter((c) => shown(lms[c.start]) && shown(lms[c.end])),
                           { color: "#f5f5f5", lineWidth: 2 });
    drawing.drawLandmarks(lms.filter(shown), { color: "#ff3030", radius: 2 });
  }
  ctx.restore();
  if (defense.attack) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(pip.x, pip.y, pip.w, pip.h);
    ctx.clip();
    drawMappedJab(defense.attack, now, pip);
    ctx.restore();
  }
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 2;
  ctx.strokeRect(pip.x + 1, pip.y + 1, pip.w - 2, pip.h - 2);
  ctx.font = `bold ${14 * S}px system-ui, sans-serif`;
  ctx.fillStyle = "#fff";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("You", pip.x + 8 * S, pip.y + 6 * S);
  ctx.restore();
}

function drawVersusFrame(lms, now, punchEvents) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (mode === "playing") stepVersus(lms, now, punchEvents);
  ctx.fillStyle = "#0c0e14";
  ctx.fillRect(0, 0, W, H);
  versus.drawStage(ctx, W, H, S, now);
  if (defense.attack) drawMappedJab(defense.attack, now, { x: 0, y: 0, w: W, h: H });
  if (defense.queue.length) {
    ctx.font = `bold ${22 * S}px system-ui, sans-serif`;
    textCentered("ATTACK QUEUED", W / 2, H * 0.3, [255, 180, 60]);
  }
  if (defense.banner && now < defense.banner.until) {
    ctx.font = `bold ${52 * S}px system-ui, sans-serif`;
    textCentered("INCOMING", W / 2, H * 0.2, ATTACK_COLOR);
  }
  drawLocalPip(lms, now);
  if (mode === "countdown") {
    const left = versus.t0 - net.netNow();
    if (left <= 0) { mode = "playing"; newRound(versus.t0 - net.clockOffset); }
    else { fitFont("3", W * 0.3, H * 0.3); textCentered(String(Math.ceil(left)), W / 2, H / 2, WHITE); }
  }
  if (mode === "playing") {
    drawVersusPlayText(now);
    if (versus.timeUp(now, st.start)) {
      versus.sendState(st.score);
      finishVersus(verdictTitle(judge(versus.hp, versus.oppHp, st.score, versus.oppScore)));
    }
  }
  if (mode === "over") versus.drawResult(ctx, W, H, S);
}

function drawVersusPlayText(now) {
  st.texts = st.texts.filter((t) => now - t.born <= FLOATING_TEXT_S);
  for (const t of st.texts) {
    const k = (now - t.born) / FLOATING_TEXT_S, pop = 1 + 0.25 * Math.max(0, 1 - k / 0.2);
    ctx.font = `bold ${36 * S * (t.scale || 1) * pop}px system-ui, sans-serif`;
    textCentered(t.text, t.pos[0], t.pos[1] - 50 * S * k, t.color, 1 - k * k);
  }
  if (now < st.vignetteUntil) drawVignette((st.vignetteUntil - now) / HIT_VIGNETTE_S);
  versus.drawHud(ctx, W, H, S, st.score, versus.remaining(now, st.start));
}

function versusScreenPos(camPos) {
  return mapCamPoint(camPos[0], camPos[1], W, H, { x: 0, y: 0, w: W, h: H });
}

function showPanel(html, buttonText) {
  lobbyEl.hidden = true;
  roomEl.hidden = true;
  statusEl.hidden = false;
  statusEl.innerHTML = html;
  startBtn.textContent = buttonText || "";
  startBtn.hidden = !buttonText;
  panel.hidden = false;
}

function showLobby() {
  statusEl.hidden = true;
  startBtn.hidden = true;
  roomEl.hidden = true;
  lobbyEl.hidden = false;
  panel.hidden = false;
  const ready = mode === "ready";
  for (const el of [soloBtn, createBtn, joinBtn, codeInput]) el.disabled = !ready;
  fightBotBtn.hidden = !BOT_MODE;
  botNote.hidden = !BOT_MODE;
  fightBotBtn.disabled = !ready;
}

function updateRoom() {
  if (!versus) return;
  const bot = versus.roomCode === "BOT";
  codeLine.hidden = bot;
  botLine.hidden = !bot;
  if (!bot) codeValue.textContent = versus.roomCode || "-----";
  roomStatus.textContent = versus.connected ? `Opponent connected: ${versus.oppName}` : "Waiting for opponent…";
  roomPing.textContent = versus.pingMs == null ? "" : `ping ${versus.pingMs} ms`;
  readyYou.textContent = `You: ${versus.localReady ? "Ready" : "not ready"}`;
  readyOpp.textContent = `Opponent: ${versus.oppReady ? "Ready" : "not ready"}`;
  readyBtn.textContent = versus.localReady ? "Cancel ready" : "Ready";
  const both = versus.connected && versus.localReady && versus.oppReady;
  const host = versus.role === "host";
  startVersusBtn.hidden = !(host && both);
  waitHost.hidden = host || !both;
}

function playerName() {
  return (nameInput.value || "Player").trim().slice(0, 16) || "Player";
}

function makeNet() {
  const n = new Net({ PeerCtor: globalThis.Peer, name: playerName(), W, H });
  return n;
}

function attachVersus(n, role, roomCode) {
  GAME_MODE = "versus";
  net = n;
  versus = new Versus(n, role, {
    onOpen() { updateRoom(); },
    onReady() { updateRoom(); },
    onLatency() { updateRoom(); },
    onStart(msg) { beginVersusCountdown(msg.t0); },
    onAttack(msg) {
      if (mode !== "playing" && mode !== "countdown") return;
      defense.enqueue({ id: msg.id });
    },
    onAttackResult(msg) { onVersusAttackResult(msg); },
    onKo(msg) { onVersusKo(msg); },
    onBothRematch() { onBothRematch(); },
    onLeft() { onOpponentLeft(); },
    onError(msg) { onNetError(msg); },
    onStaleDrop() { onOpponentLeft(); },
  });
  versus.myName = playerName();
  versus.roomCode = roomCode;
  defense.onDropped = (meta) => {
    if (meta && meta.id != null && net) net.sendEvent("attackResult", { id: meta.id, outcome: "dropped", dmg: 0 });
  };
}

function showRoom(code) {
  statusEl.hidden = true;
  startBtn.hidden = true;
  lobbyEl.hidden = true;
  roomEl.hidden = false;
  panel.hidden = false;
  if (versus) versus.roomCode = code;
  updateRoom();
}

function teardownVersus() {
  const n = net, v = versus;
  if (n) n.ignoreClose = true;
  try { v?.detach(); } catch { /* listener already gone */ }
  try { n?.close(); } catch { /* already closed */ }
  net = null;
  versus = null;
  defense.onDropped = null;
  defense.queue = [];
  defense.attack = null;
  defense.banner = null;
  GAME_MODE = "solo";
  if (mode === "countdown" || mode === "playing" || mode === "over") mode = "ready";
  showLobby();
}

function onOpponentLeft() {
  if (!versus) return;
  const forfeited = mode === "playing";
  teardownVersus();
  lobbyMsg.textContent = forfeited ? "Opponent left. You win by forfeit." : "Opponent left";
}

function onNetError(detail) {
  const message = detail?.message || "Connection failed.";
  if (versus || net) teardownVersus();
  else showLobby();
  lobbyMsg.textContent = message;
}

function beginVersusCountdown(t0) {
  if (!versus || !net || !Number.isFinite(t0)) return;
  versus.t0 = t0;
  versus.resetRound();
  defense.queue = [];
  defense.attack = null;
  defense.banner = null;
  defense.lastEvent = null;
  panel.hidden = true;
  mode = "countdown";
}

function finishVersus(title, reason = "") {
  if (!versus || versus._finished || !st) return;
  versus._finished = true;
  versus.result = {
    title, reason,
    myHp: versus.hp, oppHp: versus.oppHp,
    myScore: st.score, oppScore: versus.oppScore,
  };
  net?.setLive?.(false);
  endRound();
}

function applyVersusDefense(outcome, now) {
  const meta = defense.lastEvent?.meta;
  const pos = versusScreenPos(defense.lastEvent.pos);
  let dmg = 0;
  if (outcome === "Dodge") {
    st.score += SCORE_DEFENSE_BONUS;
    st.texts.push({ text: "DODGED!", pos, color: WHITE, born: now, scale: 1.2 });
  } else if (outcome === "Block") {
    dmg = blockDamage();
    st.score += BLOCK_SCORE;
    st.texts.push({ text: "BLOCKED!", pos, color: WHITE, born: now, scale: 1.2 });
  } else if (outcome === "Hit") {
    dmg = hitDamage();
    st.score = Math.max(0, st.score - SCORE_HIT_PENALTY);
    st.vignetteUntil = now + HIT_VIGNETTE_S;
    st.texts.push({ text: "HIT!", pos, color: [255, 60, 60], born: now, scale: 1.2 });
  }
  versus.applyDamage(dmg);
  if (meta && meta.id != null) net.sendEvent("attackResult", { id: meta.id, outcome, dmg });
  versus.sendState(st.score);
  if (versus.hp <= 0) {
    net.sendEvent("ko", { loser: "me" });
    finishVersus("YOU LOSE", "KO");
  }
}

function onVersusAttackResult(msg) {
  if (!versus) return;
  const entry = versus.takeOutgoing(msg.id);
  if (!entry || msg.outcome === "dropped" || !st) return;
  if (msg.outcome === "Hit") st.score += LANDED_BONUS;
  const label = msg.outcome === "Hit" ? `LANDED -${msg.dmg}`
    : msg.outcome === "Dodge" ? "DODGED"
    : msg.outcome === "Block" ? "BLOCKED" : null;
  if (label) {
    st.texts.push({
      text: label,
      pos: [W * 0.5, H * 0.62],
      color: msg.outcome === "Hit" ? GRADE_COLORS.PERFECT : WHITE,
      born: performance.now() / 1000,
      scale: 1.05,
    });
  }
  versus.sendState(st.score);
}

function onVersusKo(msg) {
  if (!versus || versus._finished) return;
  if (msg.loser === "me") { versus.oppHp = 0; finishVersus("YOU WIN", "KO"); }
  else { versus.hp = 0; finishVersus("YOU LOSE", "KO"); }
}

function onBothRematch() {
  if (!versus || versus.role !== "host" || !net) return;
  const t0 = net.netNow() + 3.5;
  net.sendEvent("start", { t0 });
  beginVersusCountdown(t0);
}

function startBot() {
  const bot = new BotOpponent();
  attachVersus(bot, "host", "BOT");
  showRoom("BOT");
  bot.start();
}

soloBtn.addEventListener("click", () => {
  if (mode !== "ready") return;
  GAME_MODE = "solo";
  panel.hidden = true;
  mode = "countdown";
  modeStart = performance.now() / 1000;
});

createBtn.addEventListener("click", async () => {
  if (mode !== "ready" || connecting) return;
  if (typeof globalThis.Peer !== "function") { lobbyMsg.textContent = "PeerJS failed to load."; return; }
  connecting = true;
  lobbyMsg.textContent = "Creating room…";
  const n = makeNet();
  attachVersus(n, "host", "");
  try {
    const code = await n.host();
    showRoom(code);
    lobbyMsg.textContent = "";
  } catch (e) {
    teardownVersus();
    lobbyMsg.textContent = e?.message || "Could not create room.";
  } finally { connecting = false; }
});

joinBtn.addEventListener("click", async () => {
  if (mode !== "ready" || connecting) return;
  if (typeof globalThis.Peer !== "function") { lobbyMsg.textContent = "PeerJS failed to load."; return; }
  const code = codeInput.value.toUpperCase().replace(/\s+/g, "").slice(0, 5);
  if (code.length !== 5) { lobbyMsg.textContent = "Enter a 5-character code."; return; }
  connecting = true;
  lobbyMsg.textContent = "Joining room…";
  const n = makeNet();
  attachVersus(n, "guest", code);
  try {
    await n.join(code);
    showRoom(code);
    lobbyMsg.textContent = "";
  } catch (e) {
    teardownVersus();
    lobbyMsg.textContent = e?.message || "Room not found.";
  } finally { connecting = false; }
});

fightBotBtn.addEventListener("click", () => { if (mode === "ready") startBot(); });

readyBtn.addEventListener("click", () => {
  if (!versus || !net) return;
  versus.localReady = !versus.localReady;
  net.sendEvent("ready", { ready: versus.localReady });
  updateRoom();
});

startVersusBtn.addEventListener("click", () => {
  if (!versus || !net || versus.role !== "host" || !versus.localReady || !versus.oppReady) return;
  const t0 = net.netNow() + 3.5;
  net.sendEvent("start", { t0 });
  beginVersusCountdown(t0);
});

leaveBtn.addEventListener("click", () => teardownVersus());

copyBtn.addEventListener("click", async () => {
  const code = codeValue.textContent.trim();
  try {
    await navigator.clipboard.writeText(code);
    copyBtn.textContent = "Copied";
    setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
  } catch { copyBtn.textContent = code; }
});

codeInput.addEventListener("input", () => {
  const next = codeInput.value.toUpperCase().slice(0, 5);
  if (next !== codeInput.value) codeInput.value = next;
});
codeInput.addEventListener("keydown", (e) => { if (e.key === "Enter") joinBtn.click(); });

canvas.addEventListener("pointerdown", (ev) => {
  // VERSUS: Rematch and Leave are drawn on the result card.
  if (GAME_MODE !== "versus" || mode !== "over" || !versus) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
  const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
  const hit = versus.hitButton(x, y);
  if (!hit) return;
  if (hit.id === "rematch") { versus.requestRematch(); versus.drawResult(ctx, W, H, S); }
  if (hit.id === "leave") teardownVersus();
});

async function init() {
  try {
    showPanel("Starting camera…");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540, frameRate: 30 }, audio: false });
    video.srcObject = stream;
    await video.play();
    W = canvas.width = video.videoWidth;
    H = canvas.height = video.videoHeight;
    S = W / BASE_W;
    drawing = new DrawingUtils(ctx);

    showPanel("Loading pose and punch models…");
    const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
    const makeLandmarker = (delegate) => PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate },
      runningMode: "VIDEO", numPoses: 1,
      minPoseDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    [landmarker, session] = await Promise.all([
      makeLandmarker("GPU").catch(() => makeLandmarker("CPU")),
      ort.InferenceSession.create("punch_model.onnx", { executionProviders: ["wasm"] }),
    ]);
    mode = "ready";
    // VERSUS: room buttons stay disabled until the camera and both models are ready. ?bot=1 opens the bot room.
    if (BOT_MODE) startBot();
    else showLobby();
    loop();
  } catch (e) {
    console.error(e);
    const msg = e?.name === "NotAllowedError" ? "Camera access was blocked. Allow it in the browser and reload."
              : e?.name === "NotFoundError" ? "No camera found." : `Could not start: ${e?.message ?? e}`;
    showPanel(`<p>${msg}</p>`);
  }
}

init();
