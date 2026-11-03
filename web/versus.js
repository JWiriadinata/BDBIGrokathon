// Versus rules, opponent pose buffer, damage, and the one-tab dev bot.
// No DOM and no `window` access: drawing takes a canvas context argument, and
// the bot is an EventTarget with the same surface game.js uses for Net.

export const MAX_HP = 100;
export const VERSUS_ROUND_S = 60;
export const VERSUS_RESPAWN_DELAY_S = 1.2;
export const DAMAGE = { jab: 8, hook: 12, uppercut: 15 };
export const PERFECT_MULT = 1.5;
export const GOOD_MULT = 1;
export const BLOCK_DAMAGE_FACTOR = 0.5;
export const POSE_HZ = 20;
export const OPP_RENDER_DELAY_S = 0.10;
export const POSE_STALE_WARN_S = 3;
export const POSE_STALE_DROP_S = 10;
export const MAX_QUEUED_ATTACKS = 2;
export const OPP_PANEL_W_FRAC = 0.28;
// Reserved: the attack kind is the called punch, not the CNN label. Leave false.
export const USE_PREDICTION_FOR_ATTACK = false;
export const OPP_HOLD_S = 0.4;
export const OPP_FADE_S = 0.4;
export const OUTGOING_TTL_S = 6;
export const BLOCK_SCORE = 25;
export const LANDED_BONUS = 100;
export const DEFENSE_BONUS = 50;
export const POSE_BUF_MAX = 30;

const CYAN = [0, 255, 255];

export function hitDamage(kind, mult) {
  const base = DAMAGE[kind];
  if (base == null || !Number.isFinite(mult)) return 0;
  return Math.round(base * mult);
}

export function blockDamage(kind, mult) {
  const base = DAMAGE[kind];
  if (base == null || !Number.isFinite(mult)) return 0;
  return Math.round(base * mult * BLOCK_DAMAGE_FACTOR);
}

export function applyHp(hp, dmg) {
  return Math.max(0, hp - Math.max(0, dmg || 0));
}

export function gradeMult(grade) {
  return grade === "PERFECT" ? PERFECT_MULT : GOOD_MULT;
}

// Higher HP wins. Equal HP falls through to score, then a draw.
export function judge(myHp, oppHp, myScore, oppScore) {
  if (myHp !== oppHp) return myHp > oppHp ? "win" : "lose";
  if (myScore !== oppScore) return myScore > oppScore ? "win" : "lose";
  return "draw";
}

export function verdictTitle(verdict) {
  return verdict === "win" ? "YOU WIN" : verdict === "lose" ? "YOU LOSE" : "DRAW";
}

// Returns false when the queue is full (caller sends outcome "dropped").
export function enqueueAttack(queue, item, max = MAX_QUEUED_ATTACKS) {
  if (queue.length >= max) return false;
  queue.push(item);
  return true;
}

// `buf` is sorted by t. A query between two samples is the linear midpoint of
// every component. Past the newest sample the newest pose is held for OPP_HOLD_S,
// then alpha falls to 0 across OPP_FADE_S.
export function sampleOpponent(buf, renderT) {
  if (!buf || !buf.length) return { k: null, alpha: 0, held: false };
  const newest = buf[buf.length - 1];
  if (renderT > newest.t) {
    const age = renderT - newest.t;
    if (age <= OPP_HOLD_S) return { k: newest.k, alpha: 1, held: true };
    const alpha = Math.max(0, 1 - (age - OPP_HOLD_S) / OPP_FADE_S);
    return { k: newest.k, alpha, held: true };
  }
  if (renderT <= buf[0].t || buf.length === 1) return { k: buf[0].k, alpha: 1, held: false };
  let i = 0;
  for (; i < buf.length - 1; i++) {
    if (buf[i].t <= renderT && renderT <= buf[i + 1].t) break;
  }
  const a = buf[i], b = buf[i + 1];
  const span = b.t - a.t;
  const u = span <= 1e-9 ? 0 : (renderT - a.t) / span;
  const k = a.k.map((v, j) => v + ((b.k[j] ?? 0) - v) * u);
  return { k, alpha: 1, held: false };
}

function css([r, g, b], a = 1) {
  return `rgba(${r | 0},${g | 0},${b | 0},${a})`;
}

function hpColor(frac) {
  const f = Math.max(0, Math.min(1, frac));
  return [Math.round(230 * (1 - f) + 30 * f), Math.round(40 * (1 - f) + 200 * f), 50];
}

function emitOn(target, type, detail) {
  target.dispatchEvent(new CustomEvent(type, { detail }));
}

// Packed landmark slots inside the 45-number pose: nose, shoulders, elbows, wrists, hips.
const NOSE_I = 0, LS_I = 1, RS_I = 2, LE_I = 3, RE_I = 4, LW_I = 5, RW_I = 6, LH_I = 13, RH_I = 14;

export class Versus {
  constructor(net, role, hooks = {}) {
    this.net = net;
    this.role = role;
    this.hooks = hooks;
    this.hp = MAX_HP;
    this.oppHp = MAX_HP;
    this.oppScore = 0;
    this.myName = net.name || "Player";
    this.oppName = "Opponent";
    this.pingMs = null;
    this.oppBuf = [];
    this.outgoing = new Map();
    this.attackSeq = 0;
    this.connected = false;
    this.localReady = false;
    this.oppReady = false;
    this.localRematch = false;
    this.oppRematch = false;
    this._rematchStarted = false;
    this.result = null;
    this._finished = false;
    this.lastPoseAt = 0;
    this.pauseBegan = null;
    this.pausedAccum = 0;
    this.stale = false;
    this._dropped = false;
    this.lastStateSent = 0;
    this.lastSentScore = null;
    this.lastSentHp = null;
    this.t0 = 0;
    this.buttons = [];
    this._prevDraw = null;
    this._listeners = {
      open: (e) => this._onOpen(e.detail),
      pose: (e) => this.pushPose(e.detail),
      attack: (e) => hooks.onAttack?.(e.detail),
      attackResult: (e) => hooks.onAttackResult?.(e.detail),
      state: (e) => this._onState(e.detail),
      ready: (e) => this._onReady(e.detail),
      start: (e) => { if (this.role !== "host") hooks.onStart?.(e.detail); },
      ko: (e) => hooks.onKo?.(e.detail),
      rematch: () => this._onRematch(),
      left: () => hooks.onLeft?.(),
      error: (e) => hooks.onError?.(e.detail),
      latency: (e) => {
        this.pingMs = e.detail?.rtt ?? null;
        hooks.onLatency?.(this.pingMs);
      },
    };
    for (const [type, fn] of Object.entries(this._listeners)) net.addEventListener(type, fn);
  }

  detach() {
    for (const [type, fn] of Object.entries(this._listeners)) {
      this.net.removeEventListener(type, fn);
    }
  }

  _onOpen(info) {
    this.connected = true;
    this.oppName = info?.name || "Opponent";
    this.lastPoseAt = performance.now() / 1000;
    this.hooks.onOpen?.(info);
  }

  _onReady(detail) {
    this.oppReady = !!detail?.ready;
    this.hooks.onReady?.(this.oppReady);
  }

  _onState(detail) {
    if (Number.isFinite(detail?.hp)) this.oppHp = Math.max(0, Math.min(MAX_HP, detail.hp));
    if (Number.isFinite(detail?.score)) this.oppScore = Math.max(0, detail.score);
    this.hooks.onState?.();
  }

  _onRematch() {
    this.oppRematch = true;
    this.tryRematch();
  }

  pushPose({ t, k } = {}) {
    if (!Array.isArray(k) || k.length !== 45 || !Number.isFinite(t)) return;
    const last = this.oppBuf[this.oppBuf.length - 1];
    if (last && t < last.t) return;
    this.oppBuf.push({ t, k });
    if (this.oppBuf.length > POSE_BUF_MAX) this.oppBuf.shift();
    this.lastPoseAt = performance.now() / 1000;
  }

  // Called punch only. USE_PREDICTION_FOR_ATTACK stays false, so the CNN label is not read.
  launchAttack(type, grade) {
    if (grade !== "PERFECT" && grade !== "GOOD") return;
    if (!DAMAGE[type]) return;
    const mult = gradeMult(grade);
    const id = ++this.attackSeq;
    const sentAt = this.net.netNow();
    this.outgoing.set(id, { id, kind: type, mult, grade, localT: performance.now() / 1000 });
    this.net.sendEvent("attack", { id, kind: type, mult, sentAt });
  }

  takeOutgoing(id) {
    const entry = this.outgoing.get(id);
    if (entry) this.outgoing.delete(id);
    return entry || null;
  }

  applyDamage(dmg) {
    this.hp = applyHp(this.hp, dmg);
    return this.hp;
  }

  sendState(score) {
    if (!this.net || this.net.closed) return;
    const safe = Math.max(0, score || 0);
    this.net.sendEvent("state", { hp: this.hp, score: safe });
    this.lastStateSent = performance.now() / 1000;
    this.lastSentScore = safe;
    this.lastSentHp = this.hp;
  }

  resetRound() {
    this.hp = MAX_HP;
    this.oppHp = MAX_HP;
    this.oppScore = 0;
    this.outgoing.clear();
    this.pausedAccum = 0;
    this.pauseBegan = null;
    this.stale = false;
    this._dropped = false;
    this.result = null;
    this._finished = false;
    this.localRematch = false;
    this.oppRematch = false;
    this._rematchStarted = false;
    this.buttons = [];
    this.lastSentScore = null;
    this.lastSentHp = null;
  }

  requestRematch() {
    if (this.localRematch || !this.net) return;
    this.localRematch = true;
    this.net.sendEvent("rematch", {});
    this.tryRematch();
  }

  tryRematch() {
    if (!this.localRematch || !this.oppRematch || this._rematchStarted) return;
    this._rematchStarted = true;
    this.hooks.onBothRematch?.();
  }

  tick(now, score, playing) {
    const newest = this.oppBuf[this.oppBuf.length - 1];
    if (newest) {
      const cutoff = newest.t - 2;
      while (this.oppBuf.length > 2 && this.oppBuf[0].t < cutoff) this.oppBuf.shift();
    }
    while (this.oppBuf.length > POSE_BUF_MAX) this.oppBuf.shift();
    for (const [id, entry] of this.outgoing) {
      if (now - entry.localT > OUTGOING_TTL_S) this.outgoing.delete(id);
    }
    if (this.connected && (now - this.lastStateSent >= 1 || score !== this.lastSentScore || this.hp !== this.lastSentHp)) {
      this.sendState(score);
    }
    const age = this.lastPoseAt ? now - this.lastPoseAt : 0;
    const missing = this.connected && this.lastPoseAt > 0 && age > POSE_STALE_WARN_S;
    this.stale = missing && age <= POSE_STALE_DROP_S;
    if (playing && this.stale) {
      if (this.pauseBegan == null) this.pauseBegan = now;
    } else if (this.pauseBegan != null) {
      this.pausedAccum += now - this.pauseBegan;
      this.pauseBegan = null;
    }
    if (missing && age > POSE_STALE_DROP_S && !this._dropped) {
      this._dropped = true;
      this.hooks.onStaleDrop?.();
    }
  }

  remaining(now, startLocal) {
    const endNow = this.pauseBegan ?? now;
    return VERSUS_ROUND_S - (endNow - startLocal - this.pausedAccum);
  }

  timeUp(now, startLocal) {
    if (this.pauseBegan != null || this._finished) return false;
    return this.remaining(now, startLocal) <= 0;
  }

  // Lower-right panel, unmirrored: their camera x is used as-is, so their right
  // hand (image-left) lands on the screen-left of the panel, as if they face us.
  drawOpponent(ctx, W, H, S) {
    const margin = Math.max(12, W / 60);
    const pw = OPP_PANEL_W_FRAC * W;
    const ph = pw * 9 / 16;
    const x = W - margin - pw;
    const y = H - margin - ph - 22 * S;
    ctx.save();
    ctx.fillStyle = "rgba(8,10,16,0.72)";
    ctx.fillRect(x, y, pw, ph);
    ctx.strokeStyle = "rgba(0,255,255,0.45)";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, pw, ph);
    const renderT = this.net.netNow() - OPP_RENDER_DELAY_S;
    const sample = sampleOpponent(this.oppBuf, renderT);
    if (sample.k && sample.alpha > 0.02) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, pw, ph);
      ctx.clip();
      ctx.globalAlpha = sample.alpha;
      this._drawSkeleton(ctx, sample.k, renderT, { x, y, w: pw, h: ph }, S);
      ctx.restore();
      this._prevDraw = { t: renderT, k: sample.k };
    }
    ctx.font = `bold ${14 * S}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#fff";
    ctx.fillText(this.oppName, x + pw / 2, y + ph + 4 * S);
    ctx.restore();
    if (this.stale) this._banner(ctx, W, H, S, "Opponent connection lost");
  }

  _drawSkeleton(ctx, k, renderT, box, S) {
    const pt = (i) => {
      const v = k[i * 3 + 2] ?? 1;
      if (v < 0.35) return null;
      return [box.x + k[i * 3] * box.w, box.y + k[i * 3 + 1] * box.h];
    };
    const prev = this._prevDraw;
    const dt = prev ? renderT - prev.t : 0;
    const speed = (i) => {
      if (!prev || dt <= 1e-4) return 0;
      const dx = k[i * 3] - prev.k[i * 3];
      const dy = k[i * 3 + 1] - prev.k[i * 3 + 1];
      return Math.hypot(dx, dy) / dt;
    };
    const chain = (indices, fast) => {
      ctx.beginPath();
      let on = false;
      for (const i of indices) {
        const p = pt(i);
        if (!p) { on = false; continue; }
        if (!on) { ctx.moveTo(p[0], p[1]); on = true; }
        else ctx.lineTo(p[0], p[1]);
      }
      ctx.strokeStyle = css(CYAN);
      ctx.lineWidth = (fast ? 7 : 3) * S;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
    };
    chain([LS_I, LE_I, LW_I], speed(LW_I) > 0.45);
    chain([RS_I, RE_I, RW_I], speed(RW_I) > 0.45);
    chain([LH_I, RH_I], false);
    chain([LS_I, RS_I], false);
    const nose = pt(NOSE_I);
    if (nose) {
      ctx.beginPath();
      ctx.fillStyle = css(CYAN, 0.9);
      ctx.arc(nose[0], nose[1], Math.max(4, box.w * 0.045), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawHud(ctx, W, H, S, score, remaining) {
    const margin = Math.max(12, W / 60);
    const barW = W * 0.26, barH = Math.max(14, 16 * S);
    const myFrac = this.hp / MAX_HP, oppFrac = this.oppHp / MAX_HP;
    this._bar(ctx, margin, margin, barW, barH, myFrac, "left");
    this._bar(ctx, W - margin - barW, margin, barW, barH, oppFrac, "right");
    ctx.font = `bold ${15 * S}px system-ui, sans-serif`;
    ctx.textBaseline = "top";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "left";
    ctx.fillText(`${this.myName}   ${Math.round(this.hp)}`, margin, margin + barH + 4 * S);
    ctx.fillText(`Score ${Math.max(0, score | 0)}`, margin, margin + barH + 22 * S);
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(this.oppHp)}   ${this.oppName}`, W - margin, margin + barH + 4 * S);
    ctx.fillText(`Score ${Math.max(0, this.oppScore | 0)}`, W - margin, margin + barH + 22 * S);
    const secs = Math.max(0, Math.ceil(remaining));
    const timeText = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
    ctx.textAlign = "center";
    ctx.font = `bold ${28 * S}px system-ui, sans-serif`;
    ctx.fillStyle = remaining <= 5 ? "#ff5050" : "#fff";
    ctx.fillText(timeText, W / 2, margin);
    if (this.pingMs != null) {
      ctx.font = `bold ${14 * S}px system-ui, sans-serif`;
      ctx.fillStyle = "#ddd";
      ctx.fillText(`ping ${this.pingMs} ms`, W / 2, margin + 32 * S);
    }
  }

  _bar(ctx, x, y, w, h, frac, side) {
    ctx.fillStyle = "rgba(20,20,20,0.75)";
    ctx.fillRect(x, y, w, h);
    const inner = Math.max(0, w - 4) * Math.max(0, Math.min(1, frac));
    ctx.fillStyle = css(hpColor(frac));
    if (side === "right") ctx.fillRect(x + w - 2 - inner, y + 2, inner, h - 4);
    else ctx.fillRect(x + 2, y + 2, inner, h - 4);
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
  }

  drawResult(ctx, W, H, S) {
    const r = this.result;
    if (!r) return;
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.78)";
    ctx.fillRect(0, 0, W, H);
    const titleColor = r.title === "YOU WIN" ? "#ffe600" : r.title === "YOU LOSE" ? "#ff4040" : "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `bold ${64 * S}px system-ui, sans-serif`;
    ctx.fillStyle = titleColor;
    ctx.fillText(r.title, W / 2, H * 0.32);
    ctx.font = `bold ${22 * S}px system-ui, sans-serif`;
    ctx.fillStyle = "#fff";
    const line = (name, hp, score, y) => {
      ctx.fillText(`${name}    HP ${Math.round(hp)}    Score ${Math.round(score)}`, W / 2, y);
    };
    line(this.myName, r.myHp, r.myScore, H * 0.48);
    line(this.oppName, r.oppHp, r.oppScore, H * 0.56);
    if (r.reason) {
      ctx.font = `bold ${18 * S}px system-ui, sans-serif`;
      ctx.fillStyle = "#ddd";
      ctx.fillText(r.reason, W / 2, H * 0.64);
    }
    const bw = 180 * S, bh = 52 * S, gap = 24 * S, by = H * 0.74;
    const waiting = this.localRematch && !this._rematchStarted;
    const rematch = { id: "rematch", x: W / 2 - bw - gap / 2, y: by, w: bw, h: bh,
                      label: waiting ? "Waiting…" : "Rematch" };
    const leave = { id: "leave", x: W / 2 + gap / 2, y: by, w: bw, h: bh, label: "Leave" };
    this.buttons = [rematch, leave];
    for (const b of this.buttons) {
      ctx.fillStyle = b.id === "rematch" ? "#ffe600" : "rgba(255,255,255,0.14)";
      ctx.beginPath();
      ctx.roundRect(b.x, b.y, b.w, b.h, 10 * S);
      ctx.fill();
      ctx.fillStyle = b.id === "rematch" ? "#111" : "#fff";
      ctx.font = `bold ${20 * S}px system-ui, sans-serif`;
      ctx.fillText(b.label, b.x + b.w / 2, b.y + b.h / 2);
    }
    ctx.restore();
  }

  _banner(ctx, W, H, S, text) {
    ctx.save();
    ctx.font = `bold ${28 * S}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    const tw = ctx.measureText(text).width;
    ctx.fillRect(W / 2 - tw / 2 - 16 * S, H * 0.42 - 22 * S, tw + 32 * S, 44 * S);
    ctx.fillStyle = "#ffb040";
    ctx.fillText(text, W / 2, H * 0.42);
    ctx.restore();
  }

  hitButton(x, y) {
    return this.buttons.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) || null;
  }
}

// A standing skeleton that sways and extends the right arm every few seconds.
// 15 upper-body points, flat [x, y, vis] × 15, in the sender's camera space.
export function botSkeleton(t) {
  const sway = Math.sin(t * 1.4) * 0.02;
  const cycle = ((t % 4.5) + 4.5) % 4.5;
  const punching = cycle > 3.9;
  const p = punching ? Math.sin(((cycle - 3.9) / 0.6) * Math.PI) : 0;
  const pts = [
    [0.50, 0.20],
    [0.40, 0.32], [0.60, 0.32],
    [0.36, 0.46], [0.64 - 0.04 * p, 0.46 - 0.06 * p],
    [0.34, 0.60], [0.66 - 0.12 * p, 0.60 - 0.18 * p],
    [0.33, 0.64], [0.67 - 0.10 * p, 0.64 - 0.14 * p],
    [0.32, 0.63], [0.68 - 0.11 * p, 0.63 - 0.15 * p],
    [0.35, 0.62], [0.65 - 0.09 * p, 0.62 - 0.12 * p],
    [0.44, 0.78], [0.56, 0.78],
  ];
  const k = [];
  for (const [x, y] of pts) k.push(Math.round((x + sway) * 1000) / 1000, Math.round(y * 1000) / 1000, 1);
  return k;
}

// In-page opponent. Same events as Net (open, pose, attack, attackResult, state,
// ready, start, ko, rematch, latency) and the same sendEvent / sendPose / netNow / close.
export class BotOpponent extends EventTarget {
  constructor({ attackEvery = [4000, 8000], poseEvery = 50, replyMs = 40, rng = Math.random } = {}) {
    super();
    this.attackEvery = attackEvery;
    this.poseEvery = poseEvery;
    this.replyMs = replyMs;
    this.rng = rng;
    this.name = "Bot";
    this.clockOffset = 0;
    this.hp = MAX_HP;
    this.score = 0;
    this.live = false;
    this.closed = false;
    this.connected = false;
    this.seq = 0;
    this._pose = null;
    this._atk = null;
  }

  netNow() { return performance.now() / 1000 + this.clockOffset; }

  sendPose() { /* the bot does not look at our pose */ }

  sendEvent(type, payload = {}) {
    if (this.closed) return;
    if (type === "attack") this._answerAttack(payload);
    else if (type === "attackResult") this._onAttackResult(payload);
    else if (type === "rematch") this._onRematch();
    else if (type === "start") this._onStart();
    else if (type === "ko") this.live = false;
    else if (type === "state") { /* the human owns their own totals */ }
  }

  setLive(on) {
    this.live = !!on && !this.closed;
    if (this._atk) { clearTimeout(this._atk); this._atk = null; }
    if (this.live) this._scheduleAttack();
  }

  start() {
    if (this.closed) return;
    this.connected = true;
    emitOn(this, "open", { name: "Bot", W: 960, H: 540 });
    emitOn(this, "ready", { ready: true });
    emitOn(this, "latency", { rtt: 18 });
    emitOn(this, "state", { hp: this.hp, score: this.score });
    this._pose = setInterval(() => {
      if (this.closed) return;
      emitOn(this, "pose", { t: this.netNow(), k: botSkeleton(this.netNow()) });
    }, this.poseEvery);
    if (this._pose.unref) this._pose.unref();
  }

  close() {
    this.closed = true;
    this.live = false;
    this.connected = false;
    clearInterval(this._pose);
    clearTimeout(this._atk);
  }

  _emitState() {
    emitOn(this, "state", { hp: this.hp, score: this.score });
  }

  _answerAttack(payload) {
    const mult = Math.min(1.5, Math.max(0.5, Number(payload.mult) || 1));
    const kind = DAMAGE[payload.kind] != null ? payload.kind : "jab";
    let outcome, dmg;
    if (this.rng() < 0.5) {
      outcome = "Dodge";
      dmg = 0;
      this.score += DEFENSE_BONUS;
    } else if (this.rng() < 0.5) {
      outcome = "Block";
      dmg = blockDamage(kind, mult);
      this.score += BLOCK_SCORE;
    } else {
      outcome = "Hit";
      dmg = hitDamage(kind, mult);
      this.score = Math.max(0, this.score - 50);
    }
    this.hp = applyHp(this.hp, dmg);
    const reply = { id: payload.id, outcome, dmg };
    const timer = setTimeout(() => {
      if (this.closed) return;
      emitOn(this, "attackResult", reply);
      this._emitState();
      if (this.hp <= 0) {
        this.live = false;
        emitOn(this, "ko", { loser: "me" });
      }
    }, this.replyMs);
    if (timer.unref) timer.unref();
  }

  _onAttackResult(payload) {
    if (payload?.outcome === "Hit") this.score += LANDED_BONUS;
    this._emitState();
  }

  _onRematch() {
    this.hp = MAX_HP;
    this.score = 0;
    this.setLive(false);
    emitOn(this, "rematch", {});
    this._emitState();
  }

  _onStart() {
    this.hp = MAX_HP;
    this.score = 0;
    this._emitState();
  }

  _scheduleAttack() {
    if (!this.live || this.closed) return;
    const [lo, hi] = this.attackEvery;
    const delay = lo + this.rng() * Math.max(0, hi - lo);
    this._atk = setTimeout(() => {
      this._atk = null;
      if (!this.live || this.closed) return;
      const kinds = Object.keys(DAMAGE);
      const kind = kinds[Math.floor(this.rng() * kinds.length)] || "jab";
      const mult = this.rng() < 0.25 ? PERFECT_MULT : 1;
      emitOn(this, "attack", { id: ++this.seq, kind, mult, sentAt: this.netNow() });
      this._scheduleAttack();
    }, delay);
    if (this._atk.unref) this._atk.unref();
  }
}
