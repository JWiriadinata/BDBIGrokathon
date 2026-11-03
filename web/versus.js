// Versus rules, opponent pose buffer, damage, and the one-tab dev bot.
// No DOM and no `window` access: drawing takes a canvas context argument, and
// the bot is an EventTarget with the same surface game.js uses for Net.

export const MAX_HP = 100;
export const VERSUS_ROUND_S = 60;
// One attack shape in versus: the head circle. No punch-type table and no timing multiplier.
export const ATTACK_DAMAGE = 10;
export const BLOCK_DAMAGE_FACTOR = 0.5;
// One swing sends one attack. A second punch event inside this window is ignored.
export const PUNCH_ATTACK_COOLDOWN_S = 0.6;
export const POSE_HZ = 20;
export const OPP_RENDER_DELAY_S = 0.10;
export const POSE_STALE_WARN_S = 3;
export const POSE_STALE_DROP_S = 10;
export const MAX_QUEUED_ATTACKS = 2;
export const OPP_PANEL_W_FRAC = 0.28;
export const PIP_W_FRAC = 0.28;
export const OPP_HOLD_S = 0.4;
export const OPP_FADE_S = 0.4;
export const OUTGOING_TTL_S = 6;
export const BLOCK_SCORE = 25;
export const LANDED_BONUS = 100;
export const DEFENSE_BONUS = 50;
export const POSE_BUF_MAX = 30;

const CYAN = [0, 255, 255];

export function hitDamage() {
  return ATTACK_DAMAGE;
}

export function blockDamage() {
  return Math.round(ATTACK_DAMAGE * BLOCK_DAMAGE_FACTOR);
}

export function applyHp(hp, dmg) {
  return Math.max(0, hp - Math.max(0, dmg || 0));
}

// `nextAt` is the local time the player may throw again. A punch inside the window is dropped.
export function acceptPunch(nextAt, now, cooldown = PUNCH_ATTACK_COOLDOWN_S) {
  if (now < nextAt) return { ok: false, nextAt };
  return { ok: true, nextAt: now + cooldown };
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

// Selfie mapping: camera x is mirrored into `rect`, which is either the full frame
// or the picture-in-picture. The same fraction of the rect is the head in both,
// so the dodge circle sits on the head that collision tests.
export function mapCamPoint(camX, camY, frameW, frameH, rect) {
  return [
    rect.x + ((frameW - camX) / frameW) * rect.w,
    rect.y + (camY / frameH) * rect.h,
  ];
}

export function mapCamRadius(camR, frameW, rect) {
  return camR * (rect.w / frameW);
}

export function pipRect(W, H) {
  const w = PIP_W_FRAC * W;
  const h = w * (H / Math.max(1, W));
  const margin = Math.max(12, W / 60);
  return { x: margin, y: H - margin - h, w, h };
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
    this.nextPunchAt = 0;
    this.handCue = null;
    this._fast = { i: RW_I, speed: 0 };
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

  // A PunchDetector `punch` from either hand. The CNN label is not consulted.
  notePunch(now) {
    const step = acceptPunch(this.nextPunchAt, now);
    this.nextPunchAt = step.nextAt;
    if (!step.ok) return false;
    this.launchAttack();
    return true;
  }

  launchAttack() {
    const id = ++this.attackSeq;
    const sentAt = this.net.netNow();
    this.outgoing.set(id, { id, localT: performance.now() / 1000 });
    this.net.sendEvent("attack", { id, sentAt });
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
    this.nextPunchAt = 0;
    this.handCue = null;
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

  // Pose wind-up or the attack message, whichever happens first, lights the throwing hand.
  armHandCue(now) {
    const i = this._fast.speed > 0.25 ? this._fast.i : RW_I;
    if (!this.handCue || now - this.handCue.start > 1.2) this.handCue = { i, start: now };
    else this.handCue.i = i;
  }

  // Full-screen opponent, raw camera x, so their right hand is on screen-left (facing us).
  drawStage(ctx, W, H, S, now) {
    const renderT = this.net.netNow() - OPP_RENDER_DELAY_S;
    const sample = sampleOpponent(this.oppBuf, renderT);
    if (sample.k && sample.alpha > 0.02) {
      ctx.save();
      ctx.globalAlpha = Math.max(0.35, sample.alpha);
      this._drawSkeleton(ctx, sample.k, renderT, { x: 0, y: 0, w: W, h: H }, S);
      ctx.restore();
      this._noteMotion(sample.k, renderT, now);
      this._prevDraw = { t: renderT, k: sample.k };
    }
    this._drawHandCue(ctx, W, H, S, now);
    ctx.font = `bold ${18 * S}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = "rgba(180,255,255,0.85)";
    ctx.fillText(this.oppName, W / 2, H - 16 * S);
    if (this.stale) this._banner(ctx, W, H, S, "Opponent connection lost");
  }

  _noteMotion(k, renderT, now) {
    const prev = this._prevDraw;
    const dt = prev ? renderT - prev.t : 0;
    let best = { i: RW_I, speed: 0 };
    if (prev && dt > 1e-4) {
      for (const i of [LW_I, RW_I]) {
        const dx = k[i * 3] - prev.k[i * 3];
        const dy = k[i * 3 + 1] - prev.k[i * 3 + 1];
        const speed = Math.hypot(dx, dy) / dt;
        if (speed > best.speed) best = { i, speed };
      }
    }
    this._fast = best;
    if (best.speed > 0.4) this.armHandCue(now);
  }

  _drawHandCue(ctx, W, H, S, now) {
    const cue = this.handCue;
    const k = this._prevDraw?.k;
    if (!cue || !k || now - cue.start > 1.8) {
      if (cue && now - cue.start > 1.8) this.handCue = null;
      return;
    }
    const x = k[cue.i * 3] * W;
    const y = k[cue.i * 3 + 1] * H;
    const age = now - cue.start;
    const grow = Math.min(1, age / 0.35);
    const flash = Math.max(0, 1 - age / 0.22);
    const radius = (28 + 110 * grow) * S;
    const g = ctx.createRadialGradient(x, y, 4, x, y, radius);
    g.addColorStop(0, `rgba(255,255,255,${0.85 * Math.max(flash, 0.35)})`);
    g.addColorStop(0.35, `rgba(255,170,40,${0.55 + 0.35 * flash})`);
    g.addColorStop(1, "rgba(255,120,0,0)");
    ctx.save();
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    if (flash > 0.05) {
      ctx.globalAlpha = flash;
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(x, y, 18 * S, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
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
    const thick = Math.max(S, box.w / 480);
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
      ctx.lineWidth = (fast ? 8 : 4) * thick;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
    };
    chain([LS_I, LE_I, LW_I], speed(LW_I) > 0.45);
    chain([RS_I, RE_I, RW_I], speed(RW_I) > 0.45);
    chain([LH_I, RH_I], false);
    chain([LS_I, RS_I], false);
    chain([LW_I, 9, 7, LW_I, 11], speed(LW_I) > 0.45); // index, pinky, thumb
    chain([RW_I, 10, 8, RW_I, 12], speed(RW_I) > 0.45);
    const nose = pt(NOSE_I);
    if (nose) {
      ctx.beginPath();
      ctx.fillStyle = css(CYAN, 0.9);
      ctx.arc(nose[0], nose[1], Math.max(10, box.w * 0.055), 0, Math.PI * 2);
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
  const punching = cycle > 3.4;
  const p = punching ? Math.sin(((cycle - 3.4) / 1.1) * Math.PI) : 0;
  const pts = [
    [0.50, 0.22],
    [0.38, 0.36], [0.62, 0.36],
    [0.32, 0.52], [0.68 - 0.10 * p, 0.50 - 0.12 * p],
    [0.30, 0.68], [0.74 - 0.32 * p, 0.66 - 0.36 * p],
    [0.28, 0.74], [0.76 - 0.30 * p, 0.72 - 0.32 * p],
    [0.27, 0.72], [0.78 - 0.31 * p, 0.70 - 0.30 * p],
    [0.32, 0.70], [0.72 - 0.26 * p, 0.68 - 0.28 * p],
    [0.42, 0.82], [0.58, 0.82],
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
    let outcome, dmg;
    if (this.rng() < 0.5) {
      outcome = "Dodge";
      dmg = 0;
      this.score += DEFENSE_BONUS;
    } else if (this.rng() < 0.5) {
      outcome = "Block";
      dmg = blockDamage();
      this.score += BLOCK_SCORE;
    } else {
      outcome = "Hit";
      dmg = hitDamage();
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
      emitOn(this, "attack", { id: ++this.seq, sentAt: this.netNow() });
      this._scheduleAttack();
    }, delay);
    if (this._atk.unref) this._atk.unref();
  }
}
