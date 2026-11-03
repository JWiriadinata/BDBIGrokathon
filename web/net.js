// Versus networking for the static web build. No DOM and no `window` access, so
// this module loads under node (tools/check_net.mjs passes a fake PeerCtor).
//
// PeerJS 1.5.4 provides the WebRTC data channels and its public cloud does the
// signaling. Video is never sent: the pose channel carries 15 image landmarks
// and the events channel carries small JSON messages.
//
// TURN: the default ICE list is STUN only. Strict NATs often fail to connect
// without a relay. Append `{urls, username, credential}` entries to
// TURN_SERVERS (nothing is shipped with credentials). See web/CODEBASE.md.

import { POSE_HZ } from "./versus.js";

export const PROTOCOL_VERSION = 1;
export const ROOM_PREFIX = "pp-";
export const CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
// Append TURN (or extra STUN) entries here. Empty by default; see the note above.
export const TURN_SERVERS = [];

export const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
const PING_BURST = 8, PING_GAP_MS = 100, RESYNC_EVERY_MS = 5000, RESYNC_PINGS = 3;
const BEST_SAMPLE_MAX_AGE_S = 30;

const ERROR_TEXT = {
  "not-found": "Room not found.",
  full: "Room is full.",
  version: "Update required",
};

export function randomCode() {
  let s = "";
  for (let i = 0; i < 5; i++) s += CODE_ALPHABET[Math.floor(Math.random() * CODE_ALPHABET.length)];
  return s;
}

export function normalizeCode(code) {
  return String(code ?? "").toUpperCase().replace(/\s+/g, "").slice(0, 5);
}

export function cleanName(name) {
  const s = String(name ?? "Player").replace(/[\r\n\t]/g, " ").trim().slice(0, 16);
  return s || "Player";
}

export function buildIceConfig() {
  return {
    iceServers: [
      { urls: "stun:stun.l.google.com:19302" },
      { urls: "stun:global.stun.twilio.com:3478" },
      ...TURN_SERVERS,
    ],
  };
}

export function round3(n) {
  return Math.round(Number(n) * 1000) / 1000;
}

// Image landmarks only (normalized x, y, visibility). Returns null if any of the
// 15 upper-body points is missing — callers skip the send rather than shipping zeros.
export function encodePose(lms) {
  if (!lms) return null;
  const k = new Array(POSE_INDICES.length * 3);
  for (let i = 0; i < POSE_INDICES.length; i++) {
    const lm = lms[POSE_INDICES[i]];
    if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) return null;
    const vis = lm.visibility ?? lm.vis ?? 0;
    k[i * 3] = round3(lm.x);
    k[i * 3 + 1] = round3(lm.y);
    k[i * 3 + 2] = round3(vis);
  }
  return k;
}

export function decodePose(k) {
  const out = [];
  for (let i = 0; i < POSE_INDICES.length; i++) {
    out.push({ x: k[i * 3], y: k[i * 3 + 1], visibility: k[i * 3 + 2] });
  }
  return out;
}

// One NTP-style sample. Times are seconds on each side's local clock.
// rtt = (c3 - c0) - (s2 - s1), offset = ((s1 - c0) + (s2 - c3)) / 2.
export function clockSample({ c0, s1, s2, c3 }) {
  return {
    rtt: (c3 - c0) - (s2 - s1),
    offset: ((s1 - c0) + (s2 - c3)) / 2,
  };
}

// Keep the lowest-rtt sample. A sample older than 30 s may be replaced even by a
// slower one, so a stale "best" can't pin the offset forever. `state` is mutated.
export function noteClockSample(state, sample, now) {
  const stale = state.bestAt > 0 && now - state.bestAt > BEST_SAMPLE_MAX_AGE_S;
  if (!(sample.rtt < state.bestRtt || stale)) return false;
  state.bestRtt = sample.rtt;
  state.offset = sample.offset;
  state.bestAt = now;
  return true;
}

function sleep(ms) {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    if (t.unref) t.unref();
  });
}

export class Net extends EventTarget {
  constructor({ PeerCtor, name = "Player", W = 960, H = 540 } = {}) {
    super();
    this.PeerCtor = PeerCtor;
    this.name = cleanName(name);
    this.W = W;
    this.H = H;
    this.role = null;
    this.code = null;
    this.peer = null;
    this.eventsConn = null;
    this.poseConn = null;
    this.guestId = null;
    this.clockOffset = 0;
    this.clock = { bestRtt: Infinity, bestAt: 0, offset: 0 };
    this.connected = false;
    this.closed = false;
    this.ignoreClose = false;
    this._failed = false;
    this._left = false;
    this._peerReady = false;
    this.linkReady = false;
    this.remoteHello = null;
    this._pongWait = new Map();
    this._pingSeq = 0;
    this._syncing = null;
    this._resync = null;
    this.lastPoseSent = -Infinity;
    this._linkResolve = null;
    this._linkReject = null;
    this._linkTimer = null;
  }

  localNow() { return performance.now() / 1000; }
  netNow() { return this.localNow() + this.clockOffset; }

  setProfile({ name, W, H } = {}) {
    if (name !== undefined) this.name = cleanName(name);
    if (Number.isFinite(W)) this.W = W;
    if (Number.isFinite(H)) this.H = H;
  }

  emit(type, detail) {
    try { this.dispatchEvent(new CustomEvent(type, { detail })); }
    catch (err) { console.error("net listener", type, err); }
  }

  // Host id is "pp-" + a 5-char code. An id collision retries up to 5 times.
  async host() {
    if (typeof this.PeerCtor !== "function") throw this._error("no-peer", "PeerJS failed to load.");
    this.role = "host";
    this.clockOffset = 0;
    let last;
    for (let i = 0; i < 5; i++) {
      const code = randomCode();
      try {
        await this._openPeer(ROOM_PREFIX + code);
        this.code = code;
        return code;
      } catch (err) {
        last = err;
        try { this.peer?.destroy?.(); } catch { /* already dead */ }
        this.peer = null;
        this._peerReady = false;
        if (err?.type !== "unavailable-id") break;
      }
    }
    throw this._error(last?.type || "unavailable-id", "Could not create room.");
  }

  // Resolves once both channels are open and hello has been exchanged.
  async join(code) {
    if (typeof this.PeerCtor !== "function") throw this._error("no-peer", "PeerJS failed to load.");
    this.role = "guest";
    const clean = normalizeCode(code);
    this.code = clean;
    if (clean.length !== 5) throw this._error("not-found", ERROR_TEXT["not-found"]);
    await this._openPeer(null);
    const remote = ROOM_PREFIX + clean;
    const opts = (channel, reliable) => ({
      reliable, serialization: "json", metadata: { channel },
    });
    const ready = this._waitForLink();
    this.eventsConn = this.peer.connect(remote, opts("events", true));
    this._bindGuest(this.eventsConn, "events");
    this.poseConn = this.peer.connect(remote, opts("pose", false));
    this._bindGuest(this.poseConn, "pose");
    await ready;
  }

  sendEvent(type, payload = {}) {
    if (this.closed || !this.eventsConn) return;
    try { this.eventsConn.send({ t: type, v: PROTOCOL_VERSION, ...payload }); }
    catch { /* channel already gone */ }
  }

  // Throttled to POSE_HZ. `t` is the sender's netNow; `lms` are image landmarks.
  // A missing pose is skipped (never sent as zeros).
  sendPose(t, lms) {
    const conn = this.poseConn;
    if (this.closed || !conn || !lms) return;
    const now = this.localNow();
    if (now - this.lastPoseSent < 1 / POSE_HZ) return;
    const k = encodePose(lms);
    if (!k) return;
    this.lastPoseSent = now;
    // Pose channel: `t` is the sample time (the channel is the type). `v` is still checked.
    try { conn.send({ v: PROTOCOL_VERSION, t, k }); }
    catch { /* unreliable channel dropped the send */ }
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    this.connected = false;
    clearTimeout(this._linkTimer);
    clearInterval(this._resync);
    this._resync = null;
    for (const w of this._pongWait.values()) clearTimeout(w.timer);
    this._pongWait.clear();
    const drop = (conn) => { try { conn?.close(); } catch { /* ignore */ } };
    drop(this.eventsConn);
    drop(this.poseConn);
    try { this.peer?.destroy?.(); } catch { /* ignore */ }
  }

  // ── Peer + channels ────────────────────────────────────────────────────────

  _error(code, message) {
    const err = new Error(message);
    err.code = code;
    return err;
  }

  _openPeer(id) {
    const config = buildIceConfig();
    const peer = id ? new this.PeerCtor(id, { config }) : new this.PeerCtor({ config });
    this.peer = peer;
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (fn, value) => {
        if (settled) return;
        settled = true;
        fn(value);
      };
      peer.on("open", (opened) => {
        this._peerReady = true;
        this.peerId = opened || id || peer.id;
        finish(resolve, this.peerId);
      });
      peer.on("error", (err) => {
        if (!settled) finish(reject, err);
        else this._onPeerError(err);
      });
      peer.on("connection", (conn) => this._accept(conn));
      peer.on("disconnected", () => this._gone());
      peer.on("close", () => { /* destroy() ends here; _gone handles the remote case */ });
    });
  }

  _bindGuest(conn, channel) {
    if (!conn) return;
    conn.on("open", () => {
      if (channel === "events") {
        this.eventsConn = conn;
        this._sendHello();
      } else this.poseConn = conn;
      this._checkLink();
    });
    conn.on("data", (data) => (channel === "events" ? this.handle(data) : this._handlePose(data)));
    conn.on("close", () => this._gone());
    conn.on("error", () => { if (!this.linkReady) this._fail("not-found"); });
  }

  _accept(conn) {
    try {
      const channel = conn?.metadata?.channel;
      if (channel !== "events" && channel !== "pose") {
        try { conn.close(); } catch { /* ignore */ }
        return;
      }
      const remoteId = conn.peer;
      if (this.guestId && remoteId && remoteId !== this.guestId) {
        const rejectFull = () => {
          try { conn.send({ t: "full", v: PROTOCOL_VERSION }); } catch { /* ignore */ }
          try { conn.close(); } catch { /* ignore */ }
        };
        conn.on("open", rejectFull);
        return;
      }
      if (remoteId) this.guestId = remoteId;
      conn.on("data", (data) => (channel === "events" ? this.handle(data) : this._handlePose(data)));
      conn.on("close", () => this._gone());
      conn.on("error", () => {});
      conn.on("open", () => {
        if (channel === "events") {
          this.eventsConn = conn;
          this._sendHello();
        } else this.poseConn = conn;
        this._checkLink();
      });
    } catch (err) {
      console.error("accept failed", err);
    }
  }

  _sendHello() {
    this.sendEvent("hello", { name: this.name, v: PROTOCOL_VERSION, W: this.W, H: this.H });
  }

  _waitForLink() {
    if (this.linkReady) return Promise.resolve();
    return new Promise((resolve, reject) => {
      this._linkResolve = resolve;
      this._linkReject = reject;
      this._linkTimer = setTimeout(() => this._fail("not-found"), 8000);
    });
  }

  _checkLink() {
    if (this.linkReady || this.closed) return;
    if (!this.eventsConn || !this.poseConn || !this.remoteHello) return;
    this.linkReady = true;
    this.connected = true;
    clearTimeout(this._linkTimer);
    const hello = this.remoteHello;
    this.emit("open", { name: hello.name, W: hello.W, H: hello.H });
    this._startClockSync();
    const resolve = this._linkResolve;
    this._linkResolve = null;
    this._linkReject = null;
    if (resolve) resolve();
  }

  _fail(code) {
    if (this._failed || this.closed) return;
    this._failed = true;
    const message = ERROR_TEXT[code] || "Connection failed.";
    this.emit("error", { code, message });
    const reject = this._linkReject;
    this._linkResolve = null;
    this._linkReject = null;
    this.close();
    if (reject) reject(this._error(code, message));
  }

  _gone() {
    if (this.closed || this.ignoreClose || this._left || this._failed) return;
    this._left = true;
    this.connected = false;
    this.emit("left", {});
  }

  _onPeerError(err) {
    const type = err?.type;
    if (this.linkReady) return;
    if (type === "peer-unavailable" || type === "unavailable-id" || type === "network") this._fail("not-found");
  }

  // ── Messages ───────────────────────────────────────────────────────────────

  handle(raw) {
    let msg = raw;
    try {
      if (typeof raw === "string") msg = JSON.parse(raw);
      if (!msg || typeof msg !== "object") return;
    } catch {
      return;
    }
    try {
      switch (msg.t) {
        case "hello":
          this._onHello(msg);
          break;
        case "ping":
          this._onPing(msg);
          break;
        case "pong":
          this._onPong(msg);
          break;
        case "full":
          this._fail("full");
          break;
        case "ready":
          this.emit("ready", { ready: !!msg.ready });
          break;
        case "start":
          this.emit("start", { t0: Number(msg.t0) });
          break;
        case "attack":
          this._onAttack(msg);
          break;
        case "attackResult":
          this.emit("attackResult", {
            id: msg.id, outcome: msg.outcome, dmg: Number(msg.dmg) || 0,
          });
          break;
        case "state":
          this.emit("state", { hp: Number(msg.hp), score: Number(msg.score) });
          break;
        case "ko":
          this.emit("ko", { loser: msg.loser === "you" ? "you" : "me" });
          break;
        case "rematch":
          this.emit("rematch", {});
          break;
        default:
          break; // unknown t is ignored
      }
    } catch {
      /* a bad field must not take the link down */
    }
  }

  _onHello(msg) {
    if (msg.v !== PROTOCOL_VERSION) {
      this._fail("version");
      return;
    }
    this.remoteHello = {
      name: cleanName(msg.name),
      W: Number(msg.W) || 0,
      H: Number(msg.H) || 0,
    };
    this._checkLink();
  }

  _onAttack(msg) {
    // One attack shape. `kind` and `mult` are ignored if an older client still sends them.
    if (msg.id == null) return;
    this.emit("attack", { id: msg.id, sentAt: Number(msg.sentAt) || 0 });
  }

  _handlePose(raw) {
    try {
      const msg = typeof raw === "string" ? JSON.parse(raw) : raw;
      if (!msg || !Array.isArray(msg.k) || msg.k.length !== POSE_INDICES.length * 3) return;
      if (msg.v !== undefined && msg.v !== PROTOCOL_VERSION) return;
      const t = typeof msg.t === "number" ? msg.t : Number(msg.ts);
      if (!Number.isFinite(t)) return;
      this.emit("pose", { t, k: msg.k });
    } catch {
      /* drop the frame */
    }
  }

  // ── Clock sync ─────────────────────────────────────────────────────────────

  _onPing(msg) {
    if (!Number.isFinite(msg?.c0)) return;
    const s1 = this.localNow();
    const s2 = this.localNow();
    this.sendEvent("pong", { id: msg.id, c0: msg.c0, s1, s2 });
  }

  _onPong(msg) {
    const w = this._pongWait.get(msg?.id);
    if (!w || !Number.isFinite(msg?.s1) || !Number.isFinite(msg?.s2)) return;
    if (Math.abs(w.c0 - msg.c0) > 1e-4) return;
    clearTimeout(w.timer);
    this._pongWait.delete(msg.id);
    const c3 = this.localNow();
    const sample = clockSample({ c0: w.c0, s1: msg.s1, s2: msg.s2, c3 });
    if (!Number.isFinite(sample.rtt) || !Number.isFinite(sample.offset)) {
      w.resolve();
      return;
    }
    const rttMs = Math.max(0, Math.round(sample.rtt * 1000));
    this.emit("latency", { rtt: rttMs });
    // The host's clock is the reference. Only the guest adopts an offset.
    if (this.role === "guest" && noteClockSample(this.clock, sample, this.localNow())) {
      this.clockOffset = this.clock.offset;
    }
    w.resolve();
  }

  _startClockSync() {
    if (this._syncing || this.closed) return;
    this._syncing = this._runBurst(PING_BURST, PING_GAP_MS).then(() => {
      if (this.closed) return;
      this._resync = setInterval(() => { this._runBurst(RESYNC_PINGS, PING_GAP_MS); }, RESYNC_EVERY_MS);
      if (this._resync.unref) this._resync.unref();
    });
  }

  async _runBurst(n, gapMs) {
    for (let i = 0; i < n; i++) {
      if (this.closed) return;
      await this._onePing();
      if (i < n - 1) await sleep(gapMs);
    }
  }

  _onePing() {
    if (this.closed || !this.eventsConn) return Promise.resolve();
    const id = ++this._pingSeq;
    const c0 = this.localNow();
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this._pongWait.delete(id);
        resolve();
      }, 1000);
      if (timer.unref) timer.unref();
      this._pongWait.set(id, { c0, resolve, timer });
      this.sendEvent("ping", { id, c0 });
    });
  }
}
