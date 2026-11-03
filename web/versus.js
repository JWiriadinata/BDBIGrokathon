// Versus mode state + protocol handlers. game.js renders and calls these hooks.
import { Net } from "./net.js";

export const VS_PUNCH_GAP_S = 0.6;   // matches the server rate limit; one message per detected punch
export const vs = { net: null, active: false, me: null, opp: null, code: null, health: {}, startAt: 0, endAt: 0, lastPunchT: -1,
                     oppImg: null, oppFrameAt: 0, sent: 0, recv: 0 };

const $ = (id) => document.getElementById(id);

export function remainingS() { return vs.net ? Math.max(0, (vs.endAt - vs.net.serverNow()) / 1000) : 0; }
export function countdownLeftS() { return vs.net ? (vs.startAt - vs.net.serverNow()) / 1000 : 0; }

// cb: { status(html), onMatchStart(), onAttack(kind, attackId), onState(), onEnd(text), onLeft(text) }
export async function connect(cb) {
  if (vs.net) return vs.net;
  const net = vs.net = new Net();
  net.on("room_created", (m) => { Object.assign(vs, { code: m.code, me: m.playerId });
    cb.status(`Room code: <b style="font-size:1.6em;letter-spacing:.15em">${m.code}</b><br>Waiting for opponent…`); })
    .on("room_joined", (m) => { Object.assign(vs, { code: m.code, me: m.playerId, opp: m.opponent });
      cb.status(`Room <b>${m.code}</b> — opponent: <b>${escape(m.opponent.name)}</b> connected.<br>Press Ready.`); $("vsReady").hidden = false; })
    .on("match_start", (m) => {
      Object.assign(vs, { active: true, startAt: m.startAt, endAt: m.startAt + m.durationS * 1000,
                          health: Object.fromEntries(m.players.map((p) => [p.id, 100])) });
      cb.onMatchStart();
    })
    .on("attack", (m) => { vs.recv += 1; cb.onAttack(m.kind, m.attackId); })
    .on("frame", (m) => {   // opponent's webcam
      const img = new Image();
      img.onload = () => { vs.oppImg = img; vs.oppFrameAt = performance.now(); };
      img.src = m.d;
    })
    .on("state", (m) => { vs.health = m.health; cb.onState?.(m.events ?? []); })
    .on("match_end", (m) => {
      vs.active = false; vs.health = m.health;
      const text = m.reason === "disconnect" ? "Opponent disconnected — you win"
                 : m.winner === null ? "Draw!" : m.winner === vs.me ? "You win!" : "You lose!";
      cb.onEnd(`${text} (${m.reason})`);
    })
    .on("opponent_left", () => { vs.opp = null; vs.active = false; cb.onLeft("Opponent disconnected."); })
    .on("error", (m) => { if (m.reason !== "rejected") cb.onLeft(`Error: ${m.reason.replace("_", " ")}`); })
    .on("close", () => { vs.net = null; vs.active = false; cb.onLeft("Disconnected from server."); });
  try { await net.connect(); } catch (e) { vs.net = null; throw e; }
  return net;
}

// Own detected punch → `punch` message (debounced to one per VS_PUNCH_GAP_S).
export function sendPunch(kind, side, conf, t) {
  if (!vs.active || t - vs.lastPunchT < VS_PUNCH_GAP_S) return false;
  vs.lastPunchT = t;
  vs.net.send("punch", { kind, side, conf, t });
  vs.sent += 1;
  return true;
}

export function disconnect() {
  if (!vs.net) return;
  vs.net.on("close", null); vs.net.close();
  Object.assign(vs, { net: null, active: false, opp: null, code: null });
}

// Own webcam → small JPEG for the opponent's main view (skipped while the socket is backed up).
const FRAME_W = 320, FRAME_H = 180, FRAME_EVERY_MS = 100;
let frameCanvas = null, lastFrameAt = 0;
export function sendFrame(video) {
  const now = performance.now(), ws = vs.net?.ws;
  if (!vs.opp || !ws || ws.readyState !== 1 || ws.bufferedAmount > 256 * 1024 || now - lastFrameAt < FRAME_EVERY_MS) return;
  lastFrameAt = now;
  frameCanvas ??= Object.assign(document.createElement("canvas"), { width: FRAME_W, height: FRAME_H });
  frameCanvas.getContext("2d").drawImage(video, 0, 0, FRAME_W, FRAME_H);
  vs.net.send("frame", { d: frameCanvas.toDataURL("image/jpeg", 0.6) });
}
export const oppVideoLive = () => !!vs.oppImg && performance.now() - vs.oppFrameAt < 1500;

export function sendOutcome(attackId, result) { if (vs.active) vs.net.send("outcome", { attackId, result }); }

function escape(s) { return String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`); }
