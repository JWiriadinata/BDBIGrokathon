// Node checks for versus networking and damage. No dependencies.
// Run: node web/tools/check_net.mjs
import { Net, TURN_SERVERS, buildIceConfig, clockSample, noteClockSample, encodePose, decodePose, POSE_INDICES, clampMult } from "../net.js";
import {
  sampleOpponent, hitDamage, blockDamage, applyHp, enqueueAttack, judge,
  BotOpponent, botSkeleton, OPP_HOLD_S, DAMAGE, MAX_HP,
} from "../versus.js";

let failed = 0;
function assert(cond, msg) {
  if (cond) { console.log(`  ok  ${msg}`); return; }
  failed += 1;
  console.error(`FAIL  ${msg}`);
}

function section(name) { console.log(`\n${name}`); }

section("1. clock sync");
{
  const TRUTH = 12.345;
  // Host clock = guest clock + TRUTH. One-way delays are seconds.
  function exchange(c0, d1, d2, proc = 0) {
    const s1 = c0 + d1 + TRUTH;
    const s2 = s1 + proc;
    const c3 = s2 - TRUTH + d2;
    return clockSample({ c0, s1, s2, c3 });
  }
  // Asymmetric 30 ms / 70 ms: error is (d1 - d2) / 2 = -20 ms, inside the 25 ms band.
  const fast = exchange(1001, 0.03, 0.07, 0.001);
  // A worse path with the opposite skew, so picking it would miss the 25 ms band.
  const slow = exchange(1000, 0.20, 0.05, 0);
  assert(fast.rtt < slow.rtt, `fast rtt ${fast.rtt.toFixed(3)} < slow rtt ${slow.rtt.toFixed(3)}`);
  const state = { bestRtt: Infinity, bestAt: 0, offset: 0 };
  assert(noteClockSample(state, slow, 10) === true, "first sample is adopted");
  assert(noteClockSample(state, fast, 11) === true, "lower rtt replaces it");
  assert(Math.abs(state.offset - TRUTH) < 0.025, `offset ${state.offset} within 25 ms of ${TRUTH}`);
  assert(state.bestRtt === fast.rtt, "stored rtt is the best sample");
  const worse = exchange(1002, 0.40, 0.01, 0);
  assert(noteClockSample(state, worse, 12) === false, "a later worse rtt is ignored");
  assert(state.bestRtt === fast.rtt && Math.abs(state.offset - TRUTH) < 0.025, "best sample stays");
  // After 30 s the stored best may be replaced.
  assert(noteClockSample(state, slow, 11 + 31) === true, "a sample older than 30 s may be replaced");
}

section("2. pose encode/decode");
{
  const lms = Array.from({ length: 33 }, (_, i) => ({
    x: (i * 0.137) % 1,
    y: (i * 0.091 + 0.02) % 1,
    z: 5 + i,                       // world-style extra; must not be encoded
    visibility: ((i * 3) % 10) / 10,
  }));
  const k = encodePose(lms);
  assert(k && k.length === 45, "flat array is 45 numbers");
  const back = decodePose(k);
  let maxLoss = 0;
  for (let i = 0; i < POSE_INDICES.length; i++) {
    const src = lms[POSE_INDICES[i]];
    maxLoss = Math.max(maxLoss, Math.abs(back[i].x - src.x), Math.abs(back[i].y - src.y), Math.abs(back[i].visibility - src.visibility));
  }
  assert(maxLoss <= 0.001, `round trip loss ${maxLoss} <= 0.001`);
  assert(encodePose(null) === null, "missing pose encodes to null");
  const partial = lms.slice(0, 10);
  assert(encodePose(partial) === null, "a short landmark list is not sent as zeros");
  const net = new Net({ name: "Pose" });
  const poses = [];
  net.poseConn = { send(msg) { poses.push(msg); } };
  for (let i = 0; i < 8; i++) net.sendPose(net.netNow(), lms);
  net.sendPose(net.netNow(), null);
  assert(poses.length === 1, "POSE_HZ throttle sends one frame per burst");
  assert(poses[0].k.length === 45 && poses[0].v === 1 && typeof poses[0].t === "number", "pose packet is {v, t, k}");
  assert(!("z" in poses[0]) && poses[0].k.every((n) => typeof n === "number"), "no z / world fields on the wire");
}

section("3. interpolation");
{
  const buf = [
    { t: 1.0, k: [0, 0, 1] },
    { t: 1.1, k: [1, 2, 0] },
  ];
  const mid = sampleOpponent(buf, 1.05);
  const midpoint = mid.k.every((v, i) => Math.abs(v - (buf[0].k[i] + buf[1].k[i]) / 2) < 1e-6);
  assert(midpoint && mid.alpha === 1 && !mid.held, `t=1.05 is the midpoint (${mid.k})`);
  const hold = sampleOpponent(buf, 1.1 + OPP_HOLD_S);
  const held = hold.k.every((v, i) => v === buf[1].k[i]);
  assert(held && hold.alpha === 1 && hold.held, "a query within 0.4 s of the newest sample holds it");
  const faded = sampleOpponent(buf, 1.1 + OPP_HOLD_S + 0.2);
  assert(faded.held && faded.alpha < 1 && faded.alpha > 0, `past 0.4 s the hold fades (alpha ${faded.alpha.toFixed(2)})`);
  const gone = sampleOpponent(buf, 1.1 + OPP_HOLD_S + 5);
  assert(gone.alpha === 0, "a long gap fades the skeleton out");
  assert(sampleOpponent([], 1).k === null, "empty buffer yields nothing");
}

section("4. damage");
{
  assert(hitDamage("jab", 1.5) === 12, "PERFECT jab = 12");
  assert(hitDamage("jab", 1) === 8, "GOOD jab = 8");
  assert(hitDamage("hook", 1.5) === 18, "PERFECT hook = 18");
  const blocked = blockDamage("uppercut", 1.5);
  assert(blocked === Math.round(22.5 * 0.5), `Block on a PERFECT uppercut = round(22.5*0.5) (${blocked})`);
  assert(applyHp(5, 10) === 0 && applyHp(0, 4) === 0 && applyHp(100, 12) === 88, "HP never goes below 0");
  assert(judge(10, 10, 300, 100) === "win" && judge(10, 10, 50, 50) === "draw" && judge(0, 1, 999, 0) === "lose", "winner is HP, then score, then draw");
  const q = [{ id: 1 }, { id: 2 }];
  assert(enqueueAttack(q, { id: 3 }) === false && q.length === 2, "a full attack queue drops the new one");
  assert(enqueueAttack([{ id: 1 }], { id: 2 }) === true, "a short queue accepts");
}

section("5 + 6. fake peer exchange and the message handler");
{
  const registry = new Map();
  const delivered = [];

  class FakeConn {
    constructor(remoteId, opts) {
      this.peer = remoteId;
      this.metadata = opts?.metadata || {};
      this.reliable = opts?.reliable;
      this.serialization = opts?.serialization;
      this.handlers = {};
      this.other = null;
      this._closed = false;
    }
    on(ev, fn) { (this.handlers[ev] ||= []).push(fn); return this; }
    emit(ev, ...args) { for (const fn of this.handlers[ev] || []) fn(...args); }
    send(data) {
      const packet = typeof data === "string" ? data : JSON.parse(JSON.stringify(data));
      queueMicrotask(() => {
        if (!this.other || this.other._closed) return;
        delivered.push(packet?.t);
        this.other.emit("data", packet);
      });
    }
    close() {
      if (this._closed) return;
      this._closed = true;
      const other = this.other;
      queueMicrotask(() => {
        this.emit("close");
        if (other && !other._closed) { other._closed = true; other.emit("close"); }
      });
    }
  }

  class FakePeer {
    constructor(id, options) {
      if (typeof id === "object" || id == null) { options = id || {}; id = undefined; }
      this.id = id || `anon-${Math.random().toString(36).slice(2, 10)}`;
      this.options = options || {};
      this.handlers = {};
      this.incoming = [];
      queueMicrotask(() => {
        if (registry.has(this.id)) {
          const err = new Error("unavailable-id");
          err.type = "unavailable-id";
          this.emit("error", err);
          return;
        }
        registry.set(this.id, this);
        this.emit("open", this.id);
      });
    }
    on(ev, fn) { (this.handlers[ev] ||= []).push(fn); return this; }
    emit(ev, ...args) { for (const fn of this.handlers[ev] || []) fn(...args); }
    connect(remoteId, opts) {
      const local = new FakeConn(remoteId, opts);
      queueMicrotask(() => {
        const remote = registry.get(remoteId);
        if (!remote) {
          const err = new Error("peer-unavailable");
          err.type = "peer-unavailable";
          this.emit("error", err);
          local.emit("error", err);
          return;
        }
        const incoming = new FakeConn(this.id, opts);
        local.other = incoming;
        incoming.other = local;
        remote.incoming.push(incoming);
        remote.emit("connection", incoming);
        queueMicrotask(() => { local.emit("open"); incoming.emit("open"); });
      });
      return local;
    }
    destroy() { registry.delete(this.id); }
  }

  const flush = () => new Promise((r) => setTimeout(r, 0));
  const interesting = (t) => ["hello", "ready", "start", "attack", "attackResult"].includes(t);

  TURN_SERVERS.push({ urls: "turn:example.com:3478", username: "u", credential: "p" });
  const ice = buildIceConfig();
  TURN_SERVERS.pop();
  assert(ice.iceServers.length === 3 && ice.iceServers[0].urls.includes("stun.l.google.com"), "STUN list plus TURN_SERVERS");

  const host = new Net({ PeerCtor: FakePeer, name: "Ada", W: 960, H: 540 });
  const guest = new Net({ PeerCtor: FakePeer, name: "Grace", W: 960, H: 540 });
  const attacks = [];
  const order = [];
  host.addEventListener("attack", (e) => { attacks.push(e.detail); order.push("attack"); });
  host.addEventListener("ready", () => order.push("ready"));
  guest.addEventListener("start", (e) => order.push(`start:${e.detail.t0}`));
  guest.addEventListener("attackResult", (e) => order.push(`attackResult:${e.detail.outcome}`));

  const code = await host.host();
  assert(/^[A-Z2-9]{5}$/.test(code) && !/[IO01]/.test(code), `host code ${code}`);
  const hostPeer = registry.get(`pp-${code}`);
  assert(hostPeer?.options?.config?.iceServers?.length >= 2, "Peer was constructed with the ICE config");
  await guest.join(code);
  assert(host.connected && guest.connected, "both channels open and hello exchanged");
  assert(host.remoteHello.name === "Grace" && guest.remoteHello.name === "Ada", "names from hello");
  assert(host.clockOffset === 0, "host clock offset stays 0");

  guest.sendEvent("ready", { ready: true });
  await flush();
  host.sendEvent("start", { t0: 123.5 });
  await flush();
  guest.sendEvent("attack", { id: 1, kind: "jab", mult: 1.5, sentAt: 10 });
  await flush();
  host.sendEvent("attackResult", { id: 1, outcome: "Hit", dmg: 12 });
  await flush();

  const wire = delivered.filter(interesting);
  const helloAt = wire.indexOf("hello");
  const readyAt = wire.indexOf("ready");
  const startAt = wire.indexOf("start");
  const attackAt = wire.indexOf("attack");
  const resultAt = wire.indexOf("attackResult");
  assert(wire.filter((t) => t === "hello").length >= 2, "both hellos were delivered");
  assert(helloAt !== -1 && helloAt < readyAt && readyAt < startAt && startAt < attackAt && attackAt < resultAt,
    `exchange order ${wire.join(" > ")}`);
  assert(order.join(",") === "ready,start:123.5,attack,attackResult:Hit", `events ${order.join(",")}`);

  const events = hostPeer.incoming.find((c) => c.metadata.channel === "events");
  const before = attacks.length;
  events.emit("data", "{not json");
  events.emit("data", { t: "made-up", v: 1 });
  events.emit("data", { t: "attack", v: 1, id: 7, kind: "haymaker", mult: 1.5, sentAt: 0 });
  events.emit("data", { t: "attack", v: 1, id: 8, kind: "jab", mult: 9, sentAt: 0 });
  events.emit("data", { t: "attack", v: 1, id: 9, kind: "hook", mult: 0, sentAt: 0 });
  assert(attacks.length === before + 2, "unknown t, bad JSON and invalid kind are ignored");
  assert(attacks[before].mult === 1.5 && attacks[before].kind === "jab", "mult 9 is clamped to 1.5");
  assert(attacks[before + 1].mult === clampMult(0) && attacks[before + 1].mult === 0.5, "mult 0 is clamped to 0.5");

  const stranger = new Net({ PeerCtor: FakePeer, name: "Eve" });
  let strangerMsg = "";
  try { await stranger.join(code); }
  catch (err) { strangerMsg = err.message; }
  assert(strangerMsg === "Room is full.", `third player: ${strangerMsg || "(no error)"}`);

  const missing = new Net({ PeerCtor: FakePeer, name: "No" });
  let missingMsg = "";
  try { await missing.join("ZZZZZ"); }
  catch (err) { missingMsg = err.message; }
  assert(missingMsg === "Room not found.", `wrong code: ${missingMsg || "(no error)"}`);

  let versionMsg = "";
  guest.addEventListener("error", (e) => { versionMsg = e.detail?.message || ""; });
  guest.handle({ t: "hello", v: 99, name: "Future", W: 1, H: 1 });
  assert(versionMsg === "Update required", `version mismatch: ${versionMsg || "(no error)"}`);

  host.close();
  guest.close();
  stranger.close();
  missing.close();
}

section("bot opponent");
{
  const seen = { pose: 0, attack: 0, result: null, state: null };
  const bot = new BotOpponent({
    attackEvery: [20, 20],
    poseEvery: 15,
    replyMs: 0,
    rng: (() => { const v = [0.9, 0.9, 0.2]; let i = 0; return () => v[Math.min(i++, v.length - 1)]; })(),
  });
  bot.addEventListener("pose", (e) => { if (e.detail.k.length === 45) seen.pose += 1; });
  bot.addEventListener("attack", () => { seen.attack += 1; });
  bot.addEventListener("attackResult", (e) => { seen.result = e.detail; });
  bot.addEventListener("state", (e) => { seen.state = e.detail; });
  bot.start();
  bot.sendEvent("attack", { id: 4, kind: "jab", mult: 1.5, sentAt: 0 });
  await new Promise((r) => setTimeout(r, 30));
  assert(seen.result?.outcome === "Hit" && seen.result.dmg === 12, `bot answers a PERFECT jab with Hit 12 (${seen.result?.outcome} ${seen.result?.dmg})`);
  assert(seen.state?.hp === MAX_HP - 12, "bot HP drops by the damage it reports");
  const sk = botSkeleton(1.2);
  assert(sk.length === 45 && sk.every((n) => Number.isFinite(n)), "synthetic skeleton is 45 finite numbers");
  bot.setLive(true);
  await new Promise((r) => setTimeout(r, 80));
  assert(seen.pose > 0, "bot emits pose frames");
  assert(seen.attack > 0, "bot schedules an attack while live");
  bot.close();
}

if (failed) {
  console.error(`\n${failed} assertion(s) failed`);
  process.exit(1);
}
console.log("\ncheck_net: OK");
