// Pure room/match logic (no sockets). Each player is {id, name, send(msgObj)}.
export const START_HEALTH = 100;
export const DAMAGE = { Hit: 10, Block: 2, Dodge: 0 };
export const MATCH_DURATION_S = 90, COUNTDOWN_MS = 3000, PUNCH_MIN_GAP_MS = 600;
export const KINDS = ["jab", "hook", "uppercut"];
const CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

export class Lobby {
  constructor(now = () => Date.now()) { this.rooms = new Map(); this.now = now; this.nextId = 1; this.nextAttack = 1; }

  newCode() {
    let code;
    do { code = Array.from({ length: 5 }, () => CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)]).join(""); }
    while (this.rooms.has(code));
    return code;
  }

  createRoom(player) {
    const code = this.newCode();
    const room = { code, players: [player], ready: new Set(), match: null, timer: null };
    this.rooms.set(code, room);
    player.room = room;
    player.send({ type: "room_created", code, playerId: player.id });
    return room;
  }

  joinRoom(player, code) {
    const room = this.rooms.get(String(code ?? "").toUpperCase());
    if (!room) return player.send({ type: "error", reason: "unknown_room" });
    if (room.players.length >= 2) return player.send({ type: "error", reason: "room_full" });
    const host = room.players[0];
    room.players.push(player);
    player.room = room;
    player.send({ type: "room_joined", code: room.code, playerId: player.id, opponent: { id: host.id, name: host.name } });
    host.send({ type: "room_joined", code: room.code, playerId: host.id, opponent: { id: player.id, name: player.name } });
    return room;
  }

  opponent(player) { return player.room?.players.find((p) => p !== player) ?? null; }
  broadcast(room, msg) { for (const p of room.players) p.send(msg); }

  ready(player) {
    const room = player.room;
    if (!room || room.match || room.players.length < 2) return;
    room.ready.add(player.id);
    if (room.ready.size < 2) return;
    const startAt = this.now() + COUNTDOWN_MS;
    const health = Object.fromEntries(room.players.map((p) => [p.id, START_HEALTH]));
    room.match = { startAt, endAt: startAt + MATCH_DURATION_S * 1000, health, lastPunch: {}, attacks: new Map(), over: false };
    this.broadcast(room, { type: "match_start", startAt, durationS: MATCH_DURATION_S,
                           players: room.players.map((p) => ({ id: p.id, name: p.name })) });
    return room.match;
  }

  remainingS(m) { return Math.max(0, (m.endAt - this.now()) / 1000); }

  punch(player, msg) {
    const m = player.room?.match, opp = this.opponent(player);
    if (!m || m.over || !opp || !KINDS.includes(msg.kind)) return false;
    const t = this.now();
    if (t < m.startAt) return false;
    if (t - (m.lastPunch[player.id] ?? -Infinity) < PUNCH_MIN_GAP_MS) return false;   // rate limit
    m.lastPunch[player.id] = t;
    const attackId = this.nextAttack++;
    m.attacks.set(attackId, { from: player.id, to: opp.id, kind: msg.kind });
    opp.send({ type: "attack", attackId, kind: msg.kind, from: player.id });
    return true;
  }

  outcome(player, msg) {
    const room = player.room, m = room?.match;
    if (!m || m.over || !(msg.result in DAMAGE)) return false;
    const atk = m.attacks.get(msg.attackId);
    if (!atk || atk.to !== player.id) return false;   // only the defender reports, once
    m.attacks.delete(msg.attackId);
    m.health[player.id] = Math.max(0, m.health[player.id] - DAMAGE[msg.result]);
    this.broadcast(room, { type: "state", health: { ...m.health }, remainingS: this.remainingS(m),
                           events: [{ attackId: msg.attackId, kind: atk.kind, from: atk.from, to: player.id, result: msg.result }] });
    if (m.health[player.id] <= 0) this.endMatch(room, "ko");
    return true;
  }

  // Called periodically; ends the match on time.
  tick(room) { if (room.match && !room.match.over && this.now() >= room.match.endAt) this.endMatch(room, "time"); }

  endMatch(room, reason, leaver = null) {
    const m = room.match;
    if (!m || m.over) return;
    m.over = true;
    let winner = null;
    if (reason === "disconnect") winner = room.players.find((p) => p !== leaver)?.id ?? null;
    else {
      const [a, b] = room.players;
      if (a && b && m.health[a.id] !== m.health[b.id]) winner = m.health[a.id] > m.health[b.id] ? a.id : b.id;
    }
    this.broadcast(room, { type: "match_end", winner, reason, health: { ...m.health } });
  }

  rematch(player) {
    const room = player.room;
    if (!room || (room.match && !room.match.over)) return;
    room.match = null;
    room.ready.clear();
    this.ready(player);
  }

  leave(player) {
    const room = player.room;
    if (!room) return;
    if (room.match && !room.match.over) this.endMatch(room, "disconnect", player);
    room.players = room.players.filter((p) => p !== player);
    room.ready.clear();
    room.match = null;
    player.room = null;
    for (const p of room.players) p.send({ type: "opponent_left" });
    if (room.players.length === 0) this.rooms.delete(room.code);
  }

  // Validates and dispatches one raw message. Returns false if rejected.
  handle(player, raw) {
    if (typeof raw !== "string" || raw.length > 4096) return false;
    let msg;
    try { msg = JSON.parse(raw); } catch { return false; }
    if (!msg || typeof msg !== "object") return false;
    const name = (n) => String(n ?? "Player").slice(0, 24);
    switch (msg.type) {
      case "ping": player.send({ type: "pong", t: msg.t, serverT: this.now() }); return true;
      case "create_room": if (player.room) return false; player.name = name(msg.name); this.createRoom(player); return true;
      case "join_room": if (player.room) return false; player.name = name(msg.name); this.joinRoom(player, msg.code); return true;
      case "ready": this.ready(player); return true;
      case "punch": return this.punch(player, msg);
      case "outcome": return this.outcome(player, msg);
      case "rematch": this.rematch(player); return true;
      default: return false;
    }
  }
}
