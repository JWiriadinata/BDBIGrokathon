import { test } from "node:test";
import assert from "node:assert/strict";
import { Lobby } from "../match.js";

const mk = (id) => ({ id, name: id, room: null, out: [], send(m) { this.out.push(m); } });
test("health floor + KO end", () => {
  let t = 0; const L = new Lobby(() => t), a = mk("a"), b = mk("b");
  L.handle(a, JSON.stringify({ type: "create_room", name: "a" }));
  L.handle(b, JSON.stringify({ type: "join_room", code: a.room.code, name: "b" }));
  L.ready(a); L.ready(b); t = 4000;
  // Hit -20 -> KO after 5; extra punches after KO are ignored.
  L.punch(a, { kind: "jab" }); t += 1000;
  L.outcome(b, { attackId: b.out.findLast((m) => m.type === "attack").attackId, result: "Block" });
  assert.equal(a.room.match.health.b, 95);
  for (let i = 0; i < 6; i++) {
    t += 1000; L.punch(a, { kind: "jab" });
    const id = b.out.findLast((m) => m.type === "attack").attackId;
    L.outcome(b, { attackId: id, result: "Hit" });
  }
  assert.equal(a.room.match.health.b, 0);
  const end = a.out.find((m) => m.type === "match_end");
  assert.equal(end.reason, "ko"); assert.equal(end.winner, "a");
  assert.equal(L.handle(a, "{bad json"), false);
  assert.equal(L.handle(a, "x".repeat(5000)), false);
});

test("rematch needs both players and restarts", () => {
  let t = 0; const L = new Lobby(() => t), a = mk("a"), b = mk("b");
  L.handle(a, JSON.stringify({ type: "create_room" }));
  L.handle(b, JSON.stringify({ type: "join_room", code: a.room.code }));
  L.ready(a); L.ready(b); t = 200000; L.tick(a.room);
  assert.equal(a.out.findLast((m) => m.type === "match_end").reason, "time");
  L.rematch(a);
  assert.equal(a.out.filter((m) => m.type === "match_start").length, 1);
  L.rematch(b);
  assert.equal(a.out.filter((m) => m.type === "match_start").length, 2);
});
