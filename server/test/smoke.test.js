// Two-client smoke test: create, join, ready, punch -> attack, outcome -> state, disconnect.
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import WebSocket from "ws";

const PORT = 18080 + Math.floor(Math.random() * 1000);

function client() {
  const ws = new WebSocket(`ws://localhost:${PORT}/ws`), inbox = [], waiters = [];
  ws.on("message", (d) => { const m = JSON.parse(d); inbox.push(m); waiters.splice(0).forEach((w) => w()); });
  const next = (type) => new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error(`timeout waiting for ${type}`)), 3000);
    const check = () => { const i = inbox.findIndex((m) => m.type === type);
      if (i >= 0) { clearTimeout(to); resolve(inbox.splice(i, 1)[0]); } else waiters.push(check); };
    check();
  });
  return { ws, next, send: (m) => ws.send(JSON.stringify(m)), open: new Promise((r) => ws.on("open", r)) };
}

test("two-client match smoke", async () => {
  const srv = spawn(process.execPath, ["server/index.js"], { env: { ...process.env, PORT: String(PORT) } });
  try {
    await new Promise((r) => srv.stdout.on("data", r));
    const res = await fetch(`http://localhost:${PORT}/`);
    assert.equal(res.status, 200);
    assert.equal((await fetch(`http://localhost:${PORT}/game.js`)).headers.get("content-type"), "text/javascript");

    const a = client(), b = client();
    await Promise.all([a.open, b.open]);
    a.send({ type: "ping", t: 1 });
    assert.equal((await a.next("pong")).t, 1);
    a.send({ type: "create_room", name: "A" });
    const { code, playerId: aId } = await a.next("room_created");
    assert.match(code, /^[A-Z0-9]{4,6}$/);
    b.send({ type: "join_room", code: "ZZZZZZ", name: "B" });
    assert.equal((await b.next("error")).reason, "unknown_room");
    b.send({ type: "join_room", code, name: "B" });
    const joined = await b.next("room_joined");
    assert.equal(joined.opponent.name, "A");
    assert.equal((await a.next("room_joined")).opponent.name, "B");
    const c = client(); await c.open;
    c.send({ type: "join_room", code, name: "C" });
    assert.equal((await c.next("error")).reason, "room_full");
    c.ws.close();

    a.send({ type: "not_a_type" });
    assert.equal((await a.next("error")).reason, "rejected");
    a.send({ type: "ready" }); b.send({ type: "ready" });
    const ms = await a.next("match_start"); await b.next("match_start");
    assert.equal(ms.durationS, 90);
    await new Promise((r) => setTimeout(r, ms.startAt - Date.now() + 50));

    a.send({ type: "punch", kind: "hook", side: "L", conf: 0.9, t: 0 });
    const atk = await b.next("attack");
    assert.equal(atk.kind, "hook"); assert.equal(atk.from, aId);
    a.send({ type: "punch", kind: "jab", side: "L", conf: 0.9, t: 0 });   // within 0.6 s -> rate limited
    assert.equal((await a.next("error")).reason, "rejected");
    b.send({ type: "outcome", attackId: atk.attackId, result: "Hit" });
    const st = await a.next("state");
    assert.equal(st.health[joined.playerId], 80); assert.equal(st.health[aId], 100);

    b.ws.close();
    assert.equal((await a.next("match_end")).reason, "disconnect");
    await a.next("opponent_left");
    a.ws.close();
  } finally { srv.kill(); }
});
