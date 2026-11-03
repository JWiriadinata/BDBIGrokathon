// Serves web/ statically and runs the match WebSocket at /ws (see AGENTS.md section 2).
import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer } from "ws";
import { Lobby } from "./match.js";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../web");
const PORT = Number(process.env.PORT) || 8080;
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
               ".onnx": "application/octet-stream", ".css": "text/css", ".png": "image/png", ".task": "application/octet-stream" };

const server = http.createServer(async (req, res) => {
  try {
    let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (p.endsWith("/")) p += "index.html";
    const file = path.join(WEB, path.normalize(p));
    if (!file.startsWith(WEB + path.sep)) { res.writeHead(403).end(); return; }
    const data = await readFile(file);
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] ?? "application/octet-stream" }).end(data);
  } catch { res.writeHead(404).end("Not found"); }
});

const lobby = new Lobby();
const wss = new WebSocketServer({ server, path: "/ws", maxPayload: 128 * 1024 });
wss.on("connection", (ws) => {
  const player = { id: `p${lobby.nextId++}`, name: "Player", room: null,
                   // Video frames are skipped when the socket is backed up, so game messages never queue behind them.
                   send: (m, droppable = false) => {
                     if (ws.readyState === 1 && !(droppable && ws.bufferedAmount > 512 * 1024)) ws.send(JSON.stringify(m));
                   } };
  ws.on("message", (data, isBinary) => {
    if (isBinary) return;
    if (!lobby.handle(player, data.toString())) player.send({ type: "error", reason: "rejected" });
  });
  ws.on("close", () => lobby.leave(player));
  ws.on("error", () => {});
});
setInterval(() => { for (const room of lobby.rooms.values()) lobby.tick(room); }, 250);

server.listen(PORT, () => console.log(`Perfect Punch on http://localhost:${PORT} (ws at /ws)`));
