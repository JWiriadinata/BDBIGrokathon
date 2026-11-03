// Client network layer for versus mode (protocol in AGENTS.md section 2).
export class Net {
  constructor(url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`) {
    this.url = url; this.handlers = {}; this.offset = 0; this.bestRtt = Infinity; this.ws = null;
  }
  on(type, fn) { this.handlers[type] = fn; return this; }
  send(type, body = {}) { if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ type, ...body })); }
  // Server epoch ms, estimated from the lowest-RTT ping/pong.
  serverNow() { return Date.now() + this.offset; }
  connect() {
    return new Promise((resolve, reject) => {
      const ws = this.ws = new WebSocket(this.url);
      ws.onopen = () => {
        for (let i = 0; i < 5; i++) setTimeout(() => this.send("ping", { t: Date.now() }), i * 100);
        resolve();
      };
      ws.onerror = () => reject(new Error("Could not connect to the game server"));
      ws.onclose = () => this.handlers.close?.();
      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === "pong") {
          const now = Date.now(), rtt = now - msg.t;
          if (rtt < this.bestRtt) { this.bestRtt = rtt; this.offset = msg.serverT - (msg.t + rtt / 2); }
          return;
        }
        this.handlers[msg.type]?.(msg);
      };
    });
  }
  close() { this.ws?.close(); }
}
