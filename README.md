# Perfect Punch: 1v1 Multiplayer

A webcam boxing game. Your punches become attacks on your opponent's screen; dodge or block theirs. Each player starts at 100 health; a hit costs 20 and a block costs 5 (5 clean hits = KO). Matches end by KO, after 90 seconds, or on disconnect.

## Play online (temporary public link)
**https://limiting-specially-dishes-settings.trycloudflare.com**

1. Each player opens the link on their own computer and allows camera access.
2. Enter a name. One player presses **Create room** and shares the code.
3. The other player types the code and presses **Join**.
4. Both press **Ready**. Stand back so your head, shoulders and hips are in view.

This link is a free Cloudflare quick tunnel to the game server running on the host's Mac. It only works while that Mac is awake, online, and running both the server and the tunnel. The URL changes every time the tunnel restarts; a permanent hosted version (static site on Vercel, browsers connected peer-to-peer) is in progress.

## Run it yourself
Requires Node 20+.

```bash
git checkout ariuka
npm install
npm start          # serves the game at http://localhost:8080
```

Open http://localhost:8080 in two browser windows to test locally (both windows share one webcam).

### Let someone on another computer join
Browsers only allow the webcam on `localhost` or HTTPS, so expose the server through an HTTPS tunnel:

```bash
# macOS, no account needed
brew install cloudflared            # or download from github.com/cloudflare/cloudflared/releases
cloudflared tunnel --url http://localhost:8080
```

Share the `https://….trycloudflare.com` URL it prints. Keep both `npm start` and the tunnel running during the match.

## Tests
```bash
node --test
```

## More
- `AGENTS.md`: multiplayer plan, architecture, message protocol, and the agent team workflow.
- `CLAUDE.md`: how the single-player game, pose pipeline and punch model work.
