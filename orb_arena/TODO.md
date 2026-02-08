# Orb Arena - TODO

## Features

- [ ] **Monthly High Scores Leaderboard** — Persistent top scores using SQLite (stdlib, no new deps). Write on death/disconnect, filter by month, serve via `/api/highscores`. Display on start screen and death screen. ~100 lines server, ~50 lines client. Docker volume for `.db` persistence.

## Security (OWASP Audit - Medium/Low)

- [ ] **Docker runs as root** — Add a non-root user to the Dockerfile (`RUN useradd -r gameuser && USER gameuser`)
- [ ] **Hardcoded server IP in nginx config** — Replace `REDACTED` in `game.subdomain.conf` with a hostname or Docker network alias
- [ ] **No Content Security Policy** — Add CSP headers to restrict inline scripts and external resource loading
- [ ] **Unlimited nginx body size** — Set `client_max_body_size` to a reasonable limit (e.g. `1m`) in `game.subdomain.conf`
- [ ] **Unpinned dependency version** — Pin `websockets` to a specific version in `requirements.txt` (e.g. `websockets==13.1`)
- [ ] **Predictable player IDs** — Player IDs use incrementing counters; consider UUIDs for unpredictability
