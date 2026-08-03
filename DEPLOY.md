# Deploying to Railway

Single service, single replica, SQLite on a mounted volume. Same shape as the
baseball app.

---

## 1. Push the repo

```bash
git add -A
git commit -m "Auth, invites, grading, CLV, history, live tracking, mobile UI"
git push
```

(If the remote isn't wired yet: `git remote add origin https://github.com/esusslin/the-algo.git`
then `git push -u origin main`.)

---

## 2. Create the Railway service

1. New Project → **Deploy from GitHub repo** → `esusslin/the-algo`
2. Nixpacks auto-detects Python and reads `Procfile`
3. **Do not let it scale past 1 replica.** `railway.toml` sets `numReplicas = 1`
   — leave it. SQLite on a shared volume with two writers corrupts, silently
   and unrecoverably.

---

## 3. Add the volume

Settings → Volumes → **New Volume**, mount path **`/data`**

Everything persistent lives there: the database, downloaded nflverse parquet,
and model artifacts. It survives redeploys; the container filesystem does not.

**Watch the size.** The weekly nflverse refresh downloads current-season parquet
into `/data/raw`. That's tens of MB now, but it accumulates. If you ever run a
full historical backfill, do it **locally** — that's the research plane's job
and it would blow out the volume.

---

## 4. Environment variables

Variables tab. Copy-paste block:

```
ENV=production
DATABASE_PATH=/data/nfl.db
DATA_DIR=/data
ARTIFACT_DIR=/data/artifacts
TZ=America/New_York
LOG_LEVEL=INFO
CURRENT_SEASON=2026

ODDS_API_KEY=<your 20K key>
ANTHROPIC_API_KEY=<your key>

JWT_SECRET_KEY=<generate a fresh one — see below>
JWT_EXPIRE_MINUTES=43200

ADMIN_USERNAME=emmet
ADMIN_PASSWORD=<8+ chars, not your usual password>
ADMIN_PHONE=+1...

ODDS_MONTHLY_CREDIT_BUDGET=20000
AI_MONTHLY_BUDGET_USD=150

PUBLISH_MODEL_PICKS=false
ENABLE_PROPS=false
ENABLE_SIMULATOR=false
ENABLE_AI_REDTEAM=true
ENABLE_SMS=false

MIN_EDGE_PCT=2.0
KELLY_FRACTION=0.125
MAX_BET_PCT_BANKROLL=2.0
SHARP_BOOKS=pinnacle
BETTABLE_BOOKS=draftkings,fanduel,betmgm,caesars,betrivers,espnbet,fanatics,betonlineag,lowvig,bovada
DEVIG_METHOD=power
```

Generate a **fresh** secret rather than reusing a local one:

```bash
python -c "import secrets;print(secrets.token_urlsafe(48))"
```

Two values to set deliberately:

- `MIN_EDGE_PCT=2.0` — at 5.0 you'd generate zero picks, since your live market
  has never exceeded 2.50%. Start at 2.0 so picks accumulate and get CLV-tracked,
  then raise it once measured CLV tells you where the real floor is.
- `BETTABLE_BOOKS` — **edit this to your actual accounts.** An edge at a book you
  can't bet is not an edge, and the default list is my guess.

---

## 5. First boot

On startup the app will:

1. Run migrations (schema v4)
2. Seed the 32-team stadium table
3. Create your admin account from `ADMIN_USERNAME` / `ADMIN_PASSWORD`
4. Register the scheduler and start polling

Check `https://<your-app>.up.railway.app/health` — expect `200` with
`"warming_up": true` for the first three hours (jobs haven't fired yet; this is
why the healthcheck doesn't fail the deploy).

Then open `/app` and sign in.

---

## 6. Seed the data

The scheduler will fill things in on its own cadence, but the first load is
faster triggered manually. From the Railway shell (`railway run bash`) or via
the admin endpoints:

```bash
python -m src.fetchers.nflverse games
python -m src.fetchers.nflverse players
python -m src.fetchers.odds_api link
python -m src.fetchers.odds_api poll
python -m src.market.consensus build
python -m src.picks.generator run
```

---

## 7. Invite your 12

Admin tab → **Create invite** → Copy link → send. Each link is single-use and
expires in 14 days. They set their own username and password.

---

## Monitoring

`/health` reports per-job staleness, not just HTTP 200 — a dead in-process
scheduler would still serve a green page otherwise. It returns 200 while the
process is alive and sets `"ok": false` when a job is overdue.

Point UptimeRobot (or Railway's own healthcheck) at it and **alert on the `ok`
field in the payload, not the status code.**

Worth watching in the first week:

| Check | Where |
|---|---|
| Credit burn vs. budget | `/api/stats`, or `source_freshness` row `odds_api` |
| Jobs actually firing | `/health` → `jobs` |
| Picks being generated | Admin tab, or `/api/picks` |
| Volume size | Railway metrics |

---

## Rollback

Railway keeps previous deploys — redeploy an earlier build from the Deployments
tab. The volume is untouched by rollbacks, so data survives.

Migrations are additive only (new tables and columns, never drops), so an older
build runs fine against a newer schema.
