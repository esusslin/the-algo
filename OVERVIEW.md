# The Algo — what this actually is

Plain language. No jargon. If you've been away from this for a month, start here.

Everything else in this folder is written for someone already deep in it.

---

## The one-sentence version

It shops 23 sportsbooks for NFL bets that are mispriced, and tells you which ones
are worth taking.

---

## What it does today

Every few minutes it checks 23 sportsbooks for their prices on NFL games.

Every book builds a profit margin into its odds — that's how they make money. The
app strips that margin out to work out what the true, fair price should be. Then
it compares that fair price against what each individual book is actually
offering.

When one book is meaningfully out of line with the other 22, that's a bet worth
taking. **And it's worth taking regardless of who wins the game** — the same way
buying something for £80 that everyone else sells for £100 is a good trade
whether or not you ever use it.

From there:

1. It works out how much to stake. Deliberately small — never more than a few
   percent of your bankroll on one game, with extra caps so a bad Sunday can't
   wipe you out.
2. It ranks the pick **A**, **B** or **C**. A-tier is deliberately rare: a
   handful a week that you'd be annoyed to miss.
3. An AI reviewer reads each pick and can throw out anything that looks
   structurally wrong. It can only ever *downgrade* a pick, never promote one.
4. What survives shows up on your phone.
5. After the game it grades the result — and records whether the price you got
   beat the price the market closed at.

That last number is the real scorecard. Win/loss takes a whole season to mean
anything. "Did I get a better price than the market settled on" tells you whether
you're any good within about a month.

---

## What it deliberately does NOT do

**It does not predict football games.**

We built a model that tried. Tested it across 22 seasons of data. It was a coin
flip — no better than guessing. Every check for cheating or data leakage passed,
so the result is real.

That model is switched off. It still runs quietly in the background so we can
watch whether it ever starts working, but nobody sees its picks and nobody bets
them.

This is not a failure. Full-game NFL betting lines are the most carefully priced
numbers in sports. Thousands of smart, well-funded people work on them full time.
Being unable to beat them is the expected outcome, and knowing that for certain is
worth more than pretending otherwise.

The app makes money — if it makes money — by finding *pricing mistakes between
books*, not by knowing more about football than the market does.

---

## The technology, part by part

| What it's for | Technology | Cost |
|---|---|---|
| Live betting odds from 23 books | The Odds API | ~$30/mo |
| Every NFL play since 1999 | nflverse public data | Free |
| Weather at each stadium | Open-Meteo | Free |
| Injury reports | Sleeper API | Free |
| Live scores during games | ESPN endpoints | Free |
| The live app's database | SQLite | Free |
| The research database (546 MB) | DuckDB + Parquet | Free |
| Web server and API | FastAPI (Python) | Free |
| Running jobs on a schedule | APScheduler | Free |
| The phone interface | Alpine.js + Tailwind | Free |
| Hosting | Railway | ~$5–20/mo |
| Logins and passwords | JWT + bcrypt | Free |
| Text messages for invites | Twilio | Pennies |
| The AI pick reviewer | Claude API | ~$5/mo |
| Automated testing | pytest + GitHub Actions | Free |
| The maths | numpy, scipy, scikit-learn | Free |

**Roughly $40–60 a month**, and the odds feed is most of it.

---

## How the pieces fit together

```
  23 sportsbooks ──► odds every few minutes ──► strip out the margin
                                                       │
                                                       ▼
                                            what SHOULD this cost?
                                                       │
                                                       ▼
                                       compare to what each book offers
                                                       │
                                                       ▼
                                              found a mispricing
                                                       │
                                    ┌──────────────────┼──────────────────┐
                                    ▼                  ▼                  ▼
                              how much to bet     rank A/B/C        AI review
                                    └──────────────────┼──────────────────┘
                                                       ▼
                                                  your phone
                                                       │
                                            (after the game)
                                                       ▼
                                        grade it + record the price
                                             you got vs the close
```

There are two completely separate halves of the codebase:

- **The live app** — small, fast, runs on Railway, serves the picks. Uses SQLite.
- **The research side** — big, slow, runs only on your laptop. Chews through 27
  seasons of data to test ideas. Uses DuckDB.

They share exactly one file, which lists what the models are allowed to look at.
That file is what stops the two halves quietly drifting apart, which is a bug
class that produces confident nonsense and no error message.

---

## What's planned but not built

### Player props — the big one

Bets on individual players. "Will this receiver get more than 45.5 yards?"

This is where the real opportunity is. A sportsbook prices 500+ of these per
game, mostly automatically. Nobody is carefully thinking about the third-string
receiver. That gap is real.

The plan builds it in layers, each tested on its own:

1. **Will he play at all?** (injury status, practice reports)
2. **How many snaps?** (depends on the game being close or a blowout)
3. **How many targets or carries?** (this is where most of the randomness lives)
4. **How far does each one go?** — against *this specific* defence

The clever part: we don't need to predict the game to do this. The betting market
already tells us, accurately and for free, whether a game will be close or a
blowout. That drives how much a player is used. We take the market's word for the
game and add value only at the player level.

Needs this season's data. **Weeks 4–8 of the season.**

### First-half bets

Books work out the first-half number from the full-game number using a rough rule
of thumb, rather than pricing it properly. Predicting how a game *divides* is a
much easier problem than predicting the game.

**Needs turning on before 8 September** or there's no data to work with.

### A smarter AI reviewer

Rewritten so it fetches and checks data with code rather than being handed a wall
of text — which fixes its worst habit, throwing out picks because information was
*missing* rather than *bad*.

Paired with a weekly review where you label its decisions. Right now, if we change
how it thinks, we have no way to tell whether it got better. Labelling fixes that.

**Start collecting labels Week 1**, even before rebuilding it — because those
labels can't be collected retroactively.

---

## The honest state of things

The product that launches on 8 September is a **price-discrepancy engine with a
safety layer**. It finds real 2.5% edges today. That's a genuine product.

Nobody should describe it as predicting games, because it doesn't.

Everything else in this folder is a plan for what to build *after* four weeks of
real data tells us where the edge actually lives — rather than guessing now and
finding out we were wrong in November.

---

## Where to go next

| You want to know | Read |
|---|---|
| How the live pipeline works | `ALGORITHM.md` |
| What data we have and where | `DATA_INVENTORY.md` |
| The plan for August and September | `PLAN_AUG_SEPT.md` |
| The player-props design | `PROP_ENGINE_DESIGN.md` |
| The AI reviewer redesign | `AGENT_LAYER.md` |
| How to deploy | `DEPLOY.md` |
| Task list to launch | `ROADMAP.md` |
