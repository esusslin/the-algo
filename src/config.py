"""Central configuration. Every env var the app reads is declared here.

Import as:  from src.config import settings
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


# Strings that are present, pass a truthiness check, and mean nothing.
#
# `validate()` originally tested `not self.JWT_SECRET_KEY`, which the placeholder
# shipped in `.env.example` passes cleanly — so a deploy that never rotated it would
# start, log no problem, and sign tokens with a value published in a public repo.
# A check that runs, succeeds and verifies nothing is worse than no check, because
# it is also a reason not to look.
PLACEHOLDER_MARKERS = (
    "change_me", "changeme", "change-me", "placeholder", "todo",
    "your_", "your-", "xxxxxx", "sk-ant-...", "<", "example.com",
)
# `xxxxxx` rather than `xxxx`: a real `token_urlsafe` key hits four consecutive x's
# about once in 280,000, and a validator that occasionally refuses a perfectly good
# secret is one somebody eventually deletes. Six makes it ~1e-9 and still catches
# every hand-written placeholder, which run long.

PROPS_MIN_BUDGET = 60_000
"""Smallest monthly credit budget that keeps props alive for a whole month.

`scripts/simulate_odds_credits.py` measures props-on burn at ~38,000/month against the
real tier tables. `CreditLedger.allows()` drops props below 30% remaining, i.e. once
usage passes 70% of budget — so the budget must exceed 38,000 / 0.70 = 54,300. Rounded
up for a 17-game week and byes. Tested in `test_market_math.py` against the simulator,
so this cannot quietly drift away from what the schedules actually cost."""

MIN_SECRET_LENGTH = 32
"""`secrets.token_urlsafe(48)` gives 64 characters. Anything much shorter was typed
by a human, and a human-typed signing key is a guessable one."""


def _is_placeholder(value: str) -> bool:
    low = value.strip().lower()
    return any(marker in low for marker in PLACEHOLDER_MARKERS)


class Settings:
    # ---- environment ----
    ENV: str = os.getenv("ENV", "local")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    TZ: str = os.getenv("TZ", "America/New_York")
    IS_PROD: bool = ENV == "production"

    # ---- paths ----
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", "./data/nfl.db"))
    ARTIFACT_DIR: Path = Path(os.getenv("ARTIFACT_DIR", "./data/artifacts"))
    RAW_DIR: Path = DATA_DIR / "raw"
    ARCHIVE_DIR: Path = DATA_DIR / "archive"

    # ---- external APIs ----
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")

    # ---- auth ----
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = _int("JWT_EXPIRE_MINUTES", 43200)

    # ---- twilio ----
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")

    # ---- admin bootstrap ----
    ADMIN_PHONE: str = os.getenv("ADMIN_PHONE", "")
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")

    # ---- budgets ----
    ODDS_MONTHLY_CREDIT_BUDGET: int = _int("ODDS_MONTHLY_CREDIT_BUDGET", 100_000)
    AI_MONTHLY_BUDGET_USD: float = _float("AI_MONTHLY_BUDGET_USD", 150.0)

    # ---- feature flags (staged rollout — see implementation arch doc §8) ----
    PUBLISH_MODEL_PICKS: bool = _bool("PUBLISH_MODEL_PICKS", False)
    ENABLE_PROPS: bool = _bool("ENABLE_PROPS", False)
    ENABLE_SIMULATOR: bool = _bool("ENABLE_SIMULATOR", False)
    ENABLE_AI_REDTEAM: bool = _bool("ENABLE_AI_REDTEAM", True)
    ENABLE_SMS: bool = _bool("ENABLE_SMS", False)

    # ---- betting params ----
    MIN_EDGE_PCT: float = _float("MIN_EDGE_PCT", 5.0)
    KELLY_FRACTION: float = _float("KELLY_FRACTION", 0.125)
    MAX_BET_PCT_BANKROLL: float = _float("MAX_BET_PCT_BANKROLL", 2.0)
    SHARP_BOOKS: list[str] = _list("SHARP_BOOKS", "pinnacle")
    # Books you can actually place a bet at. An edge at an offshore book you
    # have no account with is not an edge. Adjust to your real accounts.
    BETTABLE_BOOKS: list[str] = _list(
        "BETTABLE_BOOKS",
        "draftkings,fanduel,betmgm,caesars,williamhill_us,betrivers,"
        "pointsbetus,espnbet,fanatics,betonlineag,lowvig,bovada,mybookieag",
    )
    DEVIG_METHOD: str = os.getenv("DEVIG_METHOD", "power")

    # ---- season ----
    CURRENT_SEASON: int = _int("CURRENT_SEASON", 2026)

    # ---- AI models ----
    MODEL_EXTRACT: str = os.getenv("MODEL_EXTRACT", "claude-haiku-4-5-20251001")
    MODEL_REASON: str = os.getenv("MODEL_REASON", "claude-sonnet-5")

    def ensure_dirs(self) -> None:
        for d in (self.DATA_DIR, self.RAW_DIR, self.ARCHIVE_DIR, self.ARTIFACT_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Return a list of config problems. Empty list == healthy.

        Called at startup; problems are logged loudly but only fatal in production.
        """
        problems: list[str] = []
        if self.IS_PROD:
            # Order matters: absent, then present-but-meaningless, then too short.
            # Only the first of these was ever checked, and it is the one a real
            # deploy is least likely to hit.
            if not self.JWT_SECRET_KEY:
                problems.append("JWT_SECRET_KEY is not set")
            elif _is_placeholder(self.JWT_SECRET_KEY):
                problems.append(
                    "JWT_SECRET_KEY is still the placeholder from .env.example — it is "
                    "published in a public repo, so anyone can forge an admin token. "
                    'Regenerate: python -c "import secrets;print(secrets.token_urlsafe(48))"')
            elif len(self.JWT_SECRET_KEY) < MIN_SECRET_LENGTH:
                problems.append(
                    f"JWT_SECRET_KEY is {len(self.JWT_SECRET_KEY)} characters; want at "
                    f"least {MIN_SECRET_LENGTH}. Short signing keys are brute-forceable")

            if not self.ODDS_API_KEY:
                problems.append("ODDS_API_KEY is not set")
            elif _is_placeholder(self.ODDS_API_KEY):
                problems.append("ODDS_API_KEY is a placeholder, not a key")

            if not self.ANTHROPIC_API_KEY:
                problems.append("ANTHROPIC_API_KEY is not set")
            elif _is_placeholder(self.ANTHROPIC_API_KEY):
                # The exact shape that shipped once: `sk-ant-...` reads as configured to
                # every truthiness check and fails on the first real call.
                problems.append("ANTHROPIC_API_KEY is a placeholder, not a key")
            if self.ENABLE_SMS and not self.TWILIO_ACCOUNT_SID:
                problems.append("ENABLE_SMS is true but Twilio is not configured")
        # Props enabled against a budget that cannot sustain them for a month.
        #
        # Neither variable is wrong on its own, which is why this went unnoticed:
        # ENABLE_PROPS=true is a decision, 20000 is a number, and only together do
        # they mean "collect props for eleven days and then stop". The ledger sheds
        # props below 30% remaining, so the budget has to clear roughly 1.43x the
        # monthly burn; measured props-on burn is ~38,000/month.
        if self.ENABLE_PROPS and self.ODDS_MONTHLY_CREDIT_BUDGET < PROPS_MIN_BUDGET:
            problems.append(
                f"ENABLE_PROPS is true but ODDS_MONTHLY_CREDIT_BUDGET is "
                f"{self.ODDS_MONTHLY_CREDIT_BUDGET:,}. The credit ladder sheds props "
                f"below 30% remaining, so props would stop collecting partway through "
                f"the month while /health stayed green. Want at least "
                f"{PROPS_MIN_BUDGET:,}, or set ENABLE_PROPS=false deliberately")

        if self.KELLY_FRACTION > 0.5:
            problems.append(
                f"KELLY_FRACTION={self.KELLY_FRACTION} is dangerously high (max recommended 0.25)"
            )
        if self.DEVIG_METHOD not in {"power", "shin", "multiplicative", "additive"}:
            problems.append(f"DEVIG_METHOD={self.DEVIG_METHOD!r} is not a known method")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
