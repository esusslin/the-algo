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
            if not self.JWT_SECRET_KEY:
                problems.append("JWT_SECRET_KEY is not set")
            if not self.ODDS_API_KEY:
                problems.append("ODDS_API_KEY is not set")
            if not self.ANTHROPIC_API_KEY:
                problems.append("ANTHROPIC_API_KEY is not set")
            if self.ENABLE_SMS and not self.TWILIO_ACCOUNT_SID:
                problems.append("ENABLE_SMS is true but Twilio is not configured")
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
