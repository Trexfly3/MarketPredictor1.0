"""Alternative text ingestion, anonymization, and bounded sentiment scoring."""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import requests
import urllib3


@dataclass(frozen=True)
class NewsHeadline:
    timestamp: pd.Timestamp
    headline: str
    source: str


class FinancialNewsScraper:
    """Fetches financial headlines with bounded retries and rate-limit handling."""

    def __init__(self, urls: Optional[Iterable[str]] = None, timeout_seconds: float = 5.0, max_retries: int = 2) -> None:
        self.urls = list(urls or [])
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()

    def fetch_headlines(self) -> List[NewsHeadline]:
        if not self.urls:
            now = pd.Timestamp.utcnow()
            return [
                NewsHeadline(now, "Market profit outlook improves after technology breakthrough", "mock"),
                NewsHeadline(now, "Analysts warn deficit pressure could trigger valuation crash", "mock"),
            ]
        collected: List[NewsHeadline] = []
        for url in self.urls:
            collected.extend(self._fetch_url(url))
        return collected

    def _fetch_url(self, url: str) -> List[NewsHeadline]:
        attempts = 0
        while attempts <= self.max_retries:
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after else 60.0)
                    attempts += 1
                    continue
                response.raise_for_status()
                return self._parse_response(response.text, source=url)
            except requests.exceptions.Timeout:
                attempts += 1
                if attempts > self.max_retries:
                    return []
            except urllib3.exceptions.MaxRetryError:
                attempts += 1
                if attempts > self.max_retries:
                    return []
            except requests.exceptions.RequestException:
                return []
        return []

    def _parse_response(self, body: str, source: str) -> List[NewsHeadline]:
        text = re.sub(r"<[^>]+>", "\n", body)
        lines = [html.unescape(line).strip() for line in text.splitlines() if line.strip()]
        now = pd.Timestamp.utcnow()
        return [NewsHeadline(now, line, source) for line in lines[:25]]


class TextAnonymizer:
    """Cleans text and replaces company entities with neutral placeholders."""

    def __init__(self) -> None:
        self.substitutions: Dict[re.Pattern[str], str] = {
            re.compile(r"(?i)\b(apple|aapl)\b"): "COMPANY_A",
            re.compile(r"(?i)\b(nvidia|nvda)\b"): "COMPANY_B",
            re.compile(r"(?i)\b(tesla|tsla)\b"): "COMPANY_C",
        }

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        without_html = re.sub(r"<[^>]*>", " ", html.unescape(text))
        lowercase = without_html.lower()
        cleaned = re.sub(r"[^a-z0-9\s_$%.-]", " ", lowercase)
        return re.sub(r"\s+", " ", cleaned).strip()

    def anonymize(self, text: str) -> str:
        result = self.clean_text(text)
        for pattern, replacement in self.substitutions.items():
            result = pattern.sub(replacement, result)
        return result


class SentimentScorerEngine:
    """Converts anonymized financial text into strictly bounded sentiment scores."""

    POSITIVE_LEXICON: Mapping[str, float] = {
        "surge": 1.0,
        "breakthrough": 1.0,
        "profit": 0.8,
        "growth": 0.7,
        "beat": 0.6,
        "upgrade": 0.7,
        "rally": 0.8,
        "improves": 0.5,
    }
    NEGATIVE_LEXICON: Mapping[str, float] = {
        "lawsuit": -1.0,
        "deficit": -0.8,
        "crash": -1.0,
        "downgrade": -0.7,
        "loss": -0.7,
        "fraud": -1.0,
        "warn": -0.5,
        "pressure": -0.4,
    }

    def __init__(self, anonymizer: Optional[TextAnonymizer] = None) -> None:
        self.anonymizer = anonymizer or TextAnonymizer()

    def score_sentence(self, sentence: str) -> float:
        anonymized = self.anonymizer.anonymize(sentence)
        tokens = re.findall(r"\b[a-zA-Z_]+\b", anonymized.lower())
        raw = sum(self.POSITIVE_LEXICON.get(token, 0.0) + self.NEGATIVE_LEXICON.get(token, 0.0) for token in tokens)
        scale = max(len(tokens), 1) ** 0.5
        return float(np.tanh(raw / scale))

    def score_many(self, sentences: Iterable[str]) -> np.ndarray:
        return np.array([self.score_sentence(sentence) for sentence in sentences], dtype=float)

    def to_frame(self, records: Iterable[NewsHeadline]) -> pd.DataFrame:
        rows = [
            {"timestamp": pd.Timestamp(record.timestamp), "sentiment_score": self.score_sentence(record.headline)}
            for record in records
        ]
        if not rows:
            return pd.DataFrame(columns=["sentiment_score"], index=pd.DatetimeIndex([], name="timestamp"))
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.groupby("timestamp", as_index=True).mean(numeric_only=True).sort_index()
