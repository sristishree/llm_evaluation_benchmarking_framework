"""
Shared pipeline utilities for the LLM evaluation benchmarking task bank builder.

Exported symbols
----------------
Sentinel:
    _NEEDS_REVIEW           — returned by infer_domain when auto-tagging fails

Domain inference:
    infer_domain(document)          -> str
    annotate_pending(tasks)         -> None

Difficulty signals (text-level, task-agnostic):
    _readability_score(document)    -> float   (Flesch-Kincaid grade, normalised 0–1)
    _lexical_density_score(document)-> float   (content-word ratio, normalised 0–1)
"""
from __future__ import annotations

import re
from typing import Literal

import nltk
import textstat

# ── POS tags that count as content words for lexical density ──────────────────
_CONTENT_TAGS = {
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
}

_DOMAIN_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "politics": {
        "primary": [
            "parliament", "government", "minister", "election", "vote",
            "policy", "senate", "congress", "cabinet", "legislation",
            "political", "president", "chancellor", "referendum", "debate",
            "constituency", "campaign", "ballot", "opposition", "diplomat",
            "treaty", "sanction", "summit", "foreign secretary", "prime minister",
        ],
        "secondary": [
            r"\bmp\b", r"\bmps\b", "tory", "labour", "conservative", "liberal",
            "democrat", "republican", "shadow cabinet", "manifesto",
            "by-election", "party leader", "whitehall", "downing street",
            "westminster", "general election", "coalition", "brussels",
            "nato", "united nations", r"\bun\b", "european union",
        ],
    },
    "crime": {
        "primary": [
            "police", "court", "guilty", "sentenced", "arrested", "charged",
            "convicted", "verdict", "trial", "murder", "suspect", "victim",
            "prosecution", "defendant", "detective", "investigation",
            "stabbing", "shooting", "kidnap", "smuggling",
        ],
        "secondary": [
            "prison", "jail", "manslaughter", "assault", "robbery", "theft",
            "fraud", "burglary", "arson", "homicide", "crown court",
            "magistrate", "plea", "inquest", "parole", "probation",
            "constabulary", "warrant", "custody", "sentence", "acquitted",
            "cctv", "witness", "forensic", "bail",
        ],
    },
    "sports": {
        "primary": [
            r"\bmatch\b", "league", "club", r"\bteam\b", "player", "coach",
            "manager", "tournament", "championship", "fixture", "squad",
            r"\bcup\b", "athlete", "training", "stadium", "season",
            "title", "trophy", "qualifier", "final", "semi-final",
        ],
        "secondary": [
            "football", "rugby", "cricket", "tennis", "basketball", "athletics",
            "olympic", "wimbledon", "premier", "striker", "goalkeeper",
            "referee", r"\bpitch\b", "transfer", "winger", "wicket",
            "serve", "sprint", "relay", "podium", "medal", "cycling",
            "formula one", r"\bf1\b", "golf", "swimming", "boxing",
            "racing", "derby", "innings", "penalty", "offside", "try",
        ],
    },
    "business": {
        "primary": [
            "company", r"\bmarket\b", "profit", "revenue", "investment",
            r"\bbank\b", "industry", "trade", "economy", "shares",
            "shareholders", "quarterly", "earnings", "turnover", "sector",
            "firm", "corporation", "retailer", "manufacturer", "employer",
        ],
        "secondary": [
            "gdp", "inflation", "dividend", "merger", "acquisition",
            r"\bceo\b", r"\bcfo\b", "stock market", "hedge fund",
            "interest rate", "fiscal", "monetary", "startup", "venture",
            "ipo", "bankruptcy", "recession", "supply chain", "workforce",
            "redundancy", "takeover", "listed", "ftse", "dow jones",
            "interest rates", "chancellor", "budget", "deficit",
        ],
    },
    "technology": {
        "primary": [
            r"\bsoftware\b", "digital", "internet", r"\bdata\b",
            r"\bcomputer\b", r"\bnetwork\b", r"\bplatform\b", "device",
            "online", "cyber", "cloud", "tech company", "silicon valley",
            r"\bapp\b", "website", "browser", "server",
        ],
        "secondary": [
            "algorithm", "artificial intelligence", r"\bai\b",
            "machine learning", "smartphone", r"\btech\b",
            "semiconductor", "encryption", "broadband", "robotics",
            "automation", "developer", "open source", "firmware",
            "gadget", "processor", "startup", "social media",
            "streaming", "hack", "phishing", "malware",
        ],
    },
    "health": {
        "primary": [
            "hospital", "patient", r"\bhealth\b", "medical", "treatment",
            "doctor", "disease", "surgery", r"\bnhs\b", "clinical",
            "diagnosis", "medication", "pandemic", "condition",
            "healthcare", "nurse", "therapy",
        ],
        "secondary": [
            "cancer", r"\bdrug\b", "vaccine", "mental health", "epidemic",
            "pharmaceutical", r"\bgp\b", "ward", "consultant",
            "prescription", "symptom", "oncology", "antibiotic",
            "rehabilitation", "obesity", "dementia", "diabetes",
            "heart disease", "stroke", "a&e", "ambulance", "nhs trust",
        ],
    },
    "entertainment": {
        "primary": [
            "film", "movie", "music", "album", "artist", "actor",
            "actress", "director", "television", "series", "show",
            "concert", "tour", "award", "festival", "celebrity",
            "singer", "band", "drama", "comedy",
        ],
        "secondary": [
            "bafta", "oscar", "grammy", "brit award", "box office",
            "chart", "single", "debut", "sequel", "blockbuster",
            "streaming", "netflix", "bbc one", "itv", "channel 4",
            "west end", "broadway", "theatre", "exhibition",
            "gallery", "documentary", "podcast", "bbc radio",
        ],
    },
    "science": {
        "primary": [
            "scientists", "research", "study", "discovered", "species",
            "experiment", "space", "planet", "nasa", "universe",
            "fossil", "genome", "particle", "asteroid", "probe",
        ],
        "secondary": [
            "published", "journal", "findings", "laboratory", "rover",
            "telescope", "satellite", "dna", "evolution", "biology",
            "physics", "chemistry", "astronomy", "geology", "neuroscience",
            "quantum", "vaccine trial", "clinical trial", "breakthrough",
        ],
    },
    "environment": {
        "primary": [
            "climate", "environment", "wildlife", "flood", "wildfire",
            "pollution", "carbon", "species", "habitat", "conservation",
            "renewable", "emissions", "drought", "storm", "hurricane",
        ],
        "secondary": [
            "global warming", "net zero", "greenhouse", "fossil fuel",
            "solar", "wind farm", "biodiversity", "extinction", "deforestation",
            "plastic", "recycling", "ecosystem", "coral reef", "glacier",
            "sea level", "cop26", "paris agreement", "rewilding",
        ],
    },
}

_VALID_DOMAINS = list(_DOMAIN_KEYWORDS.keys()) + ["general"]
_NEEDS_REVIEW  = "__review__"   # sentinel returned when auto-tagging fails


def _matches_any(text: str, patterns: list[str]) -> bool:
    for p in patterns:
        regex = p if r"\b" in p else rf"\b{re.escape(p)}\b"
        if re.search(regex, text, re.IGNORECASE):
            return True
    return False


def _count_matches(text: str, patterns: list[str]) -> int:
    return sum(
        1 for p in patterns
        if re.search(p if r"\b" in p else rf"\b{re.escape(p)}\b", text, re.IGNORECASE)
    )


def infer_domain(document: str) -> str:
    """
    Auto-tagging rules (first match wins):
      1. 1+ primary hit       → tag immediately
      2. 2+ secondary hits    → tag
      3. Both fail            → return _NEEDS_REVIEW sentinel
    """
    text = document[:1000].lower()

    for domain, kw in _DOMAIN_KEYWORDS.items():
        if _matches_any(text, kw["primary"]):
            return domain

    for domain, kw in _DOMAIN_KEYWORDS.items():
        if _count_matches(text, kw["secondary"]) >= 2:
            return domain

    return _NEEDS_REVIEW


def annotate_pending(tasks: list[dict]) -> None:
    """
    Batch human-annotation pass for tasks that could not be auto-tagged.
    Shows total count upfront, then prompts one by one.
    Ctrl+C or blank Enter cancels remaining → tagged "general".
    """
    pending = [t for t in tasks if t["domain"] == _NEEDS_REVIEW]
    total   = len(pending)

    if not total:
        print("No tasks need human annotation.")
        return

    print(f"\n{'='*60}")
    print(f"  {total} task(s) need human annotation.")
    print(f"  Valid domains: {_VALID_DOMAINS}")
    print(f"  Leave blank + Enter (or Ctrl+C) to cancel remaining → 'general'")
    print(f"{'='*60}\n")

    for i, task in enumerate(pending, 1):
        print(f"[{i}/{total}]  {task['input'][:300]}\n")
        try:
            answer = input(f"Domain: ").strip().lower()
            if not answer:
                raise ValueError("blank")
            task["domain"] = answer if answer in _VALID_DOMAINS else "general"
        except (KeyboardInterrupt, EOFError, ValueError):
            print(f"\nCancelled — tagging remaining {total - i + 1} item(s) as 'general'.")
            for remaining in pending[i - 1:]:
                remaining["domain"] = "general"
            return
        print()


def _readability_score(document: str) -> float:
    grade = textstat.flesch_kincaid_grade(document)
    return min(max(grade, 0.0) / 20.0, 1.0)


def _lexical_density_score(document: str) -> float:
    tokens = nltk.word_tokenize(document[:1000])
    if not tokens:
        return 0.0
    pos_tags = nltk.pos_tag(tokens)
    density = sum(1 for _, tag in pos_tags if tag in _CONTENT_TAGS) / len(tokens)
    return min(max((density - 0.3) / 0.4, 0.0), 1.0)
