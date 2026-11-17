"""Transparent, explainable rule layer (EN / DE / FR).

Each rule has a weight in ]0, 1[ interpreted as the probability that a text matching it is an
injection. Scores are combined with a noisy-OR so that several weak signals add up, and every
verdict comes with the list of rules that fired - useful for audit logs and for the security team.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    weight: float


def _r(name: str, pattern: str, weight: float) -> Rule:
    return Rule(name, re.compile(pattern, re.IGNORECASE), weight)


RULES: tuple[Rule, ...] = (
    # instruction override
    _r(
        "override_en",
        r"\b(ignore|disregard|forget|skip|override)\b.{0,40}\b(previous|prior|above|earlier|all|"
        r"your|the|any)\b.{0,30}\b(instructions?|prompts?|rules?|tasks?|context|guidelines)\b",
        0.9,
    ),
    _r("forget_everything_en", r"\bforget (about )?(everything|all|that)\b", 0.8),
    _r(
        "override_de",
        r"\b(ignorier\w*|vergiss\w*|vergessen)\b.{0,40}\b(anweisung\w*|befehl\w*|vorgaben|"
        r"aufgabe\w*|alles|instruktion\w*|regeln)\b",
        0.9,
    ),
    _r("forget_everything_de", r"\b(vergiss|vergessen sie) alles\b", 0.85),
    _r(
        "override_fr",
        r"\b(ignore[rz]|oublie[rz]?)\b.{0,40}\b(instructions?|consignes?|règles|tout)\b",
        0.85,
    ),
    # new task / role hijacking
    _r("new_task", r"\b(new|next) (task|instructions?|assignment)\b", 0.6),
    _r("new_task_de", r"\bneue (aufgabe|anweisung\w*|herausforderung)\b", 0.6),
    _r(
        "role_hijack_en",
        r"\b(you are now|from now on,? you|act as|pretend (to be|you are)|imagine you are|"
        r"you will now|now you are)\b",
        0.6,
    ),
    _r(
        "role_hijack_de",
        r"\b(du bist (jetzt|nun|ab sofort)|ab (jetzt|sofort) bist du|stell dir vor,? du bist|"
        r"tu so,? als|nun bist du|jetzt bist du)\b",
        0.6,
    ),
    _r("role_hijack_fr", r"\b(tu es maintenant|fais comme si|à partir de maintenant,? tu)\b", 0.6),
    # exfiltration of the system prompt / hidden context
    _r(
        "prompt_leak",
        r"\b(show|print|reveal|repeat|output|display|tell me|write down)\b.{0,30}"
        r"\b(system prompt|your (prompt|instructions|rules)|initial (prompt|instructions)|"
        r"text above|everything above)\b",
        0.85,
    ),
    _r(
        "prompt_leak_de",
        r"\b(zeig\w*|gib\w*|wiederhole?\w*|nenne?\w*)\b.{0,30}\b(prompt|anweisungen|"
        r"text oben|obigen text)\b",
        0.75,
    ),
    # known jailbreak vocabulary
    _r(
        "jailbreak_vocab",
        r"\b(jailbreak|DAN|developer mode|do anything now|no restrictions)\b",
        0.7,
    ),
    _r("now_say", r"\b(now|then) (say|write|answer|reply|respond)\b", 0.35),
    _r("jetzt_sag", r"\b(jetzt|nun|dann) (sag|schreib|antworte)\w*\b", 0.35),
    _r("but_actually", r"\b(but|aber|mais) (actually|eigentlich|en fait)\b", 0.3),
    _r(
        "concentrate_new",
        r"\b(concentrate|focus|konzentrier\w*)\b.{0,30}\b(new|neue\w*)\b",
        0.6,
    ),
)


def matched_rules(text: str, rules: tuple[Rule, ...] = RULES) -> list[Rule]:
    return [rule for rule in rules if rule.pattern.search(text)]


def rule_score(text: str, rules: tuple[Rule, ...] = RULES) -> float:
    """Noisy-OR of the weights of the rules that fire (0 when none fire)."""
    p_clean = 1.0
    for rule in matched_rules(text, rules):
        p_clean *= 1.0 - rule.weight
    return 1.0 - p_clean


class RuleDetector:
    """sklearn-like wrapper so the rule layer can be benchmarked like a model."""

    def __init__(self, rules: tuple[Rule, ...] = RULES) -> None:
        self.rules = rules

    def fit(self, texts, labels=None) -> RuleDetector:  # noqa: ARG002 - no training
        return self

    def score(self, texts) -> list[float]:
        return [rule_score(t, self.rules) for t in texts]
