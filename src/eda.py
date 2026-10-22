"""Linguistic statistics for Arabic text (Phase 2)."""

import re
from collections import Counter

import pandas as pd

from src.data_preparation import normalize_arabic


SENTENCE_SPLIT = re.compile(r"[.!؟!?]+")
TOKEN_PATTERN = re.compile(r"[\u0621-\u064A]+")

FUNCTION_WORDS = {
    "في", "من", "على", "عن", "الى", "و", "ثم",
    "او", "لكن", "هذا", "هذه", "التي", "الذي",
    "مع", "بين", "كان", "قد", "لم", "لن",
    "لا", "ما", "هو", "هي"
}

FUNCTION_WORDS = {
    normalize_arabic(word) for word in FUNCTION_WORDS
}


def text_statistics(text):
    text = str(text) if pd.notna(text) else ""

    # Use normalized tokens for consistent word statistics.
    normalized = normalize_arabic(text)
    tokens = TOKEN_PATTERN.findall(normalized)

    word_count = len(tokens)
    lengths = [len(token) for token in tokens]

    # Sentence boundaries are based on original punctuation.
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT.split(text)
        if sentence.strip()
    ]

    sentence_count = len(sentences)
    unique_words = len(set(tokens))

    return {
        "word_count": word_count,
        "avg_word_length": (
            sum(lengths) / word_count if word_count else 0
        ),
        "avg_sentence_length_words": (
            word_count / sentence_count if sentence_count else 0
        ),
        "type_token_ratio": (
            unique_words / word_count if word_count else 0
        ),
        "function_word_ratio": (
            sum(word in FUNCTION_WORDS for word in tokens)
            / word_count if word_count else 0
        ),
        "punctuation_count": len(
            re.findall(r"[.!؟!?،؛,:]", text)
        ),
        "diacritics_count": len(
            re.findall(r"[\u064B-\u065F]", text)
        ),
    }


def summarize(df):
    stats = pd.DataFrame.from_records(
        df["text"].map(text_statistics)
    )

    stats["label"] = df["label"].values

    summary = (
        stats.groupby("label")
        .mean(numeric_only=True)
        .rename(index={0: "human", 1: "ai"})
    )

    return summary, stats