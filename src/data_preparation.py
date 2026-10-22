"""Data acquisition and Arabic preprocessing for MSIS822."""
import re
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import nltk

SOURCE = "KFUPM-JRCAI/arabic-generated-abstracts"
AI_COLUMNS = ["allam_generated_abstract", "jais_generated_abstract", "llama_generated_abstract", "openai_generated_abstract"]
TEXT_COLUMNS = ["original_abstract", *AI_COLUMNS]
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
NON_ARABIC = re.compile(r"[^\u0621-\u064A\s]")


def load_texts():
    """Convert all three subsets into a binary-labeled long-format table."""
    dataset = load_dataset(SOURCE)
    records = []
    quality = []
    for subset, split in dataset.items():
        df = split.to_pandas()
        missing = {col: int(df[col].isna().sum()) for col in TEXT_COLUMNS}
        empty = {col: int(df[col].fillna("").astype(str).str.strip().eq("").sum()) for col in TEXT_COLUMNS}
        quality.append({"subset": subset, "rows": len(df), "duplicate_rows": int(df.duplicated().sum()), "missing": missing, "empty": empty})
        for index, row in df.iterrows():
            group_id = f"{subset}:{index}"
            records.append({"group_id": group_id, "subset": subset, "source_model": "human", "label": 0, "text": row["original_abstract"]})
            for col in AI_COLUMNS:
                records.append({"group_id": group_id, "subset": subset, "source_model": col.replace("_generated_abstract", ""), "label": 1, "text": row[col]})
    return pd.DataFrame(records), pd.DataFrame(quality)


def normalize_arabic(text):
    """Normalize Arabic letters and remove diacritics/non-Arabic characters."""
    text = DIACRITICS.sub("", str(text))
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ـ", "")
    text = NON_ARABIC.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text, arabic_stopwords, stemmer):
    normalized = normalize_arabic(text)
    tokens = [t for t in normalized.split() if t not in arabic_stopwords]
    return " ".join(stemmer.stem(t) for t in tokens)


def preprocess_dataframe(df):
    nltk.download("stopwords", quiet=True)

    words = set(stopwords.words("arabic"))
    words = {normalize_arabic(word) for word in words}

    stemmer = ISRIStemmer()
    result = df.copy()

    result["normalized_text"] = result["text"].fillna("").map(normalize_arabic)

    result["text_without_stopwords"] = result["normalized_text"].map(
        lambda text: " ".join(
            token for token in text.split()
            if token not in words
        )
    )

    result["processed_text"] = result["text"].fillna("").map(
        lambda text: preprocess(text, words, stemmer)
    )

    return result