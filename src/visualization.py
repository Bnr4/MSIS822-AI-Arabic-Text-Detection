
"""EDA charts saved to reports/figures."""

from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display


def save_charts(df, output, font_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Class distribution
    class_counts = (
        df["label"]
        .map({0: "Human", 1: "AI"})
        .value_counts()
        .reindex(["Human", "AI"], fill_value=0)
    )

    fig, ax = plt.subplots()
    class_counts.plot.bar(ax=ax)
    ax.set(
        title="Class distribution",
        ylabel="Number of texts",
        xlabel=""
    )
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Process human and AI texts separately
    for label, name in [(0, "human"), (1, "ai")]:

        # Normalized texts for N-gram analysis
        corpus = df.loc[
        df["label"] == label, "text_without_stopwords"
        ].dropna().astype(str)

        # Texts without stopwords or stemming
        cloud_corpus = df.loc[
            df["label"] == label, "text_without_stopwords"
        ].dropna().astype(str)

        word_counts = Counter(
            word
            for text in cloud_corpus
            for word in text.split()
        )

        # Arabic word cloud
        if word_counts and font_path:
            arabic_counts = {
                get_display(arabic_reshaper.reshape(word)): freq
                for word, freq in word_counts.items()
            }

            wc = WordCloud(
                font_path=str(font_path),
                width=1200,
                height=650,
                background_color="white",
                collocations=False
            ).generate_from_frequencies(arabic_counts)

            wc.to_file(
                str(output / f"wordcloud_{name}.png")
            )

        else:
            (output / f"wordcloud_{name}_note.txt").write_text(
                "Arabic font is required to generate the word cloud.\n"
            )

        # N-gram analysis
        for n in (1, 2, 3):

            if not corpus.str.strip().any():
                continue

            vectorizer = CountVectorizer(
                ngram_range=(n, n),
                token_pattern=r"(?u)\b\w+\b",
                max_features=15
            )

            matrix = vectorizer.fit_transform(corpus)
            terms = vectorizer.get_feature_names_out()
            frequencies = matrix.sum(axis=0).A1

            ranked = sorted(
                zip(terms, frequencies),
                key=lambda item: item[1]
            )[-15:]

            fig, ax = plt.subplots(figsize=(11, 6))

            ax.barh(
                [str(i + 1) for i in range(len(ranked))],
                [value for _, value in ranked]
            )

            ax.set(
                title=f"Top {n}-grams: {name} (labels in CSV)",
                xlabel="Frequency",
                ylabel="Rank"
            )

            fig.tight_layout()
            fig.savefig(
                output / f"{name}_{n}grams.png",
                dpi=160
            )
            plt.close(fig)

            pd.DataFrame(
                ranked,
                columns=["ngram", "frequency"]
            ).sort_values(
                "frequency",
                ascending=False
            ).to_csv(
                output / f"{name}_{n}grams.csv",
                index=False,
                encoding="utf-8-sig"
            )
