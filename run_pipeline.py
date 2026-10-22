"""Run from the project root: python run_pipeline.py"""
import argparse
from pathlib import Path
from src.data_preparation import load_texts, preprocess_dataframe
from src.eda import summarize
from src.visualization import save_charts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arabic-font", help="Path to an Arabic-supporting .ttf font for word clouds")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    processed = root / "data" / "processed"
    figures = root / "reports" / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    print("Phase 1: loading and checking dataset...")
    df, quality = load_texts()
    quality.to_json(processed / "phase1_quality.json", orient="records", indent=2, force_ascii=False)
    print(quality.to_string(index=False))
    print("Class counts:", df.label.value_counts().to_dict())
    print("Phase 2: Arabic text preprocessing...")
    df = preprocess_dataframe(df)
    df.to_csv(processed / "texts_processed.csv", index=False, encoding="utf-8-sig")
    summary, stats = summarize(df)
    summary.to_csv(processed / "class_statistics.csv", encoding="utf-8-sig")
    stats.to_csv(processed / "text_statistics.csv", index=False)
    print(summary.to_string())
    save_charts(df, figures, args.arabic_font)
    print("Done. See data/processed and reports/figures")

if __name__ == "__main__":
    main()
