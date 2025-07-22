import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def get_language_sample_counts_by_category(language_sample_count):
    """
    Returns:
        total_counts: dict of language/country -> total sample count
        by_category: dict of (language/country, category) -> sample count
    """
    total_counts = defaultdict(int)
    by_category = defaultdict(int)
    for task, info in language_sample_count.items():
        category = info.get("category")
        language_count = info.get("language_count", {})
        # flatten language_count
        if isinstance(language_count, dict):
            for k, v in language_count.items():
                if isinstance(v, dict):
                    for lang, count in v.items():
                        if count is not None:
                            total_counts[lang] += count
                            by_category[(lang, category)] += count
                else:
                    if v is not None:
                        total_counts[k] += v
                        by_category[(k, category)] += v
    return total_counts, by_category


def plot_samples_by_category(
    by_category, total_counts, output_file="samples_per_language_by_category.png"
):
    # Aggregate by language
    data = defaultdict(dict)
    for (lang, cat), count in by_category.items():
        data[lang][cat] = count
    df = pd.DataFrame(data).fillna(0).astype(int)
    df = df.T  # languages as rows
    # Order by total sample count (descending)
    order = sorted(total_counts, key=total_counts.get, reverse=True)
    df = df.loc[order]
    ax = df.plot(kind="barh", stacked=True, figsize=(14, max(7, 0.3 * len(df))))
    plt.xlabel("Samples")
    plt.ylabel("Language/Country")
    plt.title("Samples per Language/Country by Category (All Languages)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize language sample counts.")
    parser.add_argument(
        "--input",
        type=str,
        default="language_count/language_sample_count.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--top_n", type=int, default=20, help="Top N languages/countries to plot"
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        language_sample_count = json.load(f)

    total_counts, by_category = get_language_sample_counts_by_category(
        language_sample_count
    )

    plot_samples_by_category(
        by_category,
        total_counts,
        output_file="language_count/samples_per_language_by_category.png",
    )
    print(
        "Plots saved as total_samples_per_language.png and samples_per_language_by_category.png"
    )


if __name__ == "__main__":
    main()
