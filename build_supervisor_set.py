#!/usr/bin/env python3
import pandas as pd

def merge_unique(values):
    """
    Take a Series of comma-separated values and return a single
    comma-separated string of unique tokens.
    """
    all_vals = []
    for v in values.dropna():
        # split on commas and strip whitespace
        parts = [x.strip() for x in v.split(",") if x.strip()]
        all_vals.extend(parts)

    # return unique, sorted for consistency
    unique_vals = sorted(set(all_vals))
    return ", ".join(unique_vals)


def main():
    print("Loading all_projects.csv...")
    df = pd.read_csv("all_projects.csv")

    # Group by supervisor
    groups = df.groupby("primary_supervisor")

    print("Building supervisor set...")

    supervisor_set = pd.DataFrame({
        "username": groups.size().index,
        "n_projects": groups.size().values,
        "merged_keywords": groups["keywords"].apply(merge_unique).values,
        "merged_types": groups["type"].apply(merge_unique).values,
    })

    print("Saving supervisor_set.csv...")
    supervisor_set.to_csv("supervisor_set.csv", index=False)

    print("Done! ✔  Output written to supervisor_set.csv")


if __name__ == "__main__":
    main()
