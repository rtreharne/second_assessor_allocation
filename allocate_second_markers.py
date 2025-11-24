#!/usr/bin/env python3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


###############################################################################
# LOAD INPUT DATA
###############################################################################

def load_data():
    print("Loading projects.csv, supervisor_set.csv, capacity.csv...")

    projects = pd.read_csv("projects.csv")
    supervisor_set = pd.read_csv("supervisor_set.csv")
    capacity = pd.read_csv("capacity.csv")

    # Normalise casing
    projects["Username"] = projects["Username"].str.strip().str.lower()
    supervisor_set["username"] = supervisor_set["username"].str.strip().str.lower()
    capacity["Username"] = capacity["Username"].str.strip().str.lower()

    # Compute maximum second-marking load
    capacity["max_second_mark"] = (
        capacity["Tot.Projects"].fillna(0).astype(float)
        + capacity["Difference (can be used for extra 2nd marking)"].fillna(0).astype(float)
    )

    return projects, supervisor_set, capacity


###############################################################################
# MERGE SUPERVISOR KEYWORDS/TYPES WITH CAPACITY
###############################################################################

def build_assessor_table(supervisor_set, capacity):
    assessors = supervisor_set.merge(
        capacity[["Username", "max_second_mark"]],
        left_on="username",
        right_on="Username",
        how="left"
    )

    assessors = assessors.drop(columns=["Username"])
    assessors["max_second_mark"] = assessors["max_second_mark"].fillna(0)

    return assessors


###############################################################################
# CALCULATE PROJECT → ASSESSOR SIMILARITY (TF-IDF)
###############################################################################

def build_similarity_matrix(projects, assessors):
    project_text = (
        projects["keywords_project"].fillna("") + " " +
        projects["types_project"].fillna("")
    )

    assessor_text = (
        assessors["merged_keywords"].fillna("") + " " +
        assessors["merged_types"].fillna("")
    )

    all_text = pd.concat([project_text, assessor_text], ignore_index=True)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(all_text)

    P = tfidf[:len(project_text)]
    A = tfidf[len(project_text):]

    return cosine_similarity(P, A)


###############################################################################
# ALLOCATE SECOND SUPERVISORS
###############################################################################

def allocate_assessors(projects, assessors, similarity):
    n_projects = len(projects)

    # Track remaining capacity
    capacities = dict(zip(
        assessors["username"],
        assessors["max_second_mark"]
    ))

    # Store initial max capacities for fairness scoring
    capacities_initial = capacities.copy()

    # Prevent: same assessor marking >1 project from same primary supervisor
    supervisor_pairs = {sup: set() for sup in projects["Username"].unique()}

    second_supervisor = []
    second_keywords = []
    second_types = []

    FAIRNESS_WEIGHT = 0.55
    MATCH_WEIGHT = 0.45

    for i in range(n_projects):
        primary = projects.loc[i, "Username"]

        candidates = []

        for username, sim_score in zip(assessors["username"], similarity[i]):

            # Rule 1: cannot mark own project
            if username == primary:
                continue

            # Rule 2: only one project per primary-supervisor
            if username in supervisor_pairs[primary]:
                continue

            # Rule 3: must have remaining capacity
            if capacities.get(username, 0) <= 0:
                continue

            # Compute fairness score
            max_load = capacities_initial[username]
            if max_load == 0:
                continue  # cannot be used at all

            current_load = max_load - capacities[username]
            load_ratio = current_load / max_load
            fairness_score = 1 - load_ratio  # 1 = empty, 0 = full

            # Composite score: fairness comes first
            composite_score = (
                FAIRNESS_WEIGHT * fairness_score +
                MATCH_WEIGHT * sim_score
            )

            candidates.append((username, composite_score))

        # No eligible candidates
        if not candidates:
            second_supervisor.append("UNALLOCATED")
            second_keywords.append("")
            second_types.append("")
            continue

        # Pick best composite score
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen = candidates[0][0]

        # Commit allocation
        second_supervisor.append(chosen)
        supervisor_pairs[primary].add(chosen)
        capacities[chosen] -= 1

        # Assessor metadata
        row = assessors.loc[assessors["username"] == chosen].iloc[0]
        second_keywords.append(row["merged_keywords"])
        second_types.append(row["merged_types"])

    # Add output columns
    projects["second_supervisor"] = second_supervisor
    projects["second_supervisor_keywords"] = second_keywords
    projects["second_supervisor_types"] = second_types

    # Copy project metadata
    projects["project_keywords"] = projects["keywords_project"]
    projects["project_types"] = projects["types_project"]

    return projects


###############################################################################
# MAIN
###############################################################################

def main():
    projects, supervisor_set, capacity = load_data()
    assessors = build_assessor_table(supervisor_set, capacity)

    print("Building similarity matrix...")
    similarity = build_similarity_matrix(projects, assessors)

    print("Allocating assessors...")
    result = allocate_assessors(projects, assessors, similarity)

    print("Saving: projects_with_second_assessors.csv")
    result.to_csv("projects_with_second_assessors.csv", index=False)

    ###########################################################################
    # UPDATE CAPACITY FILE WITH SECOND-MARKING LOAD
    ###########################################################################

    print("Updating capacity.csv with second-marking loads...")

    # Count how many times each username was assigned
    load_counts = (
        result[result["second_supervisor"] != "UNALLOCATED"]
        .groupby("second_supervisor")
        .size()
        .rename("second_marking_load")
    )

    capacity_updated = capacity.copy()
    capacity_updated["second_marking_load"] = capacity_updated["Username"].map(load_counts).fillna(0).astype(int)

    # Also add a remaining capacity column
    capacity_updated["remaining_capacity"] = (
        capacity_updated["max_second_mark"] - capacity_updated["second_marking_load"]
    )

    # Save the updated capacity file
    capacity_updated.to_csv("capacity_updated.csv", index=False)

    print("Saved updated capacity file as capacity_updated.csv")


    print("Done! ✔")


if __name__ == "__main__":
    main()
