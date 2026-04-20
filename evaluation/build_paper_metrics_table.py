import argparse
import json
import os


PAPER_METRIC_KEYS = [
    ("iptm_mean", "ipTM"),
    ("rt_score_mean", "RT-score"),
    ("structure_similarity_mean", "Structure Similarity"),
    ("structure_similarity_best", "Structure Similarity Best"),
    ("seq_similarity_mean", "Sequence Similarity"),
    ("seq_similarity_best", "Sequence Similarity Best"),
    ("intra_sim", "Intra-Sim"),
    ("inter_sim", "Inter-Sim"),
    ("superior_ratio", "Superior Ratio"),
    ("gacd_mean", "GACD"),
    ("instability_mean", "Instability"),
    ("bioactivity", "Bioactivity"),
    ("seconds_per_peptide", "Inference Time"),
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
    ("structure_eval_success_rate", "Structure Eval Success"),
]


def load_rows(summary_paths):
    rows = {}
    for path in summary_paths:
        with open(path) as f:
            data = json.load(f)
        for row in data:
            rows[row["model_tag"]] = row
    return rows


def write_markdown(path, rows):
    with open(path, "w") as f:
        f.write("# Paper-Aligned Metrics Table\n\n")
        f.write("Available local metrics aligned to PepCCD-style evaluation. ")
        f.write("`Bioactivity` is implemented as a reproducible sequence-level heuristic proxy when the paper's external predictor is unavailable locally. ")
        f.write("`ipTM` here is produced from the ColabFold multimer pipeline, and `RT-score` is scored with PyRosetta. ")
        f.write("`Structure Similarity` is peptide-level TM-score against the native peptide structure.\n\n")
        headers = ["Model"] + [label for _, label in PAPER_METRIC_KEYS]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] + ["---:" for _ in PAPER_METRIC_KEYS]) + " |\n")
        for model_tag, row in rows.items():
            values = [model_tag]
            for key, _ in PAPER_METRIC_KEYS:
                value = row.get(key)
                values.append("-" if value is None else f"{value:.4f}")
            f.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", nargs="+", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    rows = load_rows(args.summary_json)
    ordered = {}
    preferred_order = [
        "baseline_stage3",
        "reward_guided_legacy",
        "elbo_full",
        "elbo_no_hotspot",
        "elbo_no_implicit",
        "elbo_no_routing",
        "ur_pepccd_sota",
        "ur_pepccd_sota_ppo",
        "ur_pepccd_sota_prime_ppo",
    ]
    for key in preferred_order:
        if key in rows:
            ordered[key] = rows[key]
    for key, value in rows.items():
        if key not in ordered:
            ordered[key] = value

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    write_markdown(args.output_md, ordered)
    print(args.output_md)


if __name__ == "__main__":
    main()
