import argparse
import json
import os


METRIC_KEYS = [
    ("bind_mean", "Binding Mean"),
    ("bind_best", "Binding Best"),
    ("iptm_mean", "ipTM"),
    ("rt_score_mean", "RT-score"),
    ("structure_similarity_mean", "Structure Similarity"),
    ("structure_similarity_best", "Structure Similarity Best"),
    ("seq_similarity_mean", "Seq Similarity"),
    ("seq_similarity_best", "Seq Similarity Best"),
    ("intra_sim", "Intra-Sim"),
    ("inter_sim", "Inter-Sim"),
    ("superior_ratio", "Superior Ratio"),
    ("gacd_mean", "GACD"),
    ("instability_mean", "Instability"),
    ("bioactivity", "Bioactivity"),
    ("diversity_edit", "Diversity"),
    ("distinct_1", "Distinct-1"),
    ("distinct_2", "Distinct-2"),
    ("charge_in_range_rate", "Charge In Range"),
    ("charge_abs_error_mean", "Charge Error"),
    ("hotspot_mean", "Hotspot Mean"),
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
    ("structure_eval_success_rate", "Structure Eval Success"),
    ("seconds_per_peptide", "Sec/Peptide"),
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
        headers = ["Model"] + [label for _, label in METRIC_KEYS]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] + ["---:" for _ in METRIC_KEYS]) + " |\n")
        for model_tag, row in rows.items():
            values = [model_tag]
            for key, _ in METRIC_KEYS:
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
