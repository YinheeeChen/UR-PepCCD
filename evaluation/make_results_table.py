import argparse
import csv
import json
import os


DISPLAY_METRICS = [
    ("bind_mean", "Binding Mean", True),
    ("bind_best", "Binding Best", True),
    ("iptm_mean", "ipTM", True),
    ("rt_score_mean", "RT-score", False),
    ("structure_similarity_mean", "Structure Similarity", True),
    ("structure_similarity_best", "Structure Similarity Best", True),
    ("seq_similarity_mean", "Seq Similarity", True),
    ("seq_similarity_best", "Seq Similarity Best", True),
    ("intra_sim", "Intra-Sim", False),
    ("inter_sim", "Inter-Sim", False),
    ("superior_ratio", "Superior Ratio", True),
    ("gacd_mean", "GACD", False),
    ("instability_mean", "Instability", False),
    ("bioactivity", "Bioactivity", True),
    ("diversity_edit", "Diversity", True),
    ("distinct_1", "Distinct-1", True),
    ("distinct_2", "Distinct-2", True),
    ("charge_in_range_rate", "Charge In Range", True),
    ("charge_abs_error_mean", "Charge Error", False),
    ("hotspot_mean", "Hotspot Mean", True),
    ("validity", "Validity", True),
    ("uniqueness", "Uniqueness", True),
    ("novelty", "Novelty", True),
    ("structure_eval_success_rate", "Structure Eval Success", True),
    ("seconds_per_peptide", "Sec/Peptide", False),
]


def load_summary(path):
    with open(path) as f:
        rows = json.load(f)
    return {row["model_tag"]: row for row in rows}


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_rows(summary):
    baseline = summary.get("baseline_stage3", {})
    ours = summary.get("rl_final", {})
    rows = []
    for key, label, higher_better in DISPLAY_METRICS:
        base_val = baseline.get(key)
        our_val = ours.get(key)
        delta = None
        better = ""
        if base_val is not None and our_val is not None:
            delta = our_val - base_val
            if abs(delta) > 1e-12:
                improved = delta > 0 if higher_better else delta < 0
                better = "UR-PepCCD" if improved else "Baseline"
            else:
                better = "Tie"
        rows.append({
            "metric": label,
            "baseline_stage3": base_val,
            "ur_pepccd": our_val,
            "delta": delta,
            "better": better,
        })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "baseline_stage3", "ur_pepccd", "delta", "better"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, summary):
    baseline_targets = summary["baseline_stage3"]["num_targets"]
    with open(path, "w") as f:
        f.write(f"# PepFlow Evaluation Table\n\n")
        f.write(f"Targets evaluated: {baseline_targets}\n\n")
        f.write("| Metric | Baseline Stage3 | UR-PepCCD (RL) | Delta | Better |\n")
        f.write("| --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            delta = "-" if row["delta"] is None else fmt(row["delta"])
            f.write(
                f"| {row['metric']} | {fmt(row['baseline_stage3'])} | {fmt(row['ur_pepccd'])} | {delta} | {row['better']} |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = load_summary(args.summary_json)
    rows = build_rows(summary)

    csv_path = os.path.join(args.output_dir, "results_table.csv")
    md_path = os.path.join(args.output_dir, "results_table.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, summary)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
