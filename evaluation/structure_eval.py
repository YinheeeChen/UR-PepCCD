import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from tmtools import tm_align
from tmtools.io import get_residue_data


def clean_sequence(seq):
    return "".join(ch for ch in str(seq).strip().upper() if ch.isalpha())


def normalized_edit_similarity(a, b):
    a = clean_sequence(a)
    b = clean_sequence(b)
    if not a and not b:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return 1.0 - (dp[n] / max(m, n))


def parse_args():
    default_colabfold_bin = str(Path(sys.executable).resolve().parent / "colabfold_batch")
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--per_target_csv", required=True)
    parser.add_argument("--generated_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--raw_pepflow_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--colabfold_bin", default=default_colabfold_bin)
    parser.add_argument("--max_peptides_per_target", type=int, default=1)
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--num_recycle", type=int, default=3)
    parser.add_argument("--model_type", default="alphafold2_multimer_v3")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--rerun_colabfold", action="store_true")
    parser.add_argument("--progress_log", default="")
    return parser.parse_args()


def log_message(message, progress_log=""):
    line = f"[{datetime.now().strftime('%F %T')}] {message}"
    print(line, flush=True)
    if progress_log:
        with open(progress_log, "a") as f:
            f.write(line + "\n")


def candidate_key(row):
    return (row["model_tag"], row["target_id"], row["generated_peptide"])


def resolve_dataset_lookup(test_csv):
    df = pd.read_csv(test_csv)
    return {
        row["id"]: {
            "prot_seq": row["prot_seq"],
            "native_pep": row["pep_seq"],
        }
        for _, row in df.iterrows()
    }


def pick_candidates(generated_df, max_peptides_per_target):
    picked = []
    for (model_tag, target_id), group in generated_df.groupby(["model_tag", "target_id"], sort=False):
        seen = set()
        count = 0
        for row in group.itertuples(index=False):
            seq = clean_sequence(row.generated_peptide)
            if not seq or seq in seen:
                continue
            seen.add(seq)
            picked.append(
                {
                    "model_tag": model_tag,
                    "target_id": target_id,
                    "generated_peptide": seq,
                    "u_score": float(getattr(row, "u_score", 0.0)),
                }
            )
            count += 1
            if max_peptides_per_target > 0 and count >= max_peptides_per_target:
                break
    return picked


def _extract_best_json_metrics(output_dir):
    metrics = {}
    for json_path in sorted(Path(output_dir).glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("iptm", "ipTM", "ptm", "pTM", "ranking_confidence"):
                if key in data and isinstance(data[key], (int, float)):
                    metrics[key.lower()] = float(data[key])
    return metrics


def find_best_predicted_pdb(output_dir):
    candidates = sorted(Path(output_dir).glob("ranked*.pdb"))
    if not candidates:
        candidates = sorted(Path(output_dir).glob("*.pdb"))
    return candidates[0] if candidates else None


def run_colabfold_complex(prot_seq, pep_seq, output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pdb = find_best_predicted_pdb(output_dir)
    if best_pdb is not None and not args.rerun_colabfold:
        return best_pdb, _extract_best_json_metrics(output_dir)

    input_fasta = output_dir / "complex_input.fasta"
    input_fasta.write_text(f">complex\n{clean_sequence(prot_seq)}:{clean_sequence(pep_seq)}\n")
    cmd = [
        args.colabfold_bin,
        "--model-type",
        args.model_type,
        "--num-models",
        str(args.num_models),
        "--num-recycle",
        str(args.num_recycle),
        str(input_fasta),
        str(output_dir),
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))
    subprocess.run(cmd, check=True, env=env)
    best_pdb = find_best_predicted_pdb(output_dir)
    return best_pdb, _extract_best_json_metrics(output_dir)


def residue_to_one_letter(residue):
    name = str(residue.resname).strip().upper()
    if name in protein_letters_3to1:
        return protein_letters_3to1[name]
    title_name = name.capitalize()
    if title_name in protein_letters_3to1:
        return protein_letters_3to1[title_name]
    return None


def chain_sequence(chain):
    seq = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        aa = residue_to_one_letter(residue)
        if aa is not None:
            seq.append(aa)
    return "".join(seq)


def pick_predicted_peptide_chain(predicted_pdb, peptide_seq):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pred", str(predicted_pdb))
    best = None
    target = clean_sequence(peptide_seq)
    for chain in structure.get_chains():
        seq = chain_sequence(chain)
        if len(seq) < 2:
            continue
        similarity = normalized_edit_similarity(seq, target)
        length_gap = abs(len(seq) - len(target))
        score = (similarity, -length_gap)
        if best is None or score > best["score"]:
            best = {"chain": chain, "seq": seq, "score": score}
    if best is None:
        raise RuntimeError(f"Could not identify peptide chain in {predicted_pdb}")
    return best["chain"], best["seq"]


class _SingleChainSelect:
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id


def write_chain_pdb(chain, output_path):
    io = PDBIO()
    io.set_structure(chain)
    io.save(str(output_path))


def compute_structure_similarity(predicted_peptide_pdb, native_peptide_pdb):
    parser = PDBParser(QUIET=True)
    pred_structure = parser.get_structure("pred_pep", str(predicted_peptide_pdb))
    native_structure = parser.get_structure("native_pep", str(native_peptide_pdb))
    pred_chain = next(pred_structure.get_chains())
    native_chain = next(native_structure.get_chains())
    pred_coords, pred_seq = get_residue_data(pred_chain)
    native_coords, native_seq = get_residue_data(native_chain)
    result = tm_align(pred_coords, native_coords, pred_seq, native_seq)
    return float(max(result.tm_norm_chain1, result.tm_norm_chain2))


def row_status(iptm, rt_score, structure_similarity):
    available = [value is not None for value in (iptm, rt_score, structure_similarity)]
    if all(available):
        return "ok"
    if any(available):
        return "partial"
    return "failed"


_PYROSETTA_STATE = {"ready": False, "scorefxn": None}


def score_rt(predicted_complex_pdb):
    if not _PYROSETTA_STATE["ready"]:
        import pyrosetta

        pyrosetta.init("-mute all")
        _PYROSETTA_STATE["scorefxn"] = pyrosetta.get_fa_scorefxn()
        _PYROSETTA_STATE["ready"] = True
    import pyrosetta
    from pyrosetta import rosetta

    pose = pyrosetta.pose_from_pdb(str(predicted_complex_pdb))
    scorefxn = _PYROSETTA_STATE["scorefxn"]

    # Minimize the predicted complex before scoring to reduce severe clashes
    # from raw structure prediction and better match Rosetta-style total energy.
    movemap = rosetta.core.kinematics.MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(True)

    min_mover = rosetta.protocols.minimization_packing.MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn)
    min_mover.min_type("lbfgs_armijo_nonmonotone")
    min_mover.tolerance(0.01)
    min_mover.max_iter(200)
    min_mover.apply(pose)

    return float(scorefxn(pose))


def evaluate_candidate(candidate, dataset_lookup, raw_pepflow_root, structure_root, args):
    target_id = candidate["target_id"]
    model_tag = candidate["model_tag"]
    pep_seq = candidate["generated_peptide"]
    prot_seq = dataset_lookup[target_id]["prot_seq"]
    native_pdb = Path(raw_pepflow_root) / target_id / "peptide.pdb"
    target_dir = Path(structure_root) / model_tag / target_id / pep_seq
    try:
        predicted_complex_pdb, fold_metrics = run_colabfold_complex(prot_seq, pep_seq, target_dir, args)
        if predicted_complex_pdb is None:
            raise RuntimeError("ColabFold did not produce a predicted PDB")
        iptm = fold_metrics.get("iptm")
        if iptm is None:
            iptm = fold_metrics.get("ranking_confidence")
        rt_score = None
        structure_similarity = None
        extracted_peptide_pdb = ""
        errors = []

        try:
            rt_score = score_rt(predicted_complex_pdb)
        except Exception as exc:
            errors.append(f"rt_score: {exc}")

        try:
            peptide_chain, _ = pick_predicted_peptide_chain(predicted_complex_pdb, pep_seq)
            extracted_peptide_pdb = target_dir / "predicted_peptide_chain.pdb"
            write_chain_pdb(peptide_chain, extracted_peptide_pdb)
            structure_similarity = compute_structure_similarity(extracted_peptide_pdb, native_pdb)
        except Exception as exc:
            errors.append(f"structure_similarity: {exc}")

        status = row_status(iptm, rt_score, structure_similarity)
        return {
            "model_tag": model_tag,
            "target_id": target_id,
            "generated_peptide": pep_seq,
            "u_score": candidate["u_score"],
            "iptm": None if iptm is None else float(iptm),
            "rt_score": rt_score,
            "structure_similarity": structure_similarity,
            "predicted_complex_pdb": str(predicted_complex_pdb),
            "predicted_peptide_pdb": str(extracted_peptide_pdb) if extracted_peptide_pdb else "",
            "native_peptide_pdb": str(native_pdb),
            "status": status,
            "error": " | ".join(errors),
        }
    except Exception as exc:
        return {
            "model_tag": model_tag,
            "target_id": target_id,
            "generated_peptide": pep_seq,
            "u_score": candidate["u_score"],
            "iptm": None,
            "rt_score": None,
            "structure_similarity": None,
            "predicted_complex_pdb": "",
            "predicted_peptide_pdb": "",
            "native_peptide_pdb": str(native_pdb),
            "status": "failed",
            "error": str(exc),
        }


def update_summary(summary_rows, per_target_rows):
    grouped = defaultdict(list)
    for row in per_target_rows:
        grouped[row["model_tag"]].append(row)

    for summary_row in summary_rows:
        rows = grouped.get(summary_row["model_tag"], [])
        usable = [row for row in rows if float(row.get("structure_eval_success_rate", 0.0)) > 0.0]
        summary_row["structure_eval_success_rate"] = (len(usable) / len(rows)) if rows else 0.0
        if usable:
            iptm_mean_values = [float(row["iptm_mean"]) for row in usable if row.get("iptm_mean") not in ("", None)]
            iptm_best_values = [float(row["iptm_best"]) for row in usable if row.get("iptm_best") not in ("", None)]
            rt_mean_values = [float(row["rt_score_mean"]) for row in usable if row.get("rt_score_mean") not in ("", None)]
            rt_best_values = [float(row["rt_score_best"]) for row in usable if row.get("rt_score_best") not in ("", None)]
            sim_mean_values = [float(row["structure_similarity_mean"]) for row in usable if row.get("structure_similarity_mean") not in ("", None)]
            sim_best_values = [float(row["structure_similarity_best"]) for row in usable if row.get("structure_similarity_best") not in ("", None)]
            summary_row["iptm_mean"] = float(np.mean(iptm_mean_values)) if iptm_mean_values else None
            summary_row["iptm_best"] = float(np.mean(iptm_best_values)) if iptm_best_values else None
            summary_row["rt_score_mean"] = float(np.mean(rt_mean_values)) if rt_mean_values else None
            summary_row["rt_score_best"] = float(np.mean(rt_best_values)) if rt_best_values else None
            summary_row["structure_similarity_mean"] = float(np.mean(sim_mean_values)) if sim_mean_values else None
            summary_row["structure_similarity_best"] = float(np.mean(sim_best_values)) if sim_best_values else None
        else:
            summary_row["iptm_mean"] = None
            summary_row["iptm_best"] = None
            summary_row["rt_score_mean"] = None
            summary_row["rt_score_best"] = None
            summary_row["structure_similarity_mean"] = None
            summary_row["structure_similarity_best"] = None
        summary_row["structure_eval_note"] = "ColabFold_multimer_plus_PyRosetta_minimized_total_score"
    return summary_rows


def update_per_target(per_target_rows, structure_rows):
    grouped = defaultdict(list)
    for row in structure_rows:
        grouped[(row["model_tag"], row["target_id"])].append(row)

    for row in per_target_rows:
        target_rows = grouped.get((row["model_tag"], row["id"]), [])
        usable = [item for item in target_rows if item["status"] in {"ok", "partial"}]
        row["structure_eval_success_rate"] = (len(usable) / len(target_rows)) if target_rows else 0.0
        if usable:
            iptm_values = [item["iptm"] for item in usable if item["iptm"] is not None]
            rt_values = [item["rt_score"] for item in usable if item["rt_score"] is not None]
            sim_values = [item["structure_similarity"] for item in usable if item["structure_similarity"] is not None]
            row["iptm_mean"] = float(np.mean(iptm_values)) if iptm_values else None
            row["iptm_best"] = float(np.max(iptm_values)) if iptm_values else None
            row["rt_score_mean"] = float(np.mean(rt_values)) if rt_values else None
            row["rt_score_best"] = float(np.min(rt_values)) if rt_values else None
            row["structure_similarity_mean"] = float(np.mean(sim_values)) if sim_values else None
            row["structure_similarity_best"] = float(np.max(sim_values)) if sim_values else None
        else:
            row["iptm_mean"] = None
            row["iptm_best"] = None
            row["rt_score_mean"] = None
            row["rt_score_best"] = None
            row["structure_similarity_mean"] = None
            row["structure_similarity_best"] = None
    return per_target_rows


def write_structure_csv(path, structure_rows):
    fieldnames = [
        "model_tag", "target_id", "generated_peptide", "u_score", "iptm", "rt_score",
        "structure_similarity", "predicted_complex_pdb", "predicted_peptide_pdb",
        "native_peptide_pdb", "status", "error"
    ]
    if structure_rows:
        fieldnames = list(structure_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if structure_rows:
            writer.writerows(structure_rows)


def load_existing_rows(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    rows = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["model_tag"], row["target_id"], row["generated_peptide"])] = row
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_root = output_dir / "structure_eval_artifacts"
    structure_root.mkdir(parents=True, exist_ok=True)
    progress_log = args.progress_log or str(output_dir / "structure_eval_progress.log")

    summary_rows = json.loads(Path(args.summary_json).read_text())
    per_target_rows = list(csv.DictReader(open(args.per_target_csv)))
    generated_df = pd.read_csv(args.generated_csv)
    dataset_lookup = resolve_dataset_lookup(args.test_csv)
    candidates = pick_candidates(generated_df, args.max_peptides_per_target)
    structure_csv = output_dir / "structure_metrics_per_peptide.csv"
    existing_rows = load_existing_rows(structure_csv)
    structure_rows = []
    total = len(candidates)
    log_message(f"structure_eval start: {total} candidates in {output_dir}", progress_log)
    for idx, candidate in enumerate(candidates, start=1):
        key = candidate_key(candidate)
        if key in existing_rows and existing_rows[key].get("status") in {"ok", "partial"} and not args.force_rerun:
            row = existing_rows[key]
            structure_rows.append(row)
            log_message(
                f"[{idx}/{total}] skip existing {candidate['model_tag']} {candidate['target_id']} {candidate['generated_peptide']} status={row.get('status')}",
                progress_log,
            )
            continue

        log_message(
            f"[{idx}/{total}] start {candidate['model_tag']} {candidate['target_id']} {candidate['generated_peptide']}",
            progress_log,
        )
        row = evaluate_candidate(
            candidate,
            dataset_lookup=dataset_lookup,
            raw_pepflow_root=args.raw_pepflow_root,
            structure_root=structure_root,
            args=args,
        )
        structure_rows.append(row)
        log_message(
            f"[{idx}/{total}] done {candidate['model_tag']} {candidate['target_id']} {candidate['generated_peptide']} status={row['status']} iptm={row.get('iptm')} rt={row.get('rt_score')} tm={row.get('structure_similarity')}",
            progress_log,
        )
        write_structure_csv(structure_csv, structure_rows)
        per_target_snapshot = update_per_target(list(csv.DictReader(open(args.per_target_csv))), structure_rows)
        summary_snapshot = update_summary(json.loads(Path(args.summary_json).read_text()), per_target_snapshot)
        Path(args.summary_json).write_text(json.dumps(summary_snapshot, indent=2))
        if per_target_snapshot:
            with open(args.per_target_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_target_snapshot[0].keys()))
                writer.writeheader()
                writer.writerows(per_target_snapshot)

    per_target_rows = update_per_target(per_target_rows, structure_rows)
    summary_rows = update_summary(summary_rows, per_target_rows)

    Path(args.summary_json).write_text(json.dumps(summary_rows, indent=2))
    if per_target_rows:
        with open(args.per_target_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_target_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_target_rows)

    write_structure_csv(structure_csv, structure_rows)
    log_message(f"structure_eval completed: {output_dir}", progress_log)

    print(args.summary_json)
    print(args.per_target_csv)
    print(structure_csv)


if __name__ == "__main__":
    main()
