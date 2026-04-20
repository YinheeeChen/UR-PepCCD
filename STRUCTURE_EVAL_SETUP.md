# Structure Eval Setup

This document prepares the environment for the three high-cost PepCCD-style metrics:

- `ipTM` or a practical AF-Multimer surrogate
- `RT-score` via Rosetta
- `Structure similarity` via TM-score / TM-align

## Current machine status

Available now:

- 4 x RTX 4090 GPUs
- Conda env `/home/gf/anaconda3/envs/af2` with:
  - `jax`
  - `openmm`
  - `jackhmmer`
  - `hhsearch`
  - `kalign`

Not available now:

- `alphafold` Python package
- `colabfold_batch`
- `rosetta_scripts`
- `score_jd2`
- `TMalign`
- `TMscore`

## Recommended environments

Keep structure evaluation separate from `PepCCD`.

Suggested envs:

- `colabfold_eval`
- `tm_eval`
- `rosetta_eval`

## 1. TM-score / TM-align

Goal:

- compute `Structure similarity`

Recommended install location:

- `/workspace/guest/cyh/tools/tmalign/`

Steps:

1. Download TM-align or TM-score binary from Zhang Lab.
2. Put executable in `/workspace/guest/cyh/tools/tmalign/`.
3. Make it executable:

```bash
mkdir -p /workspace/guest/cyh/tools/tmalign
cd /workspace/guest/cyh/tools/tmalign
chmod +x TMalign
chmod +x TMscore
```

4. Add to PATH if needed:

```bash
export PATH=/workspace/guest/cyh/tools/tmalign:$PATH
```

Expected usage:

```bash
TMalign pred_peptide.pdb native_peptide.pdb
```

## 2. ColabFold / AF-Multimer surrogate for ipTM

Goal:

- predict protein-peptide complex
- extract AF-Multimer style confidence as a practical surrogate for `ipTM`

Recommended install:

- `localcolabfold`

Suggested location:

- `/workspace/guest/cyh/tools/localcolabfold/`

Suggested environment:

```bash
conda create -n colabfold_eval python=3.10 -y
conda activate colabfold_eval
```

Then install LocalColabFold following its Linux installer.

Notes:

- LocalColabFold currently expects CUDA 12.1+.
- Use this for complex prediction on selected top candidates, not full large-scale screening.

Practical workflow:

1. Prepare FASTA with two chains:
   - target protein
   - generated peptide
2. Run multimer prediction.
3. Save:
   - predicted complex PDB
   - ranking/confidence JSON
4. Use the confidence output as the AF-Multimer-style interface metric in your table, clearly labeled as a surrogate if it is not true AF3 ipTM.

## 3. Rosetta for RT-score

Goal:

- compute `RT-score` on predicted complexes

Suggested environment:

- `rosetta_eval`

Rosetta is best installed from Rosetta Commons download instead of pip.

Needed executables after install:

- `rosetta_scripts`
- `score_jd2`

Typical compile path:

```bash
cd /path/to/rosetta/main/source
./scons.py -j 16 mode=release bin
```

After build, keep the binary path, for example:

- `/path/to/rosetta/main/source/bin/rosetta_scripts.default.linuxgccrelease`
- `/path/to/rosetta/main/source/bin/score_jd2.default.linuxgccrelease`

Practical workflow:

1. Use ColabFold/AF-Multimer to produce complex PDB.
2. Feed the complex PDB into Rosetta scoring.
3. Parse total score or the score term chosen for the paper.

## Suggested directory layout

```text
/workspace/guest/cyh/workspace/PepCCD/structure_eval/
  inputs/
    fasta/
    pdb/
  predictions/
    colabfold/
  scores/
    rosetta/
    tmalign/
  reports/
```

## Minimal evaluation strategy

Do not run structure evaluation on all generated peptides first.

Recommended:

1. Select 2 to 4 best model variants from sequence-level evaluation.
2. For each target, keep only top 1 to 3 peptides.
3. Run structure evaluation only on this reduced subset.

## What each metric needs

- `ipTM surrogate`
  - input: target + generated peptide FASTA
  - tool: ColabFold / AF-Multimer
  - output: confidence/ranking data

- `RT-score`
  - input: predicted complex PDB
  - tool: Rosetta
  - output: score table

- `Structure similarity`
  - input: predicted peptide structure + native peptide structure
  - tool: TM-align / TM-score
  - output: TM-score

## Important reporting note

If AF3 is not available and you use AF-Multimer or ColabFold instead, report it honestly as a surrogate complex-confidence metric rather than claiming true AF3 `ipTM`.
