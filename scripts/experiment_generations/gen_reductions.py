#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{EDDIE}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={EDDIE}/experiments/gen_reductions --use_generative_features"

gen_reductions = [
    "avg",
    "concat",
    "codebook_0",
    "codebook_1",
    "codebook_2",
    "codebook_3",
]

output_file = open("./scripts/experiments.txt", "w")

for reduction in gen_reductions:
    expt_call = (
        f"{base_call} --exp_name=reduction_{reduction} --gen_reduction={reduction} "
    )
    print(expt_call, file=output_file)

    expt_call = f"{base_call} --exp_name=reduction_{reduction}_no_cqt --gen_reduction={reduction} --no_cqt "
    print(expt_call, file=output_file)


output_file.close()
