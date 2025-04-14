import os

USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")
REPO_HOME = f"/home/{USER}/LeadSheetTranscription"

if EDDIE is None:
    EDDIE = REPO_HOME  # Compatibility with eddie and mlt clusters

DATA_HOME = f"{EDDIE}/data/"

output_file = open("./scripts/experiments.txt", "w")

num_songs = 1
max_group_size = 200
base_call = f"python {REPO_HOME}/src/data/synthetic_data/generate_songs.py --output_dir={DATA_HOME}/synthetic_songs"

output_file = open("./scripts/experiments.txt", "w")

batch_sizes = [4,8,16,32,64]

for batch_size in batch_sizes:
    output_dir = f"batch_size_{batch_size}"
    expt = f"{base_call} --num_songs={batch_size} --batch_size={batch_size}"
    print(expt, file=output_file)


# for i in range(0, num_songs, max_group_size):
#     start_idx = i
#     group_size = min(max_group_size, num_songs - i)
#     expt_call = (
#         f"{base_call} --num_songs={group_size} "
#     )
#     print(expt_call, file=output_file)