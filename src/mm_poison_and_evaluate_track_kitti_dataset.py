import subprocess
import itertools
import sys
from pathlib import Path
from tqdm import tqdm
import json
import hydra
import pickle
from omegaconf import OmegaConf
import subprocess
from pathlib import Path
import shutil

from hydra.utils import get_original_cwd, to_absolute_path

print(f"{len(sys.argv[1:])} Arguments >> {sys.argv}")
# original_cwd = str(get_original_cwd())

# subprocess.run(['python', 'convert_kitti_tracking.py'])

python_files = [
    # 't_poison_kitti_orgtracking.py'
    't_poison_kitti_multi_orgtracking.py'
    ]
python_mot_file = 'kitti_main.py'
python_mot_env = '/home/hasan/miniforge3/envs/unitr/bin/python'

arg_list_total = []
args_dict = {}
for arguemnt in sys.argv[2:]:
    arg_parse = []
    arg = arguemnt.split("=")[0]
    vals = arguemnt.split("=")[1].split(",")
    args_dict[arg] = vals

print("args_dict : ", args_dict)

combinations = list(itertools.product(*args_dict.values()))

for combo in combinations:
    print("\n\n")
    arg_list = []
    for arg, val in zip(args_dict.keys(), combo):
        arg_list.append(f'{arg}={val}')
        print("====================\t Start Poisoning: ", arg_list, " \t====================")
    # Run the python files with the arguments
    for python_file in python_files:
        subprocess.run(
            ['python', python_file] + arg_list,
            check=True)
    print("====================\t MOT Evaluation \t====================")
    subprocess.run(
        [python_mot_env, python_mot_file],
        cwd= "../../MMF-JDT/",
        check=True)
    print("====================\t Evaluation Complete \t====================")

    # Old and new folder paths
    eval_ext = "_".join(arg_list)
    old_folder = Path("../../MMF-JDT/output")
    new_folder = Path(f"../../MMF-JDT/new_output_{eval_ext}")
    # Remove the new_folder if it already exists
    if new_folder.exists():
        shutil.rmtree(new_folder)
    old_folder.rename(new_folder)
    print(f"Output folder renamed to {new_folder}")

"""
python mm_poison_and_evaluate_detect_kitti_dataset.py -m dataset.delays.image=5,10 dataset.delays.lidar=0 && python mm_poison_and_evaluate_detect_kitti_dataset.py -m delays.lidar=5,10 dataset.delays.image=0
python mm_poison_and_evaluate_detect_kitti_dataset.py -m dataset.delays.image=10 dataset.delays.lidar=10 dataset.att_mode=constant,random && python mm_poison_and_evaluate_detect_kitti_dataset.py -m delays.lidar=5,10 dataset.delays.image=0
"""


