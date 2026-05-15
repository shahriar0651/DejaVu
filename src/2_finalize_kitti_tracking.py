import os
import subprocess
import hydra
from utils_helper_fnc import *

@hydra.main(version_base=None, config_path="../config", config_name="config")
def run_for_all_folders(args: dict) -> None:
    # Set the base directory
    base_dir = "../../datasets/multimodal/newTracking"
    srcDirRoot = Path(args.dataset.srcDirRoot)
    destDirRoot = Path(args.dataset.destDirRoot)
    bckDirRoot = Path(args.dataset.bckDirRoot)

    # Iterate over each directory in the base directory
    tripDirRoot_list = sorted([f for f in destDirRoot.iterdir() if f.is_dir()])

    # for folder_path in destDirRoot.iterdir():
    for folder_path in tripDirRoot_list[0:args.dataset.maxTrip]:
        if folder_path.is_dir():
            # Run the python script with the current directory
            print(f"Running for {folder_path}")
            subprocess.run([
                "python", "../mmdetection3d/tools/create_data.py", "kitti",
                "--root-path", str(folder_path),
                "--out-dir", str(folder_path),
                "--extra-tag", "kitti"
            ])
            
if __name__ == '__main__':
    run_for_all_folders()