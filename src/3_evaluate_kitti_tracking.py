import os
import subprocess
import hydra
from utils_helper_fnc import *
from hydra.utils import get_original_cwd, to_absolute_path


@hydra.main(version_base=None, config_path="../config", config_name="config")
def evaluate_on_all_folders(args: dict) -> None:
    """
    Create symbolic link to the trip folder
    Run kitti evaluation one by one

    Args:
        List of folders
    Returns:
        evaluation results (saved in trip folder)
    """
     
    # Set the base directory
    dest_dir_root = Path(args.dataset.destDirRoot)
    original_cwd = str(get_original_cwd())
    data_dir = f"{original_cwd}/../mmdetection3d/data"
    kitti_dir = Path(data_dir + "/kitti")
    
    # print(f"Current working directory : {os.getcwd()}")
    # print(f"Orig working directory    : {get_original_cwd()}")

    delay_image = args.dataset.delays.image
    delay_lidar = args.dataset.delays.lidar
    
    # Iterate over each directory in the base 
    

    # Iterate over each directory in the base directory
    tripDirRoot_list = sorted([f for f in dest_dir_root.iterdir() if f.is_dir()])

    # for folder_path in destDirRoot.iterdir():
    for trip_dir in tripDirRoot_list[0:args.dataset.maxTrip]:

    # for trip_dir in dest_dir_root.iterdir():
        print(f"\n\nTripID: {trip_dir}")
        # commanD = input("Give permission...")
        # if commanD == 's':
        #     continue
        if trip_dir.is_dir():
            print(f"\n\nRunning for {trip_dir}")
            if kitti_dir.is_symlink():
                kitti_dir.unlink()

            subprocess.run(["ln", "-s", str(trip_dir), kitti_dir], check=True)
            print(["ln", "-s", str(trip_dir), kitti_dir])
            print("Link created/updated!!!")
            
            # subprocess.run(
            #         [
            #             "python",
            #             "tools/test.py",
            #             "configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py",
            #             "checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth",
            #         ],
            #         cwd=f"{original_cwd}/../mmdetection3d/",)
        
            with open(trip_dir / f"kitti_results_{delay_image}_{delay_lidar}.txt", "w", encoding='utf-8') as f:
                subprocess.run(
                    [
                        "python",
                        "tools/test.py",
                        "configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py",
                        "checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth",
                    ],
                    cwd=f"{original_cwd}/../mmdetection3d/",
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )

if __name__ == "__main__":
    evaluate_on_all_folders()


# python tools/test.py configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth