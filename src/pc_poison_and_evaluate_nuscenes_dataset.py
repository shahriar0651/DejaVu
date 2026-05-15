from pathlib import Path
from tqdm import tqdm
import json
import hydra
import pickle
from omegaconf import OmegaConf
import subprocess
from hydra.utils import get_original_cwd, to_absolute_path
import torch
from utils_helper_fnc import *


@hydra.main(version_base=None, config_path="../config", config_name="config")
def poison_annot_files(args: dict) -> None:
    dataset_name = args.dataset.name
    dest_dir_root = Path(args.dataset.destDirRoot)
    delay_image = args.dataset.delays.image
    delay_lidar = args.dataset.delays.lidar
    num_gpus = torch.cuda.device_count()
    # Set the base directory
    original_cwd = str(get_original_cwd())

    for info_type in ['val']: #'train', 'test', 
        info_file_path = f'{dest_dir_root}/{dataset_name}_infos_10sweeps_{info_type}_org.pkl'
        with open(info_file_path, 'rb') as file:
            info_data = pickle.load(file)
        delayed_info_data = info_data.copy()
        print("Length of data list: ", len(info_data))
        for sample_indx in range(len(info_data)):
            
            img_target_indx = max(0, sample_indx - delay_image)
            pts_target_indx = max(0, sample_indx - delay_lidar)
            # Delay LiDAR
            delayed_info_data[sample_indx]['lidar_path'] = info_data[pts_target_indx]['lidar_path']
            # Delay Camera
            for cam_loc in info_data[sample_indx]['cams'].keys():
                delayed_info_data[sample_indx]['cams'][cam_loc]['data_path'] = delayed_info_data[img_target_indx]['cams'][cam_loc]['data_path'] 
            delayed_info_data[sample_indx]['cam_front_path'] = info_data[img_target_indx]['cam_front_path']
            


        info_file_path = f'{dest_dir_root}/{dataset_name}_infos_10sweeps_{info_type}.pkl'
        with open(info_file_path, 'wb') as file:
            pickle.dump(delayed_info_data, file)
        print(f"info_type : {info_type}")
        print(f"Saved in {info_file_path}")

        subprocess.run(
            [   
                "bash",
                "scripts/dist_test.sh",
                f"{num_gpus}",
                "--cfg_file", 
                "cfgs/nuscenes_models/bevfusion.yaml",
                "--ckpt", 
                "../checkpoints/cbgs_bevfusion.pth",
                "--eval_tag", 
                f"{delay_image}_{delay_lidar}"
            ],
                cwd=f"{original_cwd}/../OpenPCDet/tools/",
                check=True
        )

if __name__ == '__main__':
    poison_annot_files()