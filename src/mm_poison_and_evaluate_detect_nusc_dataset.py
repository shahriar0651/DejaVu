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
        info_file_path = f'{dest_dir_root}/{dataset_name}_infos_{info_type}_org.pkl'
        with open(info_file_path, 'rb') as file:
            info_data = pickle.load(file)

        delayed_info_data = info_data.copy()
        
        for sample_indx in range(len(info_data['data_list'])):
            
            img_target_indx = max(0, sample_indx - delay_image)
            pts_target_indx = max(0, sample_indx - delay_lidar)
            for cam_loc in info_data['data_list'][sample_indx]['images'].keys():
                delayed_info_data['data_list'][sample_indx]['images'][cam_loc]['img_path'] = info_data['data_list'][img_target_indx]['images'][cam_loc]['img_path']
            delayed_info_data['data_list'][sample_indx]['lidar_points']['lidar_path'] = info_data['data_list'][pts_target_indx]['lidar_points']['lidar_path']
            


        info_file_path = f'{dest_dir_root}/{dataset_name}_infos_{info_type}.pkl'
        with open(info_file_path, 'wb') as file:
            pickle.dump(delayed_info_data, file)
        print(f"info_type : {info_type}")
        print(f"Saved in {info_file_path}")

        subprocess.run(
            [   
                "bash",
                "tools/dist_test.sh",
                "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
                "checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_v2.pth",
                f"{num_gpus}",
                "--cfg-options",
                f"test_evaluator.jsonfile_prefix=results_dir/bev-nus_{delay_image}_{delay_lidar}"
            ],
                cwd=f"{original_cwd}/../mmdetection3d/",
                check=True
        )


if __name__ == '__main__':
    poison_annot_files()