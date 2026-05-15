
import json
import hydra
from utils_helper_fnc import *
import pickle
import subprocess
from utils_helper_fnc import *
from omegaconf import OmegaConf

from hydra.utils import get_original_cwd

def create_new_dict(data_all):
    new_dict = {}
    for i, data in enumerate(data_all['data_list']):
        new_dict[i] = {}
        
        # Velodyne path
        new_dict[i]['velodyne_path'] = 'training/velodyne/' + data['lidar_points']['lidar_path']
        
        # Image details
        new_dict[i]['image'] = {
            'image_shape': (data['images']['CAM2']['height'], data['images']['CAM2']['width']),
            'image_path': 'training/image_2/' + data['images']['CAM2']['img_path'],
            'image_idx': i
        }
        
        # Calibration details
        new_dict[i]['calib'] = {
            'P0': data['images']['CAM0']['cam2img'],
            'P1': data['images']['CAM1']['cam2img'],
            'P2': data['images']['CAM2']['cam2img'],
            'P3': data['images']['CAM3']['cam2img'],
            'Tr_velo_to_cam': data['lidar_points']['Tr_velo_to_cam'],
            'Tr_imu_to_velo': data['lidar_points']['Tr_imu_to_velo'],
            'R0_rect': data['images']['R0_rect']
        }
    
    return new_dict

@hydra.main(version_base=None, config_path="../config", config_name="config")
def convert_annot_files(args: dict) -> None:

    dest_dir_root = Path(args.dataset.destDirRoot)
    tripList = sorted([f for f in dest_dir_root.iterdir() if f.is_dir()])

    for trip_dir in tripList[0:args.dataset.maxTrip]:
        print(f"trip_dir : {trip_dir}")
        for info_type in ['train', 'test', 'val', 'trainval']:
            info_file_path = f'{trip_dir}/kitti_infos_{info_type}.pkl'
            with open(info_file_path, 'rb') as file:
                kitti_infos_file = pickle.load(file)

            new_dict = create_new_dict(kitti_infos_file)
            info_file_path = f'{trip_dir}/kitti_infos_{info_type}_new.pkl'
            with open(info_file_path, 'wb') as file:
                pickle.dump(new_dict, file)
            print(f"info_type : {info_type}")

if __name__ == '__main__':
    convert_annot_files()