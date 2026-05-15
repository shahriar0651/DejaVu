# # Step 1: Poison Dataset
# # Step 2: Finalize Dataset
# # Step 3: Create Individual pkl files
# # Step 4: Run inference and save figure


# #----------------- Step 3--------------------
# import pickle

# # Assuming your file name is 'data.pkl'
# file_path = '../mmdetection3d/data/kitti/kitti_infos_train.pkl'
# file_path = '/home/hasan/workspace/datasets/multimodal/tracking/newTracking/0001/kitti_infos_train.pkl'
# # Open the file in binary mode
# with open(file_path, 'rb') as file:
#     # Load the data from the file
#     data_all = pickle.load(file)
# data_all
# # Now 'data' contains the object(s) stored in the pickle file
# data_one = {}
# data_one['metainfo'] = data_all['metainfo']
# data_one['data_list'] = data_all['data_list'][0:1]
# # load pickle module
# import pickle
# single_file_path = '/home/hasan/workspace/datasets/multimodal/tracking/newTracking/0001/kitti_infos_train_000000.pkl'
# f = open(single_file_path,"wb")
# pickle.dump(data_one,f)
# f.close()


# #----------------- Step 4--------------------
# python demo/multi_modality_demo.py data/kitti/training/velodyne_reduced/000000.bin data/kitti/training/image_2/000000.png data/kitti/kitti_infos_train_000000.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth


# def create_individual_info_file():
#     return None

import json
import os
import hydra
from utils_helper_fnc import *
import pickle
import subprocess
from utils_helper_fnc import *
from omegaconf import OmegaConf

from hydra.utils import get_original_cwd

@hydra.main(version_base=None, config_path="../config", config_name="config")
def visualize_attack_impact(args: dict) -> None:

    phaseType = args.dataset.phaseType
    dest_dir_root = Path(args.dataset.destDirRoot)
    original_cwd = str(get_original_cwd())


    # Iterate over each directory in the base directory
    tripList = sorted([f for f in dest_dir_root.iterdir() if f.is_dir()])
    for trip_dir in tripList[0:args.dataset.maxTrip]:
    # for trip_dir in dest_dir_root.iterdir():

        # Extract trip ID from directory name (e.g., '0000' from '/path/to/0000')
        trip_id = trip_dir.name
        print(f"Processing trip ID: {trip_id}")

        log_file = Path(f'{trip_dir}/{phaseType}/rename_log.json')
        if Path(log_file).exists():
            print("Undoing the previous changes")
            with open(log_file, 'r') as fp:
                delay_dict_ext = json.load(fp)
        else:
            print("The folder is still clean")
            delay_dict_ext = {
                "image" : 0,
                "lidar" : 0
                }
        print("delay_dict_ext : ", delay_dict_ext)
        delay_image = delay_dict_ext['image']
        delay_lidar = delay_dict_ext['lidar']
        

        output_dir = Path(f"{trip_dir}/{args.dataset.phaseType}/vis_results")
        pred_dir = output_dir / 'preds'  
        camera_dir =  output_dir / 'camera'
        lidar_dir =  output_dir / 'lidar'
        output_dir.mkdir(exist_ok=True, parents=True)
        pred_dir.mkdir(exist_ok=True, parents=True)
        lidar_dir.mkdir(exist_ok=True, parents=True)
    
        train_file_dir = trip_dir / 'ImageSets' / 'train.txt'
        train_file_list = open(train_file_dir, 'r').read().split("\n")

        info_file_path = f'{trip_dir}/kitti_infos_train.pkl'

        with open(info_file_path, 'rb') as file:
            training_info = pickle.load(file)
        
        # Get starting frame index from config based on trip ID
        # YAML parses numeric keys as integers (e.g., 0006 -> 6), so we need to handle both string and int keys
        vis_start_config = OmegaConf.to_container(getattr(args.dataset, 'vis_start', {}), resolve=True) or {}
        
        # Try multiple key formats: string "0006", integer 6, and string "6"
        trip_id_int = int(trip_id)
        start_frame_value = (
            vis_start_config.get(trip_id) or  # Try string key "0006"
            vis_start_config.get(trip_id_int) or  # Try integer key 6
            vis_start_config.get(str(trip_id_int))  # Try string key "6"
        )
        
        if start_frame_value is None:
            start_frame_index = 0
            print(f"Starting frame index for trip {trip_id}: {start_frame_index} (default, not found in config)")
            print(f"Available keys in vis_start: {list(vis_start_config.keys())}")
        else:
            # Convert to int (handles both string "000220" and integer values)
            start_frame_index = int(start_frame_value)
            print(f"Starting frame index for trip {trip_id}: {start_frame_index} (from config: {start_frame_value})")
        
        # Calculate total samples to visualize
        total_samples = len(training_info['data_list'])
        if args.dataset.samples2vis == -1:
            print("Visualizing all the samples...")
            samples_to_vis = total_samples - start_frame_index
        else:
            samples_to_vis = args.dataset.samples2vis
        
        # Ensure we don't exceed available samples
        end_frame_index = min(start_frame_index + samples_to_vis, total_samples)
        print(f"Visualizing frames from index {start_frame_index} to {end_frame_index - 1} (total: {end_frame_index - start_frame_index} frames)")
        
        for sample_index in range(start_frame_index, end_frame_index):
            print(f"Sample index: {sample_index}")
            info_indv = {}
            info_indv['metainfo'] = training_info['metainfo']
            info_indv['data_list'] = [training_info['data_list'][sample_index]]
            file_name = train_file_list[sample_index]

            info_file_dir = Path(f'{trip_dir}/kitti_infos_train_indv/kitti_infos_train_{file_name}.pkl')
            print(f"info_indv_dir: {info_file_dir}")
            info_file_dir.parent.mkdir(exist_ok=True, parents=True)
            with open(info_file_dir,"wb") as file:
                pickle.dump(info_indv, file)
            
            # Run the code for visualization
            pcd_file_dir = f"{trip_dir}/{args.dataset.phaseType}/velodyne_reduced/{file_name}.bin"
            img_file_dir = f"{trip_dir}/{args.dataset.phaseType}/image_2/{file_name}.png"

            # output_dir_str = f"--out-dir {str(output_dir)}"

            # Get wait_time from config, default to -1 if not specified
            wait_time = args.dataset.get('wait_time', -1)
            
            subprocess.run(
                    [
                        "python",
                        "demo/multi_modality_demo.py",
                        f"{pcd_file_dir}",
                        f"{img_file_dir}",
                        f"{info_file_dir}",
                        "configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py",
                        "checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth",
                        "--show",
                        "--wait-time", str(wait_time)
                    ],
                    cwd=f"{original_cwd}/../mmdetection3d/",)
                
            org_res_file_dir = Path(f'{original_cwd}/../mmdetection3d/outputs/preds/{file_name}.json')
            org_img_file_dir = Path(f'{original_cwd}/../mmdetection3d/outputs/vis_camera/CAM2/{file_name}.png')
            org_pcd_file_dir = Path(f'{original_cwd}/../mmdetection3d/outputs/vis_lidar/{file_name}.png')

            lidar_file  = str(int(file_name) - delay_lidar).zfill(6)
            image_file  = str(int(file_name) - delay_image).zfill(6)

            new_res_file_dir = pred_dir / f'{delay_image}_{delay_lidar}_{image_file}_{lidar_file}.json'
            new_vis_file_dir = camera_dir / f'{delay_image}_{delay_lidar}_{image_file}_{lidar_file}.png'
            new_pcd_file_dir = lidar_dir / f'{delay_image}_{delay_lidar}_{image_file}_{lidar_file}.png'

            # Ensure destination directories exist
            os.makedirs(new_res_file_dir.parent, exist_ok=True)
            os.makedirs(new_vis_file_dir.parent, exist_ok=True)
            os.makedirs(new_pcd_file_dir.parent, exist_ok=True)

            # Move files only if they exist
            moved_files = []
            if org_res_file_dir.exists():
                org_res_file_dir.replace(new_res_file_dir)
                moved_files.append(f"prediction: {new_res_file_dir}")
            else:
                print(f"Warning: Prediction file not found: {org_res_file_dir}")
            
            if org_img_file_dir.exists():
                org_img_file_dir.replace(new_vis_file_dir)
                moved_files.append(f"camera: {new_vis_file_dir}")
            else:
                print(f"Warning: Camera visualization file not found: {org_img_file_dir}")
            
            if org_pcd_file_dir.exists():
                org_pcd_file_dir.replace(new_pcd_file_dir)
                moved_files.append(f"lidar: {new_pcd_file_dir}")
            else:
                print(f"Warning: LiDAR visualization file not found: {org_pcd_file_dir}")

            if moved_files:
                print(f"Result files moved: {', '.join(moved_files)}")
            else:
                print(f"Warning: No output files were created for {file_name}. The demo may not have generated outputs.")



            


    # Change  file name with trip_dir
    # python demo/multi_modality_demo.py data/kitti/training/velodyne_reduced/000000.bin data/kitti/training/image_2/000000.png data/kitti/kitti_infos_train_000000.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth
    
    # Check if ImageMagick is available
    import shutil
    convert_cmd = shutil.which('convert') or shutil.which('magick')
    
    if not convert_cmd:
        print("Warning: ImageMagick 'convert' command not found. Skipping GIF creation.")
        print("To enable GIF creation, install ImageMagick: sudo apt install imagemagick")
    else:
        for subfolder in ['camera', 'lidar']:
            print(f"Creating GIF animation for {subfolder}...")
            folder_dir = f'{output_dir}/{subfolder}'
            print("folder_dir: ", folder_dir)
            file_name = f'{output_dir}/{subfolder}/animation_{subfolder}_{delay_image}_{delay_lidar}.gif'
            
            # Check if there are any PNG files to convert
            png_files = list(Path(folder_dir).glob('*.png'))
            if not png_files:
                print(f"Warning: No PNG files found in {folder_dir}, skipping GIF creation")
                continue
            
            # Define the ImageMagick command
            # Use 'magick convert' if magick is available, otherwise just 'convert'
            if 'magick' in convert_cmd:
                command = [
                    'magick', 'convert',  # ImageMagick command (newer versions)
                    '-delay', '20',  # Frame delay
                    # '-resize', '800x400',
                    '-colors', '128', 
                    f'{folder_dir}/*.png',  # Input PNG files
                    file_name  # Output GIF file
                ]
            else:
                command = [
                    'convert',  # ImageMagick command (older versions)
                    '-delay', '20',  # Frame delay
                    # '-resize', '800x400',
                    '-colors', '128', 
                    f'{folder_dir}/*.png',  # Input PNG files
                    file_name  # Output GIF file
                ]
            
            # Run the command using subprocess
            try:
                result = subprocess.run(
                    command, 
                    cwd=f"{original_cwd}",
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"GIF created successfully: {file_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error creating GIF: {e}")
                print(f"Command output: {e.stdout}")
                print(f"Command error: {e.stderr}")
            except FileNotFoundError:
                print(f"Error: ImageMagick 'convert' command not found at: {convert_cmd}")
                print("Install ImageMagick: sudo apt install imagemagick")

if __name__ == '__main__':
    visualize_attack_impact()


"""
python 4_visualize_attacks.py -m delays.image=0 delays.lidar=5 && python 4_visualize_attacks.py -m delays.image=5 delays.lidar=0
"""