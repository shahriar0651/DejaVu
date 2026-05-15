from pathlib import Path
from tqdm import tqdm
import json
import hydra
from omegaconf import OmegaConf
from utils_helper_fnc import *


@hydra.main(version_base=None, config_path="../config", config_name="config")
def poison_kitti_dataset(args: dict) -> None:
    destDirRoot = Path(args.dataset.destDirRoot)
    phaseType = args.dataset.phaseType
    
    tripList = sorted([f.name for f in destDirRoot.iterdir() if f.is_dir()])
    for tripID in tripList[0:args.dataset.maxTrip]:
        image_folder = f'{args.dataset.destDirRoot}/{tripID}/{phaseType}/image_2'
        lidar_folder = f'{args.dataset.destDirRoot}/{tripID}/{phaseType}/velodyne'
        calib_folder = f'{args.dataset.destDirRoot}/{tripID}/{phaseType}/calib'
        log_file = f'{args.dataset.destDirRoot}/{tripID}/{phaseType}/rename_log.json'

        delay_dict_new = OmegaConf.to_container(args.dataset.delays, resolve=True)

        image_folder = Path(image_folder)
        lidar_folder = Path(lidar_folder)
        calib_folder = Path(calib_folder)
        log_file = Path(log_file)

        all_folders_dict = {"image": Path(image_folder), 
                            "lidar": Path(lidar_folder),
                            "calib": Path(calib_folder)}
        
        ## Moving all the backup files back to the original folder
        for folder_name, folder_dir in all_folders_dict.items():
            folder_parent = folder_dir.parent
            backup_dir = folder_parent / f"backup_{folder_name}" 
            if not backup_dir.is_dir():
                print(f"{folder_name} : No existing files in the backup folder")
                continue
            print(f"Moving files from {backup_dir}")
            backup_file_names_list = list(backup_dir.iterdir())
            for backup_file_name in tqdm(backup_file_names_list):
                file_name = backup_file_name.name
                org_file_dir = folder_dir / file_name
                backup_file_name.replace(org_file_dir)
            

        ## Renaming files to undo any previous changes
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
        delay_dict_cor = {key: -val for key,val in delay_dict_ext.items()}
        add_delay_to_files(delay_dict_cor, all_folders_dict)

        ## Adding updated delay to the targeted modalities
        add_delay_to_files(delay_dict_new, all_folders_dict)
        with open(log_file, 'w') as fp:
            json.dump(delay_dict_new, fp, indent=3)


        ## Moving unpaired files to the backup folders
        file_dir_dict = {}
        file_name_dict = {}
        file_names_set_dict = {}

        for folder_name, folder_dir in all_folders_dict.items():
            file_dir_dict[folder_name] = sorted(Path(folder_dir).iterdir())
            file_name_dict[folder_name] = {file.stem: file for file in file_dir_dict[folder_name]}
            file_names_set_dict[folder_name] = set(file_name_dict[folder_name].keys())

        common_file_name_set = file_names_set_dict["image"]
        for folder_name, folder_dir in all_folders_dict.items():
            common_file_name_set = common_file_name_set.intersection(file_names_set_dict[folder_name])

        add_unpaired_file_names_dict = {}
        for folder_name, folder_dir in all_folders_dict.items():
            add_unpaired_file_names_dict[folder_name] = file_names_set_dict[folder_name] - common_file_name_set
            print("folder_name : ", folder_name, (add_unpaired_file_names_dict[folder_name]))

        for folder_name, folder_dir in all_folders_dict.items():
            print(f"Folder: {folder_name}")
            add_file_names_list = list(add_unpaired_file_names_dict[folder_name])
            for add_file_name in tqdm(add_file_names_list):
                file_dir = file_name_dict[folder_name][add_file_name]
                file_name = file_dir.name
                folder_parent = folder_dir.parent
                backup_dir = folder_parent / f"backup_{folder_name}" 
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_file_dir = backup_dir / file_name
                file_dir.replace(backup_file_dir)
        complete_Imagesets_files(destDirRoot, phaseType, tripID)
    

if __name__ == '__main__':
    poison_kitti_dataset()