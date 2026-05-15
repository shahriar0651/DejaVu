from pathlib import Path
from tqdm import tqdm
import json
import hydra
from omegaconf import OmegaConf
from utils_helper_fnc import *


def add_delay_to_files_tracking(delay_dict, all_folders_dict, max_count = -1, verbose = False):
    for folder_name, delay in delay_dict.items():
        if delay == 0:
            continue
        print(f"Attacking {folder_name} with delay of {delay} frame")
        target_folder = all_folders_dict[folder_name]
        target_files = sorted(target_folder.iterdir())

        # Adding delay with "temp_" prefix
        count = 0
        for _, org_path in enumerate(target_files):
            if 'temp' in str(org_path):
                continue
            org_name = org_path.stem
            ext = org_path.suffix
            width = len(org_name)
            mal_name = f"{int(org_name) + delay:0{width}d}"
            mal_file = f"{mal_name}{ext}"
            temp_mal_file = "temp_" + str(mal_file)
            temp_mal_path = Path(target_folder / temp_mal_file)
            org_path.rename(temp_mal_path)
            if verbose:
                print(f"{count} :", org_path, ">>>", temp_mal_path)

            count += 1
            if count == max_count:
                break

        # Removing "temp_" from the named files
        count = 0
        temp_files = sorted(target_folder.iterdir())
        # Iterate over files in the image folder
        for _, temp_path in enumerate(temp_files):
            if 'temp' not in str(temp_path):
                continue
            mal_path = Path(str(temp_path).replace("temp_",''))
            temp_path.rename(mal_path)
            # if verbose:
            #     print(f"{count} :", temp_path, ">>>", mal_path)
            count += 1
            if count == max_count:
                break


@hydra.main(version_base=None, config_path="../config", config_name="config")
def poison_kitti_dataset(args: dict) -> None:
    phaseType = args.dataset.phaseType
    destTrackingDirRoot = Path(args.dataset.destTrackingDirRoot) / phaseType / 'image_2'
    print("destTrackingDirRoot : ", destTrackingDirRoot)
    # tripList = sorted([f.name for f in destTrackingDirRoot.iterdir() if f.is_dir()])
    print("destTrackingDirRoot : ", destTrackingDirRoot)
    tripList = args.dataset.tracking_seq
    if args.dataset.maxTrip == -1:
        args.dataset.maxTrip = len(tripList)
    print("tripList : ", tripList)

    for tripID in tripList[0:args.dataset.maxTrip]:
        image_folder = f'{args.dataset.destTrackingDirRoot}/{phaseType}/image_2/{tripID}/'
        lidar_folder = f'{args.dataset.destTrackingDirRoot}/{phaseType}/velodyne/{tripID}/'
        log_file = f'{args.dataset.destTrackingDirRoot}/{phaseType}/rename_log_{tripID}.json'
        print("tripID : ", tripID)
        delay_dict_new = OmegaConf.to_container(args.dataset.delays, resolve=True)

        image_folder = Path(image_folder)
        lidar_folder = Path(lidar_folder)
        log_file = Path(log_file)

        all_folders_dict = {"image": Path(image_folder), 
                            "lidar": Path(lidar_folder)}
        print("all_folders_dict :", all_folders_dict)

        # ## Moving all the backup files back to the original folder
        # for folder_name, folder_dir in all_folders_dict.items():
        #     folder_parent = folder_dir.parent.parent / f"backup_{folder_name}" 
        #     backup_dir = folder_parent / tripID
        #     if not backup_dir.is_dir():
        #         print(f"{folder_name} : No existing files in the backup folder")
        #         continue
        #     print(f"Moving files from {backup_dir}")
        #     backup_file_names_list = list(backup_dir.iterdir())
        #     for backup_file_name in tqdm(backup_file_names_list):
        #         file_name = backup_file_name.name
        #         org_file_dir = folder_dir / file_name
        #         backup_file_name.replace(org_file_dir)
            

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
        print("delay_dict_ext : ", delay_dict_ext)
        max_delay = max(delay_dict_ext.values())

        #=============================================================#
        print("Reverse Step 1: Reset file names to post-attack state")
        # input("Press Enter to continue...")
        #=============================================================#
        delay_dict_reset = {"image" : max_delay, "lidar" : max_delay}
        add_delay_to_files_tracking(delay_dict_reset, all_folders_dict, verbose=True)

        #=============================================================#
        print("Reverse Step 2: Bring back the files to original folders")
        # input("Press Enter to continue...")
        ## Moving all the backup files back to the original folder
        #=============================================================#
        for folder_name, folder_dir in all_folders_dict.items():
            folder_parent = folder_dir.parent.parent / f"backup_{folder_name}" 
            backup_dir = folder_parent / tripID
            if not backup_dir.is_dir():
                print(f"{folder_name} : No existing files in the backup folder")
                continue
            print(f"Moving files from {backup_dir}")
            backup_file_names_list = list(backup_dir.iterdir())
            for backup_file_name in tqdm(backup_file_names_list):
                file_name = backup_file_name.name
                org_file_dir = folder_dir / file_name
                backup_file_name.replace(org_file_dir)
                print(f"Moved {backup_file_name} to {org_file_dir}")
        #=============================================================#
        print("Reverse Step 3: Undo the previous changes")
        # input("Press Enter to continue...")
        #=============================================================#
        delay_dict_cor = {key: -val for key,val in delay_dict_ext.items()}
        add_delay_to_files_tracking(delay_dict_cor, all_folders_dict, verbose=True)
        # input("Press Enter to continue...")

        ####################### Attack  Phase ########################

        ## Moving unpaired files to the backup folders
        unique_file_names_set_dict = {}

        for folder_name, folder_dir in all_folders_dict.items():
            unique_file_names_set_dict[folder_name] = set([file.stem for file in sorted(Path(folder_dir).iterdir())])

        unique_file_names_set = unique_file_names_set_dict["image"] and unique_file_names_set_dict["lidar"]
        print("lidar unique files : ", len(unique_file_names_set_dict["lidar"]))
        print("image uniqie files : ", len(unique_file_names_set_dict["image"]))
        assert len(unique_file_names_set) > 0, "No common files found between image and lidar folders"
        assert len(unique_file_names_set) == len(unique_file_names_set_dict["image"]), f"Common files are not same in image and lidar folders"
        assert len(unique_file_names_set) == len(unique_file_names_set_dict["lidar"]), f"Common files are not same in image and lidar folders:"

        #=============================================================#
        print("Step 1: Add delay to the files")
        # input("Press Enter to continue...")

        ## Adding updated delay to the targeted modalities
        #=============================================================#
        add_delay_to_files_tracking(delay_dict_new, all_folders_dict, verbose=True)
        with open(log_file, 'w') as fp:
            json.dump(delay_dict_new, fp, indent=3)

        #=============================================================#
        print("Step 2: Move unpaired files to the backup folders")
        # input("Press Enter to continue...")

        #=============================================================#
        file_dir_dict = {}
        file_name_dict = {}
        file_names_set_dict = {}

        for folder_name, folder_dir in all_folders_dict.items():
            file_dir_dict[folder_name] = sorted(Path(folder_dir).iterdir())
            file_name_dict[folder_name] = {file.stem: file for file in file_dir_dict[folder_name]}
            file_names_set_dict[folder_name] = set(file_name_dict[folder_name].keys())
            unique_file_names_set = unique_file_names_set.intersection(file_names_set_dict[folder_name])

        add_file_names_dict = {}
        for folder_name, folder_dir in all_folders_dict.items():
            add_file_names_list = list(file_names_set_dict[folder_name] - unique_file_names_set)
            for add_file_name in tqdm(add_file_names_list):
                file_dir = file_name_dict[folder_name][add_file_name]
                file_name = file_dir.name
                folder_parent = folder_dir.parent.parent / f"backup_{folder_name}" 
                backup_dir = folder_parent / tripID
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_file_dir = backup_dir / file_name
                file_dir.replace(backup_file_dir)
                print(f"Moved {file_dir} to {backup_file_dir}")

        # =============================================================#
        print("Step 3: Reset file names to post-attack state")
        # input("Press Enter to continue...")

        #=============================================================#

        ## Adding updated delay to the targeted modalities
        max_delay = max(delay_dict_new.values())
        delay_dict_reset = {
            "image" : -max_delay,
            "lidar" : -max_delay
        }
        print("delay_dict_reset : ", delay_dict_reset)
        add_delay_to_files_tracking(delay_dict_reset, all_folders_dict, verbose=True)
        # input("Press Enter to continue...")

if __name__ == '__main__':
    poison_kitti_dataset()