from pathlib import Path
from tqdm import tqdm
import json
import hydra
from omegaconf import OmegaConf
from utils_helper_fnc import *
import random

def add_delay_to_files_tracking(delay_dict, all_folders_dict, att_mode='constant', max_count=-1, verbose=True):
    """
    Applies frame delay attack on file names with optional randomness.
    Ensures no overlaps by tracking available 'temp_' filenames.

    Args:
        delay_dict: dict of folder_name -> delay (int)
        all_folders_dict: dict of folder_name -> Path
        mode: 'constant' or 'random'
        max_count: int, max number of files to process per folder
        verbose: bool

    Returns:
        rename_dict: dict of original file path (str) -> final renamed path (str)
    """
    rename_dict = {}

    for folder_name, delay in delay_dict.items():
        rename_dict[folder_name] = {}
        delay = int(delay)
        if delay == 0:
            continue

        print(f"Attacking {folder_name} with {'random delay' if att_mode == 'random' else f'delay of {delay} frame'}")

        target_folder = all_folders_dict[folder_name]
        target_files = sorted(f for f in target_folder.iterdir() if f.is_file() and 'temp_' not in f.name)

        all_existing_names = {f.name for f in target_folder.iterdir() if f.is_file()}
        temp_mappings = []

        # Build a list of all possible malicious names with "temp_" prefix
        available_temp_names = set()
        for f in target_files:
            # print("f : ", f)
            org_idx = int(f.stem)
            width = len(f.stem)
            # for d in range(delay+1): 
            for d in range(0, delay + (1 if delay >= 0 else -1), 1 if delay >= 0 else -1):

                mal_idx = f"{org_idx + d:0{width}d}"
                temp_name = "temp_" + mal_idx #+ f.suffix
                available_temp_names.add(temp_name)
                
        # print(f"Available temp names: ", len(available_temp_names))
        # print(f"Candidate org names: ", len(target_files))
        # # input("Press Enter to continue...")
        count = 0
        for org_indx, org_path in tqdm(enumerate(target_files)):
            # print(f"Available temp names: ", len(available_temp_names))
            # print("Remaining files: ", len(target_files) - count)
            org_name = org_path.stem
            ext = org_path.suffix
            width = len(org_name)
            base_index = int(org_name)

            if att_mode == 'random':
                delay_candidates = list(range(1, delay))
                random.shuffle(delay_candidates)
            else:
                delay_candidates = [delay]

            chosen_name = None
            for d in delay_candidates:
                new_index = base_index + d
                candidate_name = f"{new_index:0{width}d}"
                temp_candidate = "temp_" + candidate_name # + ext
                if temp_candidate in available_temp_names:
                    chosen_name = temp_candidate
                    # print(f"Yes!!! Candidate {temp_candidate} available, in {len(sorted(available_temp_names))} files")
                    available_temp_names.discard(temp_candidate)
                    # print(f"Remaining after discarding ... {len(sorted(available_temp_names))} files")

                    # # input("Press Enter to continue...")
                    break
                else:
                    print(f"Candidate {temp_candidate} not available, in {sorted(available_temp_names)} trying next...")
                    # # input("Press Enter to continue...")
            if not chosen_name:
                # Fallback: pick the first available malicious name from the sorted pool
                # print(f"From {len(available_temp_names)} files {sorted(available_temp_names)[0]} is chosen")
                fallback = sorted(available_temp_names)[0]
                chosen_name = fallback.replace(ext, "") #.replace("temp_", "").replace(ext, "")
                available_temp_names.remove(fallback)

            if not chosen_name:
                if verbose:
                    print(f"Skipping {org_path.name}: No available temp name found.")
                continue

            temp_file_name = chosen_name+ ext
            # print("temp_file_name : ", temp_file_name)
            # # input("Press Enter to continue...")
            temp_path = target_folder / temp_file_name

            org_path.rename(temp_path)
            temp_mappings.append((org_path, temp_path))

            if verbose:
                print(f"{count}: {org_path.name} >>> {temp_path.name}")

            count += 1
            if 0 < max_count <= count:
                break

        # Final renaming: remove "temp_" prefix
        for org_path, temp_path in temp_mappings:
            final_name = temp_path.name.replace("temp_", "", 1)
            final_path = temp_path.parent / final_name
            temp_path.rename(final_path)
            rename_dict[folder_name][str(org_path)] = str(final_path)
    # print("Renaming completed.")
    # print("rename_dict: ", rename_dict)
    return rename_dict

def undo_renaming(rename_dict, verbose=False):
    """
    Reverses the renaming of files based on the provided rename_dict.
    rename_dict: dict where keys are original paths and values are renamed paths (as returned by add_delay_to_files_tracking)
    """
    # print("rename_dict : ", rename_dict)
    for folder_name, rename_dict_each in rename_dict.items():
        if len(rename_dict_each) == 0:
            # print("No renaming to undo.")
            return
        # We'll reverse the mapping: new_path -> original_path
        reverse_dict = {v: k for k, v in rename_dict_each.items()}

        # Sort by descending path length to ensure we rename deeper paths first (avoids conflicts)
        for new_path_str, orig_path_str in sorted(reverse_dict.items(), key=lambda x: -len(x[0])):
            new_path = Path(new_path_str)
            orig_path = Path(orig_path_str)

            try:
                if new_path.exists():
                    new_path.rename(orig_path)
                    if verbose:
                        print(f"Restored: {new_path} >>> {orig_path}")
                else:
                    if verbose:
                        print(f"Skipped (not found): {new_path}")
            except Exception as e:
                print(f"Error renaming {new_path} back to {orig_path}: {e}")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def poison_kitti_dataset(args: dict) -> None:
    phaseType = args.dataset.phaseType
    att_mode = args.dataset.att_mode
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
        rename_file = f'{args.dataset.destTrackingDirRoot}/{phaseType}/rename_{tripID}.json'
        print("tripID : ", tripID)
        delay_dict_new = OmegaConf.to_container(args.dataset.delays, resolve=True)
        # print("delay_dict_new : ", delay_dict_new)
        # input("Press Enter to continue...")
        image_folder = Path(image_folder)
        lidar_folder = Path(lidar_folder)
        log_file = Path(log_file)
        rename_file = Path(rename_file)

        all_folders_dict = {"image": Path(image_folder), 
                            "lidar": Path(lidar_folder)}
        print("all_folders_dict :", all_folders_dict)

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

        ## Renaming files to undo any previous changes
        if Path(rename_file).exists():
            print("Undoing the previous changes")
            # print("Reading file: ", rename_file)
            with open(rename_file, 'r') as fp:
                rename_dict = json.load(fp)
            # print("Loaded rename_dict : ", rename_dict)
        else:
            print("The folder is still clean")
            rename_dict = {
                "image" : {},
                "lidar" : {}
            }

        # print("rename_dict : ", rename_dict)
        # input("Press Enter to continue...")

        #=============================================================#
        print("Reverse Step 1: Reset file names to post-attack state")
        #=============================================================#
        delay_dict_reset = {"image" : max_delay, "lidar" : max_delay}
        # add_delay_to_files_tracking(delay_dict_reset, all_folders_dict, verbose=True)
        print("delay_dict_reset : ", delay_dict_reset)
        # # input("Press Enter to continue...")
        add_delay_to_files_tracking(delay_dict_reset, all_folders_dict, att_mode='constant', max_count=-1, verbose=False)

        #=============================================================#
        print("Reverse Step 2: Bring back the files to original folders")
        # # input("Press Enter to continue...")
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
        # # input("Press Enter to continue...")
        #=============================================================#
        # delay_dict_cor = {key: -val for key,val in delay_dict_ext.items()}
        # add_delay_to_files_tracking(delay_dict_cor, all_folders_dict, verbose=True)
        # # input("Press Enter to continue...")
        undo_renaming(rename_dict, verbose=False)

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
        # # input("Press Enter to continue...")

        ## Adding updated delay to the targeted modalities
        #=============================================================#
        rename_dict = add_delay_to_files_tracking(delay_dict_new, all_folders_dict, att_mode=att_mode, max_count=-1, verbose=False)
        # print("rename_dict : ", rename_dict)

        # add_delay_to_files_tracking(delay_dict_new, all_folders_dict, verbose=True)
        # print(f"Writing {rename_dict} on {rename_file}")
        # input("Press Enter to continue...")
        with open(rename_file, 'w') as fp:
            json.dump(rename_dict, fp, indent=3)

        # # add_delay_to_files_tracking(delay_dict_new, all_folders_dict, verbose=True)
        # with open(log_file, 'w') as fp:
        #     json.dump(delay_dict_new, fp, indent=3)
            
        #=============================================================#
        print("Step 2: Move unpaired files to the backup folders")
        # # input("Press Enter to continue...")

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
        # # input("Press Enter to continue...")

        #=============================================================#
        # image_files = set([int(Path(f).stem) for f in sorted(rename_dict['image'].values())])
        # lidar_files = set([int(Path(f).stem) for f in sorted(rename_dict['lidar'].values())])
        max_delay = int(Path(sorted(unique_file_names_set)[0]).stem)
        # print("max_delay : ", max_delay)
        # input("Press Enter to continue...")
        # print("image_files : ", image_files)
        # print("lidar_files : ", lidar_files)
        # max_delay = sorted((image_files & lidar_files))[0]

        delay_dict_new = {
            "image" : max_delay,
            "lidar" : max_delay
        }
        # add_delay_to_files_tracking(delay_dict_new, all_folders_dict, verbose=True)
        with open(log_file, 'w') as fp:
            json.dump(delay_dict_new, fp, indent=3)

        ## Adding updated delay to the targeted modalities
        # max_delay = max(delay_dict_new.values())
        delay_dict_reset = {
            "image" : -max_delay,
            "lidar" : -max_delay
        }

        # print("delay_dict_reset : ", delay_dict_reset)
        # add_delay_to_files_tracking(delay_dict_reset, all_folders_dict, verbose=True)
        add_delay_to_files_tracking(delay_dict_reset, all_folders_dict, att_mode='constant', max_count=-1, verbose=False)

        # # input("Press Enter to continue...")

if __name__ == '__main__':
    poison_kitti_dataset()
    # nohup python mm_poison_and_evaluate_track_kitti_dataset.py -m dataset.delays.image=0,1,2,3,4,5 dataset.delays.lidar=0,1,2,3,4,5 dataset.att_mode=constant > logs/april_7_const.log 2>&1 &