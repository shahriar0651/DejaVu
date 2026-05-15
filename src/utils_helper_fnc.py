import os
from pathlib import Path
import glob 
import shutil
import filecmp

def make_dir(srcDirRoot, destDirRoot, phaseType, tripID, dataType):
    if dataType in ['label_2', 'calib']:
        srcFolderDir = srcDirRoot / phaseType / dataType / f'{tripID}.txt'
        dstFolderDir = destDirRoot / tripID / phaseType / dataType 
        dstFolderDir.parent.mkdir(exist_ok = True, parents = True)
    else:
        srcFolderDir = srcDirRoot / phaseType / dataType / tripID
        dstFolderDir = destDirRoot / tripID / phaseType / dataType
        dstFolderDir.mkdir(exist_ok = True, parents = True)
    return {'src': srcFolderDir, 'dst': dstFolderDir}

def create_ImageSets_folder(destDirRoot, tripID):
    imageSetsDir = destDirRoot / tripID / 'ImageSets'
    imageSetsDir.mkdir(exist_ok = True, parents = True)
    # List of file names to be created
    file_names = ['train.txt', 'test.txt', 'val.txt', 'trainval.txt']
    # Creating empty files
    for file_name in file_names:
        open(imageSetsDir / file_name, 'w').close()
    # print("Empty files created successfully.")


def create_kitti_structure(srcDirRoot, destDirRoot, phaseType, tripID):
    dir_dict = {}
    if phaseType == 'training':
        dir_dict['calib'] = make_dir(srcDirRoot, destDirRoot, phaseType, tripID, 'calib')
        dir_dict['image_2'] = make_dir(srcDirRoot, destDirRoot, phaseType, tripID, 'image_2')
        dir_dict['label_2'] = make_dir(srcDirRoot, destDirRoot, phaseType, tripID, 'label_2')
        dir_dict['velodyne'] = make_dir(srcDirRoot, destDirRoot, phaseType, tripID, 'velodyne')

    elif phaseType == 'testing':
        dir_dict['calib'] = make_dir(srcDirRoot, destDirRoot, phaseType, tripID, 'calib')
        dir_dict['image_2'] = make_dir(srcDirRoot, destDirRoot, phaseType, tripID, 'image_2')
        dir_dict['velodyne'] = make_dir(srcDirRoot, destDirRoot, phaseType, tripID, 'velodyne')
        # dir_dict['testing_oxts'] = make_dir(srcDirRoot, destDirRoot, phaseType, tripID, 'oxts')
    create_ImageSets_folder(destDirRoot, tripID)
    # print(f"Kitti folder created under {dir_dict['image_2']['dst']}")
    return dir_dict
    
def compare_dirs(dir1, dir2):
    # Check if both directories exist
    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        return False
    # Compare the directory contents
    comparison = filecmp.dircmp(dir1, dir2)
    # Check if any of the common files or subdirectories differ
    if comparison.left_only or comparison.right_only or comparison.diff_files or comparison.funny_files:
        # print(comparison.left_only, comparison.right_only, comparison.diff_files, comparison.funny_files)
        return False
    # Recursively compare the common subdirectories
    for common_dir in comparison.common_dirs:
        dir1_subdir = os.path.join(dir1, common_dir)
        dir2_subdir = os.path.join(dir2, common_dir)
        if not compare_dirs(dir1_subdir, dir2_subdir):
            return False
    return True

def process_kitti_labels(label_file_path, output_dir):
    """
    Reads a KITTI label file and creates individual label text files for each frame.
    Parameters:
    - label_file_path (str or Path): Path to the input KITTI label file.
    - output_dir (str or Path): Directory where individual label files will be saved.
    """
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Read the label file
    label_file_path = Path(label_file_path)
    lines = label_file_path.read_text().splitlines()
    # Dictionary to hold data for each frame
    frames_data = {}
    for line in lines:
        # Split the line into components
        parts = line.strip().split()
        # Extract the frame number
        frame_number = int(parts[0])
        # Extract the rest of the data (category and attributes)
        category_data = ' '.join(parts[2:])
        # Append the data to the corresponding frame number
        if frame_number not in frames_data:
            frames_data[frame_number] = []
        frames_data[frame_number].append(f"{category_data}")
    # Write the data for each frame to individual files
    for frame_number, data in frames_data.items():
        frame_file_path = output_dir / f"{frame_number:06d}.txt"
        frame_file_path.write_text('\n'.join(data) + '\n')
        # print(f"frame_number : {frame_number:06d}, {frame_file_path}", )

def copy_file_dir(src_path, dest_path, test = False):
    """
    Copies a file or directory from src to dest.
    - If src_path is a file, it copies the file to dest.
    - If src_path is a directory, it copies all files within that directory to dest.
    """
    try:
        if src_path.is_file():
            # print(f"{src_path} is a file")
            if 'label' in str(src_path):
                # print("Creating labels: ")
                process_kitti_labels(src_path, dest_path)
            elif 'calib' in str(src_path):
                dest_path = dest_path / src_path.name
                dest_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(src_path, dest_path)
        elif src_path.is_dir():
            if dest_path.is_dir():
                if compare_dirs(src_path, dest_path):
                    # print(f"Files are already copied from {src_path.stem} to {dest_path.stem}")
                    return 
            for item in src_path.iterdir():
                s = item
                d = dest_path / item.name
                if s.is_file():
                    if test == False:
                        shutil.copy2(s, d)
        else:
            print(f"Source {src_path} is not compatible with {dest_path}")
    except Exception as e:
        print(f"Error copying: {e}")

def copy_calibration_files(calib_file_path, label_dir):
    """
    Copies the calib.txt file as many times as there are frames/labels within the label directory.
    Names the calib files the same as the label files (e.g., 000000.txt, 000001.txt).
    Parameters:
    - calib_file_path (str or Path): Path to the input calib.txt file.
    - label_dir (str or Path): Directory where label files are stored.
    """
    # Convert paths to Path objects
    calib_file_path = Path(calib_file_path)
    label_dir = Path(label_dir)
    # List all label files in the label directory
    label_files = sorted(label_dir.glob("*"))
    # print("label_files : ", label_files)
    # Copy the calibration file for each label file
    for label_file in label_files:
        # Create the destination path for the calibration file
        label_file.stem
        calib_dest_path = calib_file_path.parent / f'{label_file.stem}.txt'
        # Copy the calib.txt file to the destination path
        shutil.copy2(calib_file_path, calib_dest_path)
    # Delete the original calibration file
    calib_file_path.unlink()

def create_missing_labels(calib_dir, label_dir):
    """
    Creates missing labels as empty text
    """
    # Convert paths to Path objects
    calib_dir = Path(calib_dir)
    label_dir = Path(label_dir)

    list_of_label_dir = list(label_dir.iterdir())
    list_of_label_name = [file.name for file in list_of_label_dir]

    list_of_calib_dir = list(calib_dir.iterdir())
    list_of_calib_name = [file.name for file in list_of_calib_dir]

    missing_labels = list(set(list_of_calib_name) - set(list_of_label_name))

    for missing_label_name in missing_labels:
        print(missing_label_name)
        missing_label_dir = label_dir / missing_label_name
        open(missing_label_dir, 'w').close()


def complete_Imagesets_files(destDirRoot, phaseType, tripID):
    image_pattern = f'{destDirRoot}/{tripID}/{phaseType}/image_2/*'
    imageSetsDir = destDirRoot/tripID/'ImageSets'

    if phaseType == 'training':
        fileNames = ['train.txt', 'val.txt', 'trainval.txt'] 
    if phaseType == 'testing':
        fileNames = ['test.txt']

    image_list = sorted(glob.glob(image_pattern))

    for fileName in fileNames: #FIXME: Adding the same names to all the files
        # Write the list of image names to the appropriate text file
        with open(imageSetsDir/fileName, 'w') as file:
            for image_path in image_list:
                image_name_without_extension = Path(image_path).stem  # Get the name without directory and extension
                file.write(image_name_without_extension + '\n')

def add_delay_to_files(delay_dict, all_folders_dict, max_count = -1, verbose = False):
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
            if verbose:
                print(f"{count} :", temp_path, ">>>", mal_path)
            count += 1
            if count == max_count:
                break