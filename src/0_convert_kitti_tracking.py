import hydra
from utils_helper_fnc import *

@hydra.main(version_base=None, config_path="../config", config_name="config")
def convert_kitti_tracking_to_detection(args: dict) -> None:
    test = args.dataset.test_only
    srcDirRoot = Path(args.dataset.srcDirRoot)
    destDirRoot = Path(args.dataset.destDirRoot)
    bckDirRoot = Path(args.dataset.bckDirRoot)

    tempDir = srcDirRoot / 'training' / 'image_2'
    tripIDTrain = [f.name for f in tempDir.iterdir() if f.is_dir()]

    tempDir = srcDirRoot / 'testing' / 'image_2'
    tripIDTest = [f.name for f in tempDir.iterdir() if f.is_dir()]

    tripIDCommon = sorted(list(set(tripIDTrain) & set(tripIDTest)))[0:args.dataset.maxTrip]

    directoryTree = {}
    for phaseType in ['training', 'testing']:
        # tempDir = srcDirRoot / phaseType / 'image_2'
        # dictTrip = [f for f in tempDir.iterdir() if f.is_dir()]
        dictDataType = {}
        for tripID in tripIDCommon:
            dir_dict = create_kitti_structure(srcDirRoot, destDirRoot, phaseType, tripID)
            dictDataType[tripID] = dir_dict
        directoryTree[phaseType] = dictDataType

    for phaseType, phaseData in directoryTree.items():
        for tripNo, (tripID, tripData) in enumerate(phaseData.items()):
            print(f"phaseType: {phaseType} --> tripID : {tripID}")
            if tripNo == args.dataset.maxTrip:
                break #FIXME: Remove break (using for testing only)
            for dataType, typeData in tripData.items():
                src_path = typeData['src']
                dest_path = typeData['dst']
                copy_file_dir(src_path, dest_path, test = test)
            calib_dir = destDirRoot / tripID / phaseType / 'calib' 
            path_to_calib = calib_dir / f'{tripID}.txt'
            image_dir = destDirRoot / tripID / phaseType / 'image_2'
            copy_calibration_files(path_to_calib, image_dir)            
            complete_Imagesets_files(destDirRoot, phaseType, tripID)
            if phaseType == 'training':
                label_dir = destDirRoot / tripID / phaseType / 'label_2'
                create_missing_labels(calib_dir, label_dir)
                
if __name__ == '__main__':
    convert_kitti_tracking_to_detection()