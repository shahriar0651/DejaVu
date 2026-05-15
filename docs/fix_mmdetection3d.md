- Go to line 100 of [mmdetection3d/mmdet3d/evaluation/metrics/kitti_metric.py](mmdetection3d/mmdet3d/evaluation/metrics/kitti_metric.py) and add the following line:
    ```
    cat2label = {'Pedestrian': 0,
                    'Cyclist': 1,
                    'Car': 2,
                    'Van': 3,
                    'Truck': 4,
                    'Person_sitting': 5,
                    'Tram': 6,
                    'Misc': 7,
                    'DontCare': -1}
    ```

- Go to line 717 of [mmdetection3d/mmdet3d/evaluation/functional/kitti_utils/eval.py](mmdetection3d/mmdet3d/evaluation/functional/kitti_utils/eval.py) and add the following line:
    ```
    if len(anno['alpha']) > 0 and anno['alpha'][0] != -10:

    ```


- Added scripts to OneDrive