```sh
pip install --upgrade mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA}/torch${PYTORCH}/index.html
```


```sh
pip install --upgrade mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA}/torch${PYTORCH}/index.html
```


https://mmcv.readthedocs.io/en/latest/get_started/installation.html
pip install mmcv-full==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
https://pytorch.org/get-started/previous-versions/


pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html


```sh
python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
```


pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html


Not working!!!
```sh
python demo/multi_modality_demo.py demo/data/kitti/kitti_000008.bin demo/data/kitti/kitti_000008.png demo/data/kitti/kitti_000008_infos.pkl configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth --show
```


```sh
python demo/multi_modality_demo.py demo/data/kitti/000008.bin demo/data/kitti/000008.png demo/data/kitti/000008.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth --show
```

```sh
python demo/multi_modality_demo.py ../../datasets/multimodal/kitti-multiview/test/velodyne/000008.bin ../../datasets/multimodal/kitti-multiview/test/right/000008.png demo/data/kitti/000008.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth --show
```


```sh
python demo/multi_modality_demo.py ../../datasets/multimodal/kitti-multiview/test/velodyne/000009.bin ../../datasets/multimodal/kitti-multiview/test/right/000009.png demo/data/kitti/000008.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth --show
```

```sh
python demo/multi_modality_demo.py data/kitti/training/velodyne_reduced/000008.bin data/kitti/training/image_2/000008.png data/kitti/kitti_infos_train.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth
```

```sh
python demo/multi_modality_demo.py data/kitti/training/velodyne_reduced/000008.bin data/kitti/training/image_2/000008.png data/kitti/kitti_infos_train.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth
```