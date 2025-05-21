# See through the Dark: Learning Illumination-affined Representations for Nighttime Occupancy Prediction

## Get Started

###  Installation and Data Preparation

Step1、Prepare environment as that in [Install](doc/install.md).

Step2、Prepare nuScene and generate pkl file by runing：

```python
python tools/create_data_bevdet.py
```

The final directory structure for 'data' folder is like

```shell
└── data
  └── nuscenes
      ├── v1.0-trainval
      ├── maps  
      ├── sweeps  
      ├── samples
      ├── gts
      ├── bevdetv2-nuscenes_infos_train.pkl 
      └── bevdetv2-nuscenes_infos_val.pkl
```
###  Train & Test

```shell
# train:
tools/dist_train.sh ${config} ${num_gpu}

# test:
tools/dist_test.sh ${config} ${ckpt} ${num_gpu} --eval mAP
```

## Model weights

The pretrained weights in 'ckpt' folder can be found [here]("https://drive.google.com/drive/folders/1eWT82gFlY-ivyoeWza2GKD71RvwoXUDk?usp=drive_link").

All model weights can be found [here](https://drive.google.com/drive/folders/1yuPtZylYFgHQD3G7lyqCNh6v61QbJRdu?usp=sharing).
