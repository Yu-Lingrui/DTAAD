[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/Yu-Lingrui/DTAAD/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FYu-Lingrui%2FDTAAD&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=DTAAD&edge_flat=false)](https://hits.seeyoufarm.com)

# DTAAD
This repository supplements our paper "DTAAD: Dual Tcn-Attention Networks for Anomaly Detection in Multivariate Time Series Data" accepted in. This is a refactored version of the code used for results in the paper for ease of use. Follow the below steps to replicate each cell in the results table.

## Results
![Alt text](results/result.PNG?raw=true "results")

## Installation
This code needs Python-3.7 or higher.
```bash
$ git clone https://github.com/Yu-Lingrui/DTAAD  # clone
$ cd DTTAD
$ pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip3 install -r requirements.txt
```

## Dataset Preprocessing
Preprocess all datasets using the command
```bash
$ python3 preprocess.py SMAP MSL SWaT WADI SMD MSDS UCR MBA NAB
```
Distribution rights to some datasets may not be available. Check the readme files in the `./data/` folder for more details. If you want to ignore a dataset, remove it from the above command to ensure that the preprocessing does not fail.

## Result Reproduction
To run a model on a dataset, run the following command:
```bash
$ python3 main.py --model <model> --dataset <dataset> --retrain
```
where `<model>` can be either of 'DTAAD', 'GDN', 'MAD_GAN', 'MTAD_GAT', 'MSCRED', 'USAD', 'OmniAnomaly', 'LSTM_AD', and dataset can be one of 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'MSDS', 'MBA', 'UCR' and 'NAB. To train with 20% data, use the following command 
```bash
python3 main.py --model <model> --dataset <dataset> --retrain --less
```
You can use the parameters in `src/params.json` to set values in `src/constants.py` for each file.

For ablation studies, use the following models: 'DTAAD_Tcn_Local', 'DTAAD_Tcn_Global', 'DTAAD_Callback', 'DTAAD_Transformer'.

The output will provide anomaly detection and diagnosis scores and training time. For example:
```bash
$ python3 main.py --model DTAAD --dataset SMAP --retrain 
Using backend: pytorch
Creating new model: DTAAD
Training DTAAD on SMAP
Epoch 0,        L1 = 0.14307714637902505
Epoch 1,        L1 = 0.028436464400400804
Epoch 2,        L1 = 0.02235450599727518
Epoch 3,        L1 = 0.02094299886140656
Epoch 4,        L1 = 0.019440197744748654
100%|███████████████████████████████████████████████████████████████████| 5/5 [00:04<00:00,  1.57it/s]
Training time:     4.1132 s
Testing DTAAD on SMAP
{'FN': 0,
 'FP': 167,
 'Hit@100%': 1.0,
 'Hit@150%': 1.0,
 'NDCG@100%': 1.0,
 'NDCG@150%': 1.0,
 'ROC/AUC': 0.9911355291994327,
 'TN': 7590,
 'TP': 748,
 'f1': 0.8995741135930116,
 'precision': 0.8174863298635374,
 'recall': 0.9999999866310163,
 'threshold': 0.23806398726921338}
```

All outputs can be run multiple times to ensure statistical significance. 


## License

BSD-3-Clause.      
Copyright (c) 2022, Yu Lingrui     
Copyright (c) 2022, Shreshth Tuli.   
All rights reserved.

See License file for more details.
