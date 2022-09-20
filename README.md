# LaST: Learning Latent Seasonal-Trend Representations for Time Series Forecasting

![](https://img.shields.io/badge/python-3.8.12-green)![](https://img.shields.io/badge/pytorch-1.8.1-green)![](https://img.shields.io/badge/cudatoolkit-10.2-green)![](https://img.shields.io/badge/cudnn-7.6.5-green)

In this repository, we provide source code of LaST framework for reproductivity.

## Dataset

We conducted extensive experiments on seven real-world benchmark datasets from four covering the categories of mainstream time series forecasting applications.  

Please download from the following buttons and place them into `datasets` folder.

[![](https://img.shields.io/badge/Download-Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/13Ae_qDDxTQDroHCKUIG4xp3Sfi6yuhjX?usp=sharing)



## Usage

#### Requirements

The code was tested with `python 3.8`, `pytorch 1.8.1`, `cudatookkit 10.2`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name LaST python=3.8

# activate environment
conda activate LaST

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts

# install other requirements
conda install numpy pandas
```



#### Run code

To train and evaluate LaST framework on a dataset, run the following command:

```shell
python run.py --data <dataset_name>  --features <forecasting_mode>  --seq_len <input_length>  --pred_len <pred_length>  --latent_size <latent_size>  --batch_size <batch_size>  --patience <patience>  --seed <random_seed>
```

The detailed descriptions about the arguments are as following:

| Parameter name   | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| dataset_name     | The dataset name can be selected from ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange_rate", "Electricity", "Weather"] |
| forecasting_mode | A value in ["S", "M"]. "S" denotes univariate forecasting while "M" denotes multivariate forecasting. |
| input_length     | The input (historical) sequence length, default is 201.      |
| pred_length      | The output (forecasting) sequence length.                    |
| latent_size      | The dimension of latent representations, default is 128.     |
| batch_size       | Batch size, default is 32.                                   |
| patience         | The steps of early stop strategy in training.                |
| random_seed      | The random seed.                                             |



## Directory Structure

The code directory structure is shown as follows:
```shell
LaST
├── datasets  # seven datasets files
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   ├── ETTm2.csv
│   ├── electricity.csv
│   ├── exchange_rate.csv
│   └── weather.csv
├── expriments  # training, validation, and test code of LaST
│   ├── exp_basic.py
│   └── exp_LaST.py
├── models  # code of LaST and its dependencies
│   ├── LaST.py  # LaST main code
│   └── utils.py  # modules for LaST including autocorrelation, cort, etc.
├── utlis
│   ├── data_loader.py  # data loading and preprocessing code
│   ├── metrics.py  # metrics for evaluation
│   ├── timefeatures.py  # extract time-related features
│   └── tools.py  # tools for training, such as early stopping and learning rate controls 
├── LICENSE  # code license
├── run.py  # entry for model training, validation, and test 
└── README.md  # This file
```

