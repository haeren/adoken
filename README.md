# ADOKEN
Deep Learning Based Decision Support Software For MRI

## Dependency List

CUDA and cudNN need to be installed. Versions used in this project are CUDA 10.0 and cudNN 7.5.0
- Install/update NVIDIA driver
- Install Anaconda and update it in Anaconda Prompt:
```
conda update conda
conda update --all 
```
- Install CUDA Toolkit
- Download cuDNN and extract the archive to C drive like this:
```
C:\cudnn-10.0-windows10-x64-v7.5.0.56
```
- Edit environment variables: Select "Path" under the user variables and click "Edit". Add "YOUR_CUDNN_PATH\cuda\bin" after clicking "New". The path should look like this:
```
C:\cudnn-10.0-windows10-x64-v7.5.0.56\cuda\bin
```
- You can check the new PATH with the following command:
```
echo %PATH%
```
- Create a new Anaconda environment (Python 3.6) and activate it (You can edit "myenv" part, it is the environment's name):
```
conda create -n myenv python=3.6
activate myenv
```

Install Tensorflow-GPU (for Keras backend) with the following command (old version is required for deepbrain module):
```
pip install tensorflow-gpu==1.15
```

Other dependencies are:

| Module | Version |
| ------ | ------ |
| numpy | 1.18.1 |
| keras | 2.3.1 |
| matplotlib | 3.1.3 |
| nibabel | 3.0.2 |
| pandas | 1.1.4 |
| deepbrain | 0.1 |
| intensity-normalization | 1.4.3 |
| scikit-learn | 0.22.2.post1 |
| nilearn | 0.6.2 |
| split-folders | 0.4.3 |
| pyqt5 | 5.9.2 |
| arff2pandas | 1.0.1 |

## Usage

Activate the environment if it is not activated (You can change the settings, but by default the base environment will be activated each time you open a new Anaconda prompt)

Start the application with:
```
python Application.py
```

For MRI files (.nii), directory and file structure should look like this:
```
DataSet
|---AD
|       adSample1.nii
|       adSample2.nii
|
|---Normal
        normalSample1.nii
        normalSample2.nii
```

| Select Data | Pre-process | New Model | Train | Test |
| ------ | ------ | ------ | ------ | ------ |
| ![tabSelectData.jpg](https://github.com/haeren/adoken/blob/main/readme-images/tabSelectData.jpg?raw=true) | ![tabPreprocess.jpg](https://github.com/haeren/adoken/blob/main/readme-images/tabPreprocess.jpg?raw=true) |![tabCreateModel.jpg](https://github.com/haeren/adoken/blob/main/readme-images/tabCreateModel.jpg?raw=true) |![tabTrain.jpg](https://github.com/haeren/adoken/blob/main/readme-images/tabTrain.jpg?raw=true) |![tabTest.jpg](https://github.com/haeren/adoken/blob/main/readme-images/tabTest.jpg?raw=true) |

## Future Work
- To make the system compatible with data types in different formats such as csv, arff, xlsx (Data frame/feature vector)
  - Currently only the split data function under the Select Data tab is available for files with these extensions
- Integrating new deep learning model architectures into the application

## License

GNU General Public License v3.0

If you find this code useful in your research, please consider citing [our paper](https://dergipark.org.tr/en/pub/jesd/issue/62893/887327):

```
@article{article,
author = {Eren, Hakan and Okyay, Savaş and Adar, Nihat},
year = {2021},
month = {06},
pages = {406-413},
title = {ADOKEN: MR İçin Derin Öğrenme Tabanlı Karar Destek Yazılımı},
volume = {9},
journal = {Mühendislik Bilimleri ve Tasarım Dergisi},
doi = {10.21923/jesd.887327}
}
```
