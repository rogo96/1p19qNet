# 1p19qNet

This is the official repository for our paper, 

"From diagnosis to visualization of IDH-mutant glioma via 1p/19q codeletion detection using weakly-supervised deep learning"

---
# Abstract

1p/19qNET is an advanced AI network designed to enhance glioma diagnosis and treatment. It predicts alterations in 1p and 19q chromosomes and classifies IDH-mutant gliomas using whole slide images (WSIs). With a weakly-supervised learning approach, it reduces reliance on human annotation and workload. Its superior predictive power over traditional FISH methods is demonstrated through training on extensive next-gen sequencing data and independent validation. The network not only provides explainable heatmaps for clinician use but also holds potential for broad applications in diverse tumor classifications.

![Fig1](https://github.com/rogo96/1p19qNet/assets/65914374/44462e9f-8696-410e-9f29-d0ba12c25a61)
---


# Environment setup
To start, we prefer creating the environment using conda:
~~~
conda env create -f environment.yaml
conda activate 1p19qNet
~~~
Our GPU is NVIDIA RTX A5000 and we used CUDA 11.3.

---
# Data Preparation
Due to patient data privacy concerns, we do not publicly disclose Our Whole Slide Image(WSI). 

## Layout
```
1p19qNet
├── custom dataset
       │ ── 0 (WSI splitting into two directories, one for 0 and one for 1, is necessary)
            │──── custom_a.svs
            │──── ...
       │ ── 1
            │──── custom_b.svs
            │──── ...
```
---
# Running Code
1. Data Preprocessing:
   
* WSIs are divided into smaller tiles and transformed into features.
* Perform additional tasks to improve performance & model training.

2. Pretrained Model Evaluation:

   Evaluate the performance of the pretrained model that has been downloaded or trained.

3. Training and Evaluation with Custom Data:

   Train and evaluate the model using your own custom dataset with cross-validation or all data.

## Data preprocessing
~~~
cd preprocess

a) WSI to tiles 
python3 deepzoom_tiler.py -v [your wsi format] -c [wsi dir path] -d [tile dir path]
ex) python3 deepzoom_tiler.py -v svs -c ../Data/WSI -d ../Data/Tile

b) (optional) Normalize the color of the tile to resemble our tile color 
python3 stain_normalize.py -s [tile dir path] -d [normalized tile dir path]
ex) python3 stain_normalize.py -s ../Data/Tile -d ../Data/Norm_tile

c) tile to feature 
python3 compute_feats.py --tile_path [tile dir path] --feat_path [feature dir path] --num_class 2 --batch_size [batch_size] --num_workers [num_workers] --gpu_index [gpu_num] 
ex) python3 compute_feats.py --tile_path ../Data/Tile --feat_path ../Data/Feature

d) (optional) If you want to train the model on your custom dataset

Fill the empty values in the xlsx files in your Feature directory with your data, as shown in the illustration below:
~~~
![excel](https://github.com/rogo96/1p19qNet/assets/65914374/c3ff4f1c-3570-41ef-9b4a-ac7e8515300a)

## Evaluation Code 
Evaluation using a pretrained model.
* feat_dir : path of the features after preprocessing for evaluation.
* dname_1pNet, dname_19qNet : path of the pretrained models / default directory(trained on our WSIs data) 
* If you want to use a pretrained model trained on custom data, utilize the trained model from the Training Code below.
* The AUC, accuracy, precision, recall, f1 score, and confusion matrix for 1p, 19q, and the final result are available in the Performance and Result directories.
~~~
python3 test_model.py --dname_1pNet=[1p_model] --dname_19qNet=[19q_model] --feat_dir=[feature dir path] --max_r=[eMethod3 N] --gpu=[gpu_num] --boot_num=[bootstrap iterations]

run inference (no bootstrap, our WSIs model)
ex) python3 test_model.py --dname_1pNet 1pNet --dname_19qNet 19qNet --feat_dir=[feature dir path] --max_r=100 --gpu=0 

run inference (no bootstrap, custom WSIs model )
ex) python3 test_model.py --dname_1pNet [custom_1p] --dname_19qNet [custom_19q] --feat_dir=[feature dir path] --max_r=100 --gpu=0 

run inference (bootstrap N times, our WSIs model)
ex) python3 test_model.py --dname_1pNet 1pNet --dname_19qNet 19qNet --max_r=100 --gpu=0 --feat_dir=Data/Feature --boot_num=1000

run inference (bootstrap N times, custom WSIs model )
ex) python3 test_model.py --dname_1pNet [custom_1p] --dname_19qNet [custom_19q] --feat_dir=[feature dir path] --max_r=100 --gpu=0 --boot_num=1000
~~~

## Training Code with Evaluation (Custom dataset)
~~~
run train (cross validation, CV)
ex) python3 train_model.py --dname_1pNet=[custom_1p] --dname_19qNet=[custom_19q] --feat_dir=[feature dir path] --n_fold=[CV num]

run train (all data, no bootstrap evaluation)
ex) python3 train_model.py --dname_1pNet=[custom_1p] --dname_19qNet=[custom_19q] --feat_dir=[feature dir path] --all_data

run train (all data, bootstrap N times evaluation)
ex) python3 train_model.py --dname_1pNet=[custom_1p] --dname_19qNet=[custom_19q] --feat_dir=[feature dir path] --all_data --boot_num=1000
~~~
---
# Visualization (Heatmap, Key tiles)
Visualization of which tiles the model relies on to make label predictions.
* The visualization result images are located in the Result directory.
~~~
run visualization (all data)
ex) python3 visualization.py --dname_1pNet=[custom_1p] --dname_19qNet=[custom_1p] --wsi_dir=[custom wsi path] --tile_dir=[custom tile path] --feat_dir=[feature dir path]

run visualization (cross validation, CV)
ex) python3 visualization.py --n_fold=[pretrained CV num] --dname_1pNet=[custom_1p] --dname_19qNet=[custom_1p] --wsi_dir=[custom wsi path] --tile_dir=[custom tile path] --feat_dir=[feature dir path] 
~~~
![Fig4 (1) (1)](https://github.com/rogo96/1p19qNet/assets/65914374/e4caed20-4fb8-4634-9a0f-8fc76d58b681)

![Fig4 (1)](https://github.com/rogo96/1p19qNet/assets/65914374/4f80a77c-c591-4f1a-9c56-a87133bc5769)




