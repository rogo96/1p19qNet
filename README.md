# 1p19qNet

This is the official repository for our paper, 

"From diagnosis to visualization of IDH-mutant glioma via 1p/19q codeletion detection using weakly-supervised deep learning"

---
# Abstract

1p/19qNET is an advanced AI network designed to enhance glioma diagnosis and treatment. It predicts alterations in 1p and 19q chromosomes and classifies IDH-mutant gliomas using whole slide images (WSIs). With a weakly-supervised learning approach, it reduces reliance on human annotation and workload. Its superior predictive power over traditional FISH methods is demonstrated through training on extensive next-gen sequencing data and independent validation. The network not only provides explainable heatmaps for clinician use but also holds potential for broad applications in diverse tumor classifications.

---


# Environment setup
coming soon

---
# Data Preparation
Due to patient data privacy concerns, we do not publicly disclose Our Whole Slide Image(WSI). 

Instead, we provide feature files from TCGA data for inference purposes.

## Download preprocessing TCGA features & pickle files
The files can be downloaded from the link provided, and the layout is organized as follows:

The layout is organized as follows.
```
1p19qNet
├── Data
│   ├── Feature
    │   │   │── total
                │──── tcga_1.h5
                │──── ...
    │   │   │── TCGA.xlsx
    ├── Tile_position
    │   │   │── tcga_1.pickle
    │   │   │── ...
    ├── WSI(recommend for your WSI directory)
         │   │── 0
                │──── custom_a.svs
                │──── ...
         │   │── 1
                │──── custom_b.svs
                │──── ...
```

---
# Running Code
1. Pretrained Model Evaluation:

   Evaluate the performance of the pretrained model that has been downloaded. 

2. Training and Evaluation with Custom Data:

   Train and evaluate the model using your own custom dataset with cross-validation.

## Evaluation Code
scripts for running the evaluation code only.
```
python3 test_external.py --dname_1pNet [1p_model] --dname_19qNet [19q_model] --feat_dir=[feature_dir] --max_r=[eMethod3 N] --gpu=[gpu_num] --boot_num=[bootstrap iterations]

ex) run inference (no bootstrap)
python3 test_external.py --dname_1pNet 1pNet --dname_19qNet 19qNet --max_r=100 --gpu=0 --feat_dir=Data/Feature

ex) run inference (bootstrap 1000)
python3 test_external.py --dname_1pNet 1pNet --dname_19qNet 19qNet --max_r=100 --gpu=0 --feat_dir=Data/Feature --boot_num=1000
```
---
## Custom data
1. Data preprocessing
2. train model
3. evaluation model
### Data preprocessing
~~~
cd preprocess

a) WSI to tiles 
python3 deepzoom_tiler.py -v [your wsi format] -c [wsi directory path] -d [tile directory path]
ex) python3 deepzoom_tiler.py -v svs -c ../Data/WSI -d ../Data/Tile

b) (optional) Normalize the color of the tile to resemble our tile color 
python3 stain_normalize.py -s [tile directory path] -d [normalized tile directory path]
ex) python3 stain_normalize.py -s ../Data/Tile -d ../Data/Norm_tile

c) tile to feature 
python3 compute_feats.py --tile_path [tile directory path] --num_class 2 --batch_size [batch_size] --num_workers [num_workers] --gpu_index [gpu_num] 
ex) python3 compute_feats.py --tile_path ../Data/Tile 

~~~

### Training



