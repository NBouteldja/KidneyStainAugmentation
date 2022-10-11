# **KidneyStainAugmentation**

This repository represents a python framework to train, evaluate and apply different techniques including Image Registration, Stain Translation, and Stain Augmentation to tackle intra-stain variation, i.e. color variation within a particular stain across different laboratories and centers. In our experiments, we utilized human tissue data sampled from different cohorts including inhouse biopsies and nephrectomies (AC), the *Human BioMolecular Atlas Program* cohort (HuBMAP), the *Kidney Precision Medicine Project* cohort (KPMP), and the *Validation of the Oxford classification of IgA Nephropathy* cohort (VALIGA).

# Installation
1. Clone this repo using [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git):<br>
```
git clone https://github.com/NBouteldja/KidneyStainAugmentation
```
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and use conda to create a suitable python environment as prepared in *environment.yml* that lists all library dependencies:<br>
```
conda env create -f ./environment.yml
```
3. Activate installed python environment:
```
source activate python37
```
4. Install [pytorch](https://pytorch.org/) depending on your system:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
# Training
Train a structure segmentation network, e.g. using the following command:
```
python training.py -m custom -s train_val_test -e 500 -b 6 -r 0.001 -w 0.00001
```
Note:<br>
- Before, you need to specify the path to results folder (variable: *resultsPath*) in *training.py* and the path to your data set folder (variable: *image_dir_base*) in *dataset.py*
- *training.py* is parameterized as follows:
```
training.py --model --setting --epochs --batchSize --lrate --weightDecay 
```
