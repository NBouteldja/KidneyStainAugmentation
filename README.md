# **KidneyStainAugmentation**

This repository represents a python framework to train, evaluate and apply segmentation networks for renal histological analysis. In particular, we trained an [nnUnet](https://github.com/MIC-DKFZ/nnUNet) for kidney tissue segmentation followed by training another [U-net-like](https://arxiv.org/pdf/1505.04597.pdf) CNN for the segmentation of several renal structures including ![#ff0000](https://via.placeholder.com/15/ff0000/000000?text=+) tubulus, ![#00ff00](https://via.placeholder.com/15/00ff00/000000?text=+) glomerulus, ![#0000ff](https://via.placeholder.com/15/0000ff/000000?text=+) glomerular tuft, ![#00ffff](https://via.placeholder.com/15/00ffff/000000?text=+) non-tissue background (including veins, renal pelvis), ![#ff00ff](https://via.placeholder.com/15/ff00ff/000000?text=+) artery, and ![#ffff00](https://via.placeholder.com/15/ffff00/000000?text=+) arterial lumen from PAS-stained histopathology data. In our experiments, we utilized human tissue data sampled from different cohorts including inhouse biopsies (AC_B) and nephrectomies (AC_N), the *Human BioMolecular Atlas Program* cohort (HuBMAP), the *Kidney Precision Medicine Project* cohort (KPMP), and the *Validation of the Oxford classification of IgA Nephropathy* cohort (VALIGA).

# Installation
1. Clone this repo using [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git):<br>
```
git clone https://git-ce.rwth-aachen.de/labooratory-ai/flash.git/
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
python ./FLASH/training.py -m custom -s train_val_test -e 500 -b 6 -r 0.001 -w 0.00001
```
Note:<br>
- Before, you need to specify the path to results folder (variable: *resultsPath*) in *training.py* and the path to your data set folder (variable: *image_dir_base*) in *dataset.py*
- *training.py* is parameterized as follows:
```
training.py --model --setting --epochs --batchSize --lrate --weightDecay 
```
