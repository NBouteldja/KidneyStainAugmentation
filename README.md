# **KidneyStainAugmentation**

This repository represents a python framework to train, evaluate and apply image-to-image-based translation models for computational pathology applications. More precisely, CycleGANs are trained on data from different centers to learn intra-stain variation, i.e. color variation within a particular stain across different laboratories and centers, which is finally used to apply <b>stain augmentation</b>. The CycleGAN framework builds upon this [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).<br>
In our experiments, we utilized human tissue data sampled from different cohorts including inhouse biopsies and nephrectomies (AC), the *Human BioMolecular Atlas Program* cohort (HuBMAP), the *Kidney Precision Medicine Project* cohort (KPMP), and the *Validation of the Oxford classification of IgA Nephropathy* cohort (VALIGA). We applied stain augmentation as follows: First, we trained translation models between AC and the other cohorts using the commands below. Second, we used the translators to augment the annotated training data of our previously published [segmentation model](https://git-ce.rwth-aachen.de/labooratory-ai/flash) (which was originally trained on AC). Third, we retrained the model on the cohort-augmented annotated data.<br>

# Training
To train a CycleGAN translator (e.g. incorporating a prior segmentation model), use the following command:
```
python ./KidneyStainTranslation/train.py --stain aSMA --stainB PAS --dataroot <path-to-data> --resultsPath <path-to-store-results> --netD n_layers --netG unet_7 --ngf 32 --ndf 32 --batch_size 3 --niters_init 0 --lr 0.0001 --preprocess none --niters 300000 --load_size 640 --crop_size 640 --lambda_A 1 --lambda_B 1 --lambda_id 1 --niters_linDecay 100 --saveModelEachNIteration 10000 --validation_freq 1000 --n_layers_D 4 --gpu_ids 0 --update_TB_images_freq 5000 --use_segm_model --lambda_Seg 1
```

# Testing
Use the same arguments to test the trained translator:
```
python ./KidneyStainTranslation/test.py --stain aSMA --stainB PAS --dataroot <path-to-data> --resultsPath <path-to-store-results> --netD n_layers --netG unet_7 --ngf 32 --ndf 32 --batch_size 3 --niters_init 0 --lr 0.0001 --preprocess none --niters 300000 --load_size 640 --crop_size 640 --lambda_A 1 --lambda_B 1 --lambda_id 1 --niters_linDecay 100 --saveModelEachNIteration 10000 --validation_freq 1000 --n_layers_D 4 --gpu_ids 0 --update_TB_images_freq 5000 --use_segm_model --lambda_Seg 1
```

# Contact
Nassim Bouteldja<br>
Institute of Pathology<br>
RWTH Aachen University Hospital<br>
Pauwelsstrasse 30<br>
52074 Aachen, Germany<br>
E-mail: 	nbouteldja@ukaachen.de<br>
<br>

#
    /**************************************************************************
    *                                                                         *
    *   Copyright (C) 2022 by RWTH Aachen University                          *
    *   http://www.rwth-aachen.de                                             *
    *                                                                         *
    *   License:                                                              *
    *                                                                         *
    *   This software is dual-licensed under:                                 *
    *   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)         *
    *   • AGPL (GNU Affero General Public License) open source license        *
    *                                                                         *
    ***************************************************************************/    
