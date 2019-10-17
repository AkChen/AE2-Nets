# AE$^2$-Nets:Autoencoder in Autoencoder Networks for Multi-View Representation Learning

A TensorFlow implementation of *AE2-Nets:Autoencoder in Autoencoder Networks for Multi-View Representation Learning*

## Requirements

- TensorFlow 1.12.0
- Python 3.6.7
- sklearn
- numpy
- scipy
- h5py

## Introduction

Learning on data represented with multiple views (e.g.,multiple types of descriptors or modalities) is a rapidlygrowing direction in machine learning and computer vi-sion. Although effectiveness achieved, most existing algo-rithms usually focus on classification or clustering tasks.Differently, in this paper, we focus on unsupervised repre-sentation learning and propose a novel framework termedAutoencoder in Autoencoder Networks (AE2-Nets), whichintegrates information from heterogeneous sources into anintact representation by the nested autoencoder framework.The proposed method has the following merits: (1) ourmodel jointly performs view-specific representation learn-ing (with the inner autoencoder networks) and multi-viewinformation encoding (with the outer autoencoder network-s) in a unified framework; (2) due to the degradation pro-cess from the latent representation to each single view, ourmodel flexibly balances the complementarity and consis-tence among multiple views. The proposed model is effi-ciently solved by the alternating direction method (ADM),and demonstrates the effectiveness compared with state-of-the-art algorithms.

## Example Experiments

This repository contains a subset of the experiments mentioned in the [paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_AE2-Nets_Autoencoder_in_Autoencoder_Networks_CVPR_2019_paper.html).

## Testing

```
python test_hand.py
```

## Citation
If you find AE2-Nets helps your research, please cite our paper:
```
@InProceedings{Zhang_2019_CVPR,
author = {Zhang, Changqing and Liu, Yeqing and Fu, Huazhu},
title = {AE2-Nets: Autoencoder in Autoencoder Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Questions

For any additional questions, feel free to email zongbo@tju.edu.cn.
