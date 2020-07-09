## Deep Robust Classification under Domain Shift with Conservative Uncertainty Estimation

### Introduction

Popular deep learning techniques tend to generate over-confident probabilities. This hinders practitioners to apply deep learning methods to tasks that require calibrated uncertainties. Under domain shift, this problem become even more challenging. In this paper, we investigate classification and uncertainty estimation under domain shift. We aim to provide accurate predictions yet avoid over-confidence.

We propose **RESCUE**, an end-to-end calibrated classification framework. In particular, we adopt the robust classification method derived from the an adversarial estimation framework and develop a differentiable density ratio estimation method within the framework. The resulting density ratios adjust the predictive confidence according to the source distribution support and can be trained in a end-to-end fashion. Our proposed method provides a practical approach for conservative prediction under domain shift on high-dimensional data. We demonstrate that our method generates more calibrated probabilities while achieving competitive accuracy on benchmark domain shift datasets.

### Usage

#### Environment

* Python 2.7, Python 3.6: Python2.7 trains different models, Python 3.6 is for BNN training and acc_conf.py
* PyTorch 1.2.0
* Pyro 1.1.0
* CUDA version 10.0

#### File introduction

We provide the introduction to the folders and files in this section.

* Create two folders: office/ and OfficeHome/ to contain the raw images data. Data can be found and downloaded from https://github.com/jindongwang/transferlearning/tree/master/data
* Create aligned_data/ for saving the aligned data generated from Deep CORAL models. models/ is for saving trained models
* model_layers.py defines the foundations of IW and RESCUE models, and all other Python files can be directly run
* Some programs support self-defined arguments, you can change it via command or modify the code
* **If you use the code for new tasks and only need our best model, focus on train.py and model_layers.py**, note that we
get the uncertainty scores in uncertainty_scores.py
* If data is vector-based, see function softlabels(x_s, y_s, x_t, y_t, task) in train.py for reference; If image-based, see function train_end2end_alter(source, target, task) in train_e2e.py for reference. Comments are generally made in train.py

#### Dataset tree (example)
Provides what the dataset folder should look like:


├── Art

├── Clipart

├── ImageInfo.csv

├── imagelist.txt

├── Product

└── RealWorld
