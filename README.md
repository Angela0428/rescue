## Distributionally Robust Learning for Unsupervised Domain Adaptation

### Introduction

We propose a  distributionally robust learning (DRL) method for unsupervised domain adaptation (UDA)  that scales to modern computer-vision benchmarks.  DRL can be naturally formulated as  a competitive two-player game between a predictor and an adversary that is allowed to    corrupt the labels, subject to certain constraints, and reduces to incorporating  a density ratio between the source and target domains (under the standard log loss).  This formulation motivates the use of two neural networks that are jointly trained-- a discriminative network between the  source and target domains  for density-ratio estimation, in addition to the standard classification network. The use of a density ratio in DRL prevents the model from being overconfident on target inputs far away from the source domain. Thus,  DRL   provides conservative confidence estimation in  the target domain, even when the  target labels are not available. This conservatism motivates the use of DRL in   self-training  for sample selection, and we term the approach distributionally robust self-training (DRST). In our experiments, DRST generates more calibrated  probabilities and  achieves state-of-the-art self-training accuracy on benchmark datasets. We demonstrate that DRST captures  shape features more effectively, and reduces the extent of distributional shift during self-training.

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
