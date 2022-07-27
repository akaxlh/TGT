# Temporal Graph Transformer

This repository contains TensorFLow-based project for the paper:

> Lianghao Xia, Chao Huang, Yong Xu and Jian Pei (2022). Multi-Behavior Sequential Recommendation with Temporal Graph Transformer, <a href='https://arxiv.org/pdf/2206.02687.pdf'>Paper in arXiv</a>, <a href='https://ieeexplore.ieee.org/abstract/document/9774907'>Paper in IEEE</a>. In IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022.

## Introduction
This paper proposes Temporal Graph Transformer (TGT) for multi-behavior recommendation. TGT divides users' multi-behavior interaction sequences into sub-sequences for fine-grained sequential pattern learning. It combines sequence-level self-attention with global-level graph transformer for the multi-typed interaction data.

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{tgt2022,
  author    = {Xia, Lianghao and
               Huang, Chao and
	       Xu, Yong and
	       Pei, Jian},
  title     = {Multi-Behavior Sequential Recommendation with Temporal Graph Transformer},
  booktitle = {IEEE Transactions on Knowledge and Data Engineering (TKDE).},
  year      = {2022},
}
```
## Environment
The code of TGT is implemented and tested under the following development environment:
* python=3.7.0
* tensorflow=1.15.0
* numpy=1.12.6
* scipy=1.7.3

## Datasets
We employ two datasets to evaluate TGT: <i>Taobao</i> and <i>IJCAI Contest</i>. The two datasets are both E-Commerce-related user behavior data released by Alibaba. It contains four behavior categories: <i>click, add-to-cart, tag-as-favorite,</i> and <i>purchase</i>. We filtered out users and items with too few interactions. We adopt the leave-one-out evaluation protocol, in which the last interacted item for each test user is left out to compose the test set.

## How to Run the Codes
Please unzip the data files in `Datasets/` directory first. The commands to train and test TGT on the two datasets are as follows.
* Taobao
```
python .\labcode_taobao.py --data taobao --reg 1e-2 --mult 1 --subUsrSize 5
```
* IJCAI
```
# training
python .\labcode_ijcai.py --data ijcai --graphSampleN 10000 --subUsrSize 10 --reg 1e-4 --batch 512 --save_path ijcai
# testing
python .\labcode_ijcai.py --data ijcai --graphSampleN 10000 --subUsrSize 10 --reg 1e-4 --batch 512 --load_model ijcai --epoch 0
```
Important argumentes:
* `reg`: It is the weight for weight-decay regularization, which is tuned from the set `{1e-2, 1e-3, 1e-4, 1e-5}`.
* `graphSampleN`: This parameter adjust the size of the sampled subgraph. We use `10000` when training and `20000` when testing.
* `subUsrSize`: It is the number of interactions for each sub-user. It is used when dividing users into sub-users. It is tuned from `{5, 10, 15, 20}`.

## Achknowledgements
We thank the reviewers for their valuable feedback and
comments. This research work is supported by the research
grants from the Department of Computer Science & Musketeers Foundation Institute of Data Science at the University
of Hong Kong (HKU). The research is also supported by
National Nature Science Foundation of China (62072188),
Major Project of National Social Science Foundation of
China (18ZDA062), Science and Technology Program of
Guangdong Province (2019A050510010).
