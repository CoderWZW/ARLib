# ARLib (Pytorch)
An open-source framework for conducting data poisoning attacks on recommendation systems, designed to assist researchers and practitioners. <br>

**Members:** <br><br>
Zongwei Wang, Chongqing University, China, zongwei@cqu.edu.cn <br>
Hao Ma, Chongqing University, China, ma_hao@cqu.edu.cn <br>

**Supported by:** <br><br>
Prof. Min Gao, Chongqing University, China, gaomin@cqu.edu.cn 

<h2>Usage</h2>

1. Two configure files **attack_parser.py** and **recommend_parser** are in the directory named conf, and you can select and configure the recommendation model and attack model by modifying the configuration files. <br>
2. Run main.py.

<h2>Implemented Models</h2>

| **Recommend Model** | **Paper** | **Type** |
| --- | --- | --- |
| GMF | Yehuda et al. Matrix Factorization Techniques for Recommender Systems, IEEE Computer'09. | MF|
| WRMF | Hu et al.Collaborative Filtering for Implicit Feedback Datasets, KDD'09. | MF |
| NCF | He et al. Neural Collaborative Filtering, WWW'17. | Deep Learning |
| NGCF | Wang et al. Neural Graph Collaborative Filtering, SIGIR'19. | Graph |
| SGL | Wu et al. Self-supervised Graph Learning for Recommendation, SIGIR'21. | Graph + CL |
| SimGCL | Yu et al. Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation, SIGIR'22. | Graph + CL |

* CL is short for contrastive learning (including data augmentation); DA is short for data augmentation only

| **Attack Model** | **Paper** | **Type** |
| --- | --- | --- |
| RandomAttack | Lam et al. Shilling Recommender Systems for Fun and Profit. WWW'2004| Heuristic |
| BandwagonAttack | Gunes et al. Shilling Attacks against Recommender Systems: A Comprehensive Survey. Artif.Intell.Rev.'2014 | Heuristic |
| PGA | Li et al. Data poisoning attacks on factorization-based collaborative filtering. NIPS'2016. | Direct Gradient Optimization |
| AUSH | Lin C et al. Attacking recommender systems with augmented user profiles. CIKM'2020. | GAN |
| GOAT | Wu et al. Ready for emerging threats to recommender systems? A graph convolution-based generative shilling attack. Information Sciences'2021. | GAN |
| FedRecAttack | Rong  et al. Fedrecattack: Model poisoning attack to federated recommendation. ICDE'2022. | Direct Gradient Optimization |

<h2>Requirements</h2>

```
base==1.0.4
numba==0.53.1
numpy==1.18.0
scipy==1.4.1
torch==1.7.1.info
```




