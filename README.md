# Learning Regularization for Graph Inverse Problems (GRIPs)

**Conference:** AAAI-2025 (Oral)  
**Paper:** Learning Regularization for Graph Inverse Problems (https://arxiv.org/abs/2408.10436)  
**Authors:** Moshe Eliasof, Md. Shahriar Rahim Siddiqui, Carola-Bibiane Schönlieb, Eldad Haber  
**Repository:** https://github.com/nafi007/GraphInverseProblems

---

## Overview

This repository contains the source code for our AAAI-2025 paper. Comments have been incorporated into the python scripts to help with clarity. 

---

## Abstract

In recent years, Graph Neural Networks (GNNs) have been utilized for various applications ranging from drug discovery to network design and social networks. In many applications, it is impossible to observe some properties of the graph directly; instead, noisy and indirect measurements of these properties are available. These scenarios are coined as Graph Inverse Problems (GRIPs). In this work, we introduce a framework leveraging GNNs to solve GRIPs. The framework is based on a combination of likelihood and prior terms, which are used to find a solution that fits the data while adhering to learned prior information. Specifically, we propose to combine recent deep learning techniques that were developed for inverse problems, together with GNN architectures, to formulate and solve GRIPs. We study our approach on a number of representative problems that demonstrate the effectiveness of the framework.

---


### 1. Clone the repository to your local machine:

git clone https://github.com/nafi007/GraphInverseProblems.git

### 2. Move into the GraphInverseProblems folder:

cd GraphInverseProblems

### 3. Create the Python virtual environment:

conda env create -f environment.yml

### 4. Activate the virtual environment:

conda activate GRIP

## Citation
If you use our work, please cite our paper:  

>@inproceedings{eliasof2025learning,
  title={Learning Regularization for Graph Inverse Problems},
  author={Eliasof, Moshe and Siddiqui, Md Shahriar Rahim and Sch{\"o}nlieb, Carola-Bibiane and Haber, Eldad},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={16},
  pages={16471--16479},
  year={2025}
}
  

## Contact  
For any questions or inquiries, please contact: eldadhaber@gmail.com


