# Awesome-Urban-Foundation-Models
An Awesome Collection of Urban Foundation Models (UFMs).

<p align="center">

![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Visits Badge](https://badges.pufler.dev/visits/usail-hkust/Awesome-Urban-Foundation-Models)
![Stars](https://img.shields.io/github/stars/usail-hkust/Awesome-Urban-Foundation-Models)

</p>




## Outline
  - [1-Language-based Models](#1-language-based-models)
  - [2-Vision-based Models](#2-vision-based-models) 
  - [3-Trajectory-based Models](#3-trajectory-based-models)
  - [4-Time Series-based Models](#4-time-series-based-models)
  - [5-Multimodal-based Models](#5-multimodal-based-models)
  - [6-Others](#6-others)
  - [7-Citation](#7-citation)
  - [8-Contributing](#8-contributing)


![A data-centric taxonomy for existing UFMs-related works based on the types of urban data modalities.](./figs/taxonomy.png)

## 1-Language-based Models

### Unimodal Approaches
#### Pre-training
**Geo-text**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

#### Adaptation
**Prompt engineering**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Model fine-tuning**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]




## 2-Vision-based Models
### Unimodal Approaches
#### Pre-training
**On-site urban visual data**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Remote sensing data**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Grid-based meteorological data**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

#### Adaptation
**Prompt engineering**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Model fine-tuning**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]




## 3-Trajectory-based Models
### Unimodal Approaches
#### Pre-training
**Road network trajectory**
- (*ICDE'18*) Deep representation learning for trajectory similarity computation [[paper](https://doi.org/10.1109/ICDE.2018.00062)]
- (*IJCNN'17*) Trajectory clustering via deep representation learning [[paper](https://doi.org/10.1109/IJCNN.2017.7966345)]
- (*TIST'20*) Trembr: Exploring road networks for trajectory representation learning [[paper](https://doi.org/10.1145/3361741)]
- (*CIKM'21*) Robust road network representation learning: When traffic patterns meet traveling semantics [[paper](https://doi.org/10.1145/3459637.3482293)]
- (*IJCAI'21*) Unsupervised path representation learning with curriculum negative sampling [[paper](https://www.ijcai.org/proceedings/2021/0452.pdf)]
- (*KDD'23*) Lightpath: Lightweight and scalable path representation learning [[paper](https://doi.org/10.1145/3580305.3599415)]
- (*ICDM'23*) Self-supervised Pre-training for Robust and Generic Spatial-Temporal Representations [[paper](https://users.wpi.edu/~yli15/Includes/23_ICDM_MingzhiCR.pdf)]
- (*TKDE'23*) Pre-Training General Trajectory Embeddings With Maximum Multi-View Entropy Coding [[paper](https://doi.org/10.1109/TKDE.2023.3347513)]
- (VLDB'22*) Unified route representation learning for multi-modal transportation recommendation with spatiotemporal pre-training [[paper](https://doi.org/10.1007/s00778-022-00748-y)]
- (*ICDE'23*) Self-supervised trajectory representation learning with temporal regularities and travel semantics [[paper](https://doi.org/10.1109/ICDE55515.2023.00070)]

**Free space trajectory**
- (*KDD'20*) Learning to simulate human mobility [[paper](https://doi.org/10.1145/3394486.3412862)]
- (*KBS'21*) Self-supervised human mobility learning for next location prediction and trajectory classification [[paper](https://doi.org/10.1016/j.knosys.2021.107214)]
- (*AAAI'21*) Pre-training context and time aware location embeddings from spatial-temporal trajectories for user next location prediction [[paper](https://doi.org/10.1609/aaai.v35i5.16548)]
- (*AAAI'23*) Contrastive pre-training with adversarial perturbations for check-in sequence representation learning [[paper](https://doi.org/10.1609/aaai.v37i4.25546)]

#### Adaptation
**Model fine-tuning**
- (*ToW'23*) Pre-Training Across Different Cities for Next POI Recommendation [[paper](https://doi.org/10.1145/3605554)]
- (*CIKM'21*) Region invariant normalizing flows for mobility transfer [[paper](https://doi.org/10.1145/3459637.3482169)]
- (*TIST'23*) Doing more with less: overcoming data scarcity for poi recommendation via cross-region transfer [[paper](https://doi.org/10.1145/3511711)]

### Cross-modal Transfer Approaches
**Prompt engineering**
- (*SIGSPATIAL'22*) Leveraging language foundation models for human mobility forecasting [[paper](https://doi.org/10.1145/3557915.3561026)]
- (*arXiv 2023.08*) Where would i go next? large language models as human mobility predictors [[paper](https://arxiv.org/abs/2308.15197)]
- (*arXiv 2023.10*) Large Language Models for Spatial Trajectory Patterns Mining [[paper](https://arxiv.org/abs/2310.04942)]
- (*arXiv 2023.11*) Exploring Large Language Models for Human Mobility Prediction under Public Events [[paper](https://arxiv.org/abs/2311.17351)]
- (*arXiv 2023.09*) Can you text what is happening? Integrating pre-trained language encoders into trajectory prediction models for autonomous driving [[paper](https://arxiv.org/abs/2309.05282)]
- (*arXiv 2023.10*) Gpt-driver: Learning to drive with gpt [[paper](https://arxiv.org/abs/2310.01415)]
- (*arXiv 2023.10*) Languagempc: Large language models as decision makers for autonomous driving [[paper](https://arxiv.org/abs/2310.03026)]




## 4-Time Series-based Models
### Unimodal Approaches
#### Pre-training
**Ordinary time series**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Spatial-correlated time series**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

#### Adaptation
**Prompt tuning**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]


### Cross-modal Transfer Approaches
**Prompt engineering**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Model fine-tuning**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Model reprogramming**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]




## 5-Multimodal-based Models
### Pre-training
**Single-domain models**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Multi-domain models**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

### Adaptation
**Prompt engineering**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

**Model fine-tuning**
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]




## 6-Others
### Unimodal Approaches
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]

### Cross-modal Transfer Approaches
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]
- (*arXiv 2023.04*) Segment Anything [[paper](https://arxiv.org/abs/2304.02643)]




## 7-Citation

If you find our paper useful, please kindly cite us via:
```

```




## 8-Contributing

👍 Contributions to this repository are welcome! 

If you have come across relevant resources, feel free to open an issue or submit a pull request.
```
- (*conference|journal*) paper_name [[pdf](link)][[code](link)]
```
