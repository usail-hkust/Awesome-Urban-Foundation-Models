# Awesome-Urban-Foundation-Models

<p align="center">

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Visits Badge](https://badges.pufler.dev/visits/usail-hkust/Awesome-Urban-Foundation-Models)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
![Stars](https://img.shields.io/github/stars/usail-hkust/Awesome-Urban-Foundation-Models)

</p>

An Awesome Collection of Urban Foundation Models (UFMs).

**Urban Foundation Models (UFMs)** are large-scale models pre-trained on vast **multi-source, multi-granularity, and multimodal urban data**.  It benefits significantly from its pre-training phase, exhibiting emergent capabilities and remarkable adaptability to a broader range of multiple downstream tasks and domains in urban contexts.

![UFM](./figs/UFM_fig.png)

## Survey Paper

[**Towards Urban General Intelligence: A Review and Outlook of Urban Foundation Models**](https://drive.google.com/file/d/1NrODovTVoAT1u_u-yVSzrigmGUvOdYod/view?usp=sharing)  

**Authors**: [Weijia Zhang](https://scholar.google.com/citations?user=lSi3CIoAAAAJ&hl=en), [Jindong Han](https://scholar.google.com/citations?user=e9lFam0AAAAJ&hl=en), [Zhao Xu](https://xzbill.top/zhaoxu/), [Hang Ni](https://scholar.google.com/citations?user=2jk7gKYAAAAJ&hl=en), [Hao Liu](https://raymondhliu.github.io/), [Hui Xiong](https://facultyprofiles.hkust-gz.edu.cn/faculty-personal-page?id=253)

🌟 If you find this resource helpful, please consider starring this repository and citing our survey paper:

```

```




## Outline
- [Awesome-Urban-Foundation-Models](#awesome-urban-foundation-models)
  - [Survey Paper](#survey-paper)
  - [Outline](#outline)
    - [Taxonomy](#taxonomy)
  - [1. Language-based Models](#1-language-based-models)
    - [Unimodal Approaches](#unimodal-approaches)
      - [Pre-training](#pre-training)
      - [Adaptation](#adaptation)
  - [2. Vision-based Models](#2-vision-based-models)
    - [Unimodal Approaches](#unimodal-approaches-1)
      - [Pre-training](#pre-training-1)
      - [Adaptation](#adaptation-1)
  - [3. Trajectory-based Models](#3-trajectory-based-models)
    - [Unimodal Approaches](#unimodal-approaches-2)
      - [Pre-training](#pre-training-2)
      - [Adaptation](#adaptation-2)
    - [Cross-modal Transfer Approaches](#cross-modal-transfer-approaches)
  - [4. Time Series-based Models](#4-time-series-based-models)
    - [Unimodal Approaches](#unimodal-approaches-3)
      - [Pre-training](#pre-training-3)
      - [Adaptation](#adaptation-3)
    - [Cross-modal Transfer Approaches](#cross-modal-transfer-approaches-1)
  - [5. Multimodal-based Models](#5-multimodal-based-models)
    - [Pre-training](#pre-training-4)
    - [Adaptation](#adaptation-4)
  - [6. Others](#6-others)
    - [Unimodal Approaches](#unimodal-approaches-4)
    - [Cross-modal Transfer Approaches](#cross-modal-transfer-approaches-2)
  - [7. Contributing](#7-contributing)

### Taxonomy

![A data-centric taxonomy for existing UFMs-related works based on the types of urban data modalities.](./figs/taxonomy.png)




## 1. Language-based Models

### Unimodal Approaches
#### Pre-training
**Geo-text**
- (*KDD'22*) ERNIE-GeoL: A Geography-and-Language Pre-trained Model and its Applications in Baidu Maps [[paper](https://dl.acm.org/doi/pdf/10.1145/3534678.3539021)]
- (*SIGIR'22*) MGeo: Multi-Modal Geographic Language Model Pre-Training [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591728)]

#### Adaptation
**Prompt engineering**
- (*arXiv 2023.10*) Can Large Language Models be Good Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning [[paper](https://arxiv.org/abs/2310.03249)]
- (*arXiv 2023.10*) GeoLLM: Extracting Geospatial Knowledge from Large Language Models [[paper](https://arxiv.org/abs/2310.06213)]
- (*arXiv 2023.05*) Towards Human-AI Collaborative Urban Science Research Enabled by Pre-trained Large Language Models [[paper](https://arxiv.org/abs/2305.11418)]
- (*arXiv 2023.05*) GPT4GEO: How a Language Model Sees the World's Geography [[paper](https://arxiv.org/abs/2306.00020)]
- (*arXiv 2023.05*) On the Opportunities and Challenges of Foundation Models for Geospatial Artificial Intelligence [[paper](https://arxiv.org/abs/2304.06798)]
- (*arXiv 2023.05*) ChatGPT is on the Horizon: Could a Large Language Model be Suitable for Intelligent Traffic Safety Research and Applications? [[paper](https://arxiv.org/abs/2303.05382)]
- (*GIScience'23*) Evaluating the Effectiveness of Large Language Models in Representing Textual Descriptions of Geometry and Spatial Relations [[paper](https://arxiv.org/abs/2307.03678)]
- (*SIGSPATIAL'23*) Are Large Language Models Geospatially Knowledgeable? [[paper](https://dl.acm.org/doi/abs/10.1145/3589132.3625625)]
- (*SIGSPATIAL'23*) Towards Understanding the Geospatial Skills of ChatGPT: Taking a Geographic Information Systems (GIS) Exam [[paper](https://dl.acm.org/doi/abs/10.1145/3615886.3627745)]
- (*SIGSPATIAL'22*) Towards a Foundation Model for Geospatial Artificial Intelligence (Vision Paper) [[paper](https://arxiv.org/abs/2306.00020)]

**Model fine-tuning**
- (*arXiv 2023.11*) Optimizing and Fine-tuning Large Language Model for Urban Renewal [[paper](https://arxiv.org/abs/2311.15490)]
- (*arXiv 2023.09*) K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization [[paper](https://arxiv.org/abs/2306.05064)]
- (*EMNLP'23*) GeoLM: Empowering Language Models for Geospatially Grounded Language Understanding [[paper](https://aclanthology.org/2023.emnlp-main.317/)]
- (*KDD'23*) QUERT: Continual Pre-training of Language Model for Query Understanding in Travel Domain Search [[paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599891)]
- (*TOIS'23*) Improving First-stage Retrieval of Point-of-interest Search by Pre-training Models [[paper](https://dl.acm.org/doi/full/10.1145/3631937)]
- (*EMNLP'22*) SpaBERT: A Pretrained Language Model from Geographic Data for Geo-Entity Representation [[paper](https://aclanthology.org/2022.findings-emnlp.200/)]


## 2. Vision-based Models
### Unimodal Approaches
#### Pre-training
**On-site urban visual data**
- (*WWW'23*) Knowledge-infused Contrastive Learning for Urban Imagery-based Socioeconomic Prediction [[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583876)]
- (*CIKM'22*) Predicting Multi-level Socioeconomic Indicators from Structural Urban Imagery [[paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557153)]
- (*AAAI'20*) Urban2Vec: Incorporating Street View Imagery and POIs for Multi-Modal Urban Neighborhood Embedding [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/5450)]

**Remote sensing data**
- (*arXiv 2023.04*) A Billion-scale Foundation Model for Remote Sensing Images [[paper](https://arxiv.org/abs/2304.05215)]
- (*TGRS'23*) RingMo-Sense: Remote Sensing Foundation Model for Spatiotemporal Prediction via Spatiotemporal Evolution Disentangling [[paper](https://ieeexplore.ieee.org/abstract/document/10254320)]
- (*ICCV'23*) Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning [[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.html)]
- (*ICML'23*) CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations [[paper](https://dl.acm.org/doi/10.5555/3618408.3619389)]
- (*TGRS'22*) Advancing Plain Vision Transformer Toward Remote Sensing Foundation Model [[paper](https://ieeexplore.ieee.org/abstract/document/9956816)]
- (*TGRS'22*) RingMo: A Remote Sensing Foundation Model With Masked Image Modeling [[paper](https://ieeexplore.ieee.org/abstract/document/9844015)]

**Grid-based meteorological data**
- (*arXiv 2023.04*) FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead [[paper](https://arxiv.org/abs/2304.02948)]
- (*arXiv 2023.04*) W-MAE: Pre-trained Weather Model with Masked Autoencoder for Multi-variable Weather Forecasting [[paper](https://arxiv.org/abs/2304.08754)]
- (*arXiv 2022.02*) FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators [[paper](https://arxiv.org/abs/2202.11214)]
- (*Nature'23*) Accurate Medium-range Global Weather Forecasting with 3D Neural Networks [[paper](https://www.nature.com/articles/s41586-023-06185-3)]
- (*ICML'23*) ClimaX: A Foundation Model for Weather and Climate [[paper](https://icml.cc/virtual/2023/28654)]

#### Adaptation
**Prompt engineering**
- (*TGRS'24*) RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model [[paper](https://ieeexplore.ieee.org/abstract/document/10409216)]
- (*NeurIPS'23*) SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [[paper](https://arxiv.org/abs/2305.02034)]

**Model fine-tuning**
- (*arXiv 2023.11*) GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure [[paper](https://arxiv.org/abs/2311.11319)]
- (*arXiv 2023.02*) Learning Generalized Zero-Shot Learners for Open-Domain Image Geolocalization [[paper](https://arxiv.org/abs/2302.00275)]
- (*TGRS'23*) RingMo-SAM: A Foundation Model for Segment Anything in Multimodal Remote-Sensing Images [[paper](https://ieeexplore.ieee.org/abstract/document/10315957)]
- (*IJAEOG'22*) Migratable Urban Street Scene Sensing Method based on Vsion Language Pre-trained Model [[paper](https://www.sciencedirect.com/science/article/pii/S1569843222001807)]

## 3. Trajectory-based Models
### Unimodal Approaches
#### Pre-training
**Road network trajectory**
- (*KDD'23*) Lightpath: Lightweight and scalable path representation learning [[paper](https://doi.org/10.1145/3580305.3599415)]
- (*ICDM'23*) Self-supervised Pre-training for Robust and Generic Spatial-Temporal Representations [[paper](https://users.wpi.edu/~yli15/Includes/23_ICDM_MingzhiCR.pdf)]
- (*TKDE'23*) Pre-Training General Trajectory Embeddings With Maximum Multi-View Entropy Coding [[paper](https://doi.org/10.1109/TKDE.2023.3347513)]
- (*ICDE'23*) Self-supervised trajectory representation learning with temporal regularities and travel semantics [[paper](https://doi.org/10.1109/ICDE55515.2023.00070)]
- (*VLDBJ'22*) Unified route representation learning for multi-modal transportation recommendation with spatiotemporal pre-training [[paper](https://doi.org/10.1007/s00778-022-00748-y)]
- (*CIKM'21*) Robust road network representation learning: When traffic patterns meet traveling semantics [[paper](https://doi.org/10.1145/3459637.3482293)]
- (*IJCAI'21*) Unsupervised path representation learning with curriculum negative sampling [[paper](https://www.ijcai.org/proceedings/2021/0452.pdf)]
- (*TIST'20*) Trembr: Exploring road networks for trajectory representation learning [[paper](https://doi.org/10.1145/3361741)]
- (*ICDE'18*) Deep representation learning for trajectory similarity computation [[paper](https://doi.org/10.1109/ICDE.2018.00062)]
- (*IJCNN'17*) Trajectory clustering via deep representation learning [[paper](https://doi.org/10.1109/IJCNN.2017.7966345)]

**Free space trajectory**
- (*AAAI'23*) Contrastive pre-training with adversarial perturbations for check-in sequence representation learning [[paper](https://doi.org/10.1609/aaai.v37i4.25546)]
- (*KBS'21*) Self-supervised human mobility learning for next location prediction and trajectory classification [[paper](https://doi.org/10.1016/j.knosys.2021.107214)]
- (*AAAI'21*) Pre-training context and time aware location embeddings from spatial-temporal trajectories for user next location prediction [[paper](https://doi.org/10.1609/aaai.v35i5.16548)]
- (*KDD'20*) Learning to simulate human mobility [[paper](https://doi.org/10.1145/3394486.3412862)]

#### Adaptation
**Model fine-tuning**
- (*ToW'23*) Pre-Training Across Different Cities for Next POI Recommendation [[paper](https://doi.org/10.1145/3605554)]
- (*TIST'23*) Doing more with less: overcoming data scarcity for poi recommendation via cross-region transfer [[paper](https://doi.org/10.1145/3511711)]
- (*CIKM'21*) Region invariant normalizing flows for mobility transfer [[paper](https://doi.org/10.1145/3459637.3482169)]

### Cross-modal Transfer Approaches
**Prompt engineering**
- (*arXiv 2023.11*) Exploring Large Language Models for Human Mobility Prediction under Public Events [[paper](https://arxiv.org/abs/2311.17351)]
- (*arXiv 2023.10*) Large Language Models for Spatial Trajectory Patterns Mining [[paper](https://arxiv.org/abs/2310.04942)]
- (*arXiv 2023.10*) Gpt-driver: Learning to drive with gpt [[paper](https://arxiv.org/abs/2310.01415)]
- (*arXiv 2023.10*) Languagempc: Large language models as decision makers for autonomous driving [[paper](https://arxiv.org/abs/2310.03026)]
- (*arXiv 2023.09*) Can you text what is happening? Integrating pre-trained language encoders into trajectory prediction models for autonomous driving [[paper](https://arxiv.org/abs/2309.05282)]
- (*arXiv 2023.08*) Where would i go next? large language models as human mobility predictors [[paper](https://arxiv.org/abs/2308.15197)]
- (*SIGSPATIAL'22*) Leveraging language foundation models for human mobility forecasting [[paper](https://doi.org/10.1145/3557915.3561026)]

## 4. Time Series-based Models
### Unimodal Approaches
#### Pre-training
**Ordinary time series**
- (*arXiv 2023.12*) Prompt-based Domain Discrimination for Multi-source Time Series Domain Adaptation [[paper](https://arxiv.org/pdf/2312.12276)]
- (*arXiv 2023.11*) PT-Tuning: Bridging the Gap between Time Series Masked Reconstruction and Forecasting via Prompt Token Tuning [[paper](https://arxiv.org/pdf/2311.03768)]
- (*arXiv 2023.10*) UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting [[paper](https://arxiv.org/pdf/2310.09751)]
- (*arXiv 2023.03*) SimTS: Rethinking Contrastive Representation Learning for Time Series Forecasting [[paper](https://arxiv.org/pdf/2303.18205)]
- (*arXiv 2023.01*) Ti-MAE: Self-Supervised Masked Time Series Autoencoders [[paper](https://arxiv.org/pdf/2301.08871)]
- (*NeurIPS'23*) SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling [[paper](https://arxiv.org/pdf/2302.00861)]
- (*NeurIPS'23*) Lag-llama: Towards foundation models for time series forecasting [[paper](https://arxiv.org/pdf/2310.08278)]
- (*ICLR'23*) A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[paper](https://arxiv.org/pdf/2211.14730)]
- (*KDD'23*) TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting [[paper](https://arxiv.org/pdf/2306.09364)]
- (*AAAI'22*) TS2Vec: Towards Universal Representation of Time Series [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20881/20640)]
- (*ICLR'22*) CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting [[paper](https://arxiv.org/pdf/2202.01575)]
- (*TNNLS'22*) Self-Supervised Autoregressive Domain Adaptation for Time Series Data [[paper](https://ieeexplore.ieee.org/iel7/5962385/6104215/09804766.pdf)]
- (*IJCAI'21*) Time-Series Representation Learning via Temporal and Contextual Contrasting [[paper](https://arxiv.org/pdf/2106.14112)]
- (*ICLR'21*) Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding [[paper](https://arxiv.org/pdf/2106.00750)]
- (*AAAI'21*) Meta-Learning Framework with Applications to Zero-Shot Time-Series Forecasting [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17115/16922)]
- (*AAAI'21*) Time Series Domain Adaptation via Sparse Associative Structure Alignment [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16846/16653)]
- (*KDD'21*) A Transformer-based Framework for Multivariate Time Series Representation Learning [[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467401)]
- (*KDD'20*) Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data [[paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403228)]
- (*NeurIPS'19*) Unsupervised Scalable Representation Learning for Multivariate Time Series [[paper](https://proceedings.neurips.cc/paper/2019/file/53c6de78244e9f528eb3e1cda69699bb-Paper.pdf)]

**Spatial-correlated time series**
- (*NeurIPS'23*) GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks [[paper](https://arxiv.org/pdf/2311.04245)]
- (*CIKM'23*) Mask- and Contrast-Enhanced Spatio-Temporal Learning for Urban Flow Prediction [[paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3614958)]
- (*CIKM'23*) Cross-city Few-Shot Traffic Forecasting via Traffic Pattern Bank [[paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3614829)]
- (*KDD'23*) Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities [[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599529)]
- (*KDD'22*) Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting [[paper](https://dl.acm.org/doi/pdf/10.1145/3534678.3539250)]
- (*WSDM'22*) ST-GSP: Spatial-Temporal Global Semantic Representation Learning for Urban Flow Prediction [[paper](https://dl.acm.org/doi/pdf/10.1145/3488560.3498444)]
- (*SIGSPATIAL'22*) When Do Contrastive Learning Signals Help Spatio-Temporal Graph Forecasting? [[paper](https://dl.acm.org/doi/pdf/10.1145/3557915.3560939)]
- (*KDD'22*) Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting [[paper](https://dl.acm.org/doi/pdf/10.1145/3534678.3539396)]
- (*WWW'19*) Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction [[paper](https://dl.acm.org/doi/pdf/10.1145/3308558.3313577)]
- (*IJCAI'18*) Cross-City Transfer Learning for Deep Spatio-Temporal Prediction [[paper](https://arxiv.org/pdf/1802.00386)]

#### Adaptation
**Prompt tuning**
- (*arXiv 2023.12*) Prompt-based Domain Discrimination for Multi-source Time Series Domain Adaptation [[paper](https://arxiv.org/pdf/2312.12276)]
- (*arXiv 2023.11*) PT-Tuning: Bridging the Gap between Time Series Masked Reconstruction and Forecasting via Prompt Token Tuning [[paper](https://arxiv.org/pdf/2311.03768)]
- (*arXiv 2023.05*) Spatial-temporal Prompt Learning for Federated Weather Forecasting [[paper](https://arxiv.org/pdf/2305.14244)]
- (*CIKM'23*) PromptST: Prompt-Enhanced Spatio-Temporal Multi-Attribute Prediction [[paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3615016)]
- (*IJCAI'23*) Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data [[paper](https://arxiv.org/pdf/2301.09152)]


### Cross-modal Transfer Approaches
**Prompt engineering**
- (*TKDE'22*) PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting [[paper](https://ieeexplore.ieee.org/iel7/69/4358933/10356715.pdf)]
- (*NeurIPS'23*) Large Language Models Are Zero-Shot Time Series Forecasters [[paper](https://arxiv.org/pdf/2310.07820.pdf?trk=public_post_comment-text)]

**Model fine-tuning**
- (*ICLR'24*) TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting [[paper](https://arxiv.org/pdf/2310.04948)]
- (*arXiv 2023.11*) One Fits All: Universal Time Series Analysis by Pretrained LM and Specially Designed Adaptors [[paper](https://arxiv.org/pdf/2311.14782)]
- (*arXiv 2023.11*) GATGPT: A Pre-trained Large Language Model with Graph Attention Network for Spatiotemporal Imputation [[paper](https://arxiv.org/pdf/2311.14332)]
- (*arXiv 2023.08*) LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs [[paper](https://arxiv.org/pdf/2308.08469)]
- (*NeurIPS'23*) One Fits All: Power General Time Series Analysis by Pretrained LM [[paper](https://www.researchgate.net/profile/Tian-Zhou-18/publication/368753168_One_Fits_AllPower_General_Time_Series_Analysis_by_Pretrained_LM/links/647d80e3b3dfd73b77662460/One-Fits-AllPower-General-Time-Series-Analysis-by-Pretrained-LM.pdf)]

**Model reprogramming**
- (*ICLR'24*) Time-LLM: Time Series Forecasting by Reprogramming Large Language Models [[paper](https://arxiv.org/pdf/2310.01728.pdf?trk=public_post_comment-text)]
- (*arXiv 2023.08*) TEST: Text Prototype Aligned Embedding to Activate LLM’s Ability for Time Series [[paper](https://arxiv.org/pdf/2308.08241)]

## 5. Multimodal-based Models
### Pre-training
**Single-domain models**
- (*WWW'24*) When Urban Region Profiling Meets Large Language Models [[paper](https://arxiv.org/abs/2310.18340)]
- (*TITS'23*) Parallel Transportation in TransVerse: From Foundation Models to DeCAST [[paper](https://ieeexplore.ieee.org/document/10260246)]

**Multi-domain models**
- (*arXiv 2023.12*) AllSpark: A Multimodal Spatiotemporal General Model [[paper](https://arxiv.org/abs/2401.00546)]
- (*arXiv 2023.10*) City Foundation Models for Learning General Purpose Representations from OpenStreetMap [[paper](https://arxiv.org/abs/2310.00583)]

### Adaptation
**Prompt engineering**
- (*arXiv 2023.09*) TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models [[paper](https://arxiv.org/abs/2309.06719)]
- (*arXiv 2023.07*) GeoGPT: Understanding and Processing Geospatial Tasks through An Autonomous GPT [[paper](https://arxiv.org/abs/2307.07930)]

**Model fine-tuning**
- (*arXiv 2023.12*) Urban Generative Intelligence (UGI): A Foundational Platform for Agents in Embodied City Environment [[paper](https://arxiv.org/abs/2312.11813)]
- (*arXiv 2023.07*) VELMA: Verbalization Embodiment of LLM Agents for Vision and Language Navigation in Street View [[paper](https://arxiv.org/abs/2307.06082)]


## 6. Others
### Unimodal Approaches
- (*EDBT'23*) Spatial Structure-Aware Road Network Embedding via Graph Contrastive Learning [[paper](https://people.eng.unimelb.edu.au/jianzhongq/papers/EDBT2023_RoadNetworkEmbedding.pdf)]
- (*CIKM'21*) GeoVectors: A Linked Open Corpus of OpenStreetMap Embeddings on World Scale [[paper](https://dl.acm.org/doi/pdf/10.1145/3459637.3482004)]

### Cross-modal Transfer Approaches
- (*arXiv 2023.12*) Large Language Models as Traffic Signal Control Agents: Capacity and Opportunity [[paper](https://arxiv.org/pdf/2312.16044)]
- (*arXiv 2023.08*) Llm powered sim-to-real transfer for traffic signal control [[paper](https://arxiv.org/pdf/2308.14284)]
- (*arXiv 2023.06*) Can ChatGPT Enable ITS? The Case of Mixed Traffic Control via Reinforcement Learning [[paper](https://arxiv.org/pdf/2306.08094)]




## 7. Contributing

👍 Contributions to this repository are welcome! 

If you have come across relevant resources, feel free to open an issue or submit a pull request.
```
- (*conference|journal*) paper_name [[pdf](link)][[code](link)]
```
