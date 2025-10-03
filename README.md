
# Efficient Autonomous Driving (EAD)

[![nuScenes](https://img.shields.io/badge/Benchmark-nuScenes-blue)](https://www.nuscenes.org/)

**Official implementation of "Efficiently Representing Vectorized Scenes: Improvements in BEV Encoding and Temporal Interaction for Autonomous Driving" ([arXiv link / preprint if available])**

*Authors: Zhaoxuan Lu, Lyuchao Liao, Liangpeng Gao, Sijing Cai, Feng Wang*  
*School of Transportation & School of Computer Science and Mathematics, Fujian University of Technology*  
*Contact: [Zhaoxuan Lu](luzhaoxuan@smail.fjut.edu.cn)*  
[Project Page / Paper / Demo] • [nuScenes Dataset](https://www.nuscenes.org/)  
[**Code**](https://github.com/RDXiaoLu/EAD.git)

---

## Abstract
Safe and efficient navigation in complex autonomous driving environments remains challenging. Existing end-to-end schemes often ignore temporal dependencies or incur high computational overhead. **Efficient Autonomous Driving (EAD)** introduces:
- **Sparse BEV Encoder:** Focuses computation on salient features in the bird’s-eye view, reducing cost without sacrificing fidelity.
- **History-Aware Graph Convolution:** Fuses temporal context for improved trajectory prediction.
- **Streamlined Feature Extraction:** Carefully balances accuracy and efficiency.

**On nuScenes, EAD achieves:**
- **30.1%** ↓ average planning displacement error
- **29.0%** ↓ collision rate
- **2.5×** faster inference  
compared to prior baselines.

_These results underscore the promise of the vectorized paradigm for safer, real-world navigation._

---



## Dataset Preparation

EAD is evaluated on [nuScenes](https://www.nuscenes.org/):
1. **Download** official nuScenes data following [instructions](https://www.nuscenes.org/download).
2. **Organize** the data directory structure as described in `configs/dataset.yaml` or in the documentation.
3. **Preprocessing:** Scripts for vectorization, BEV rasterization, and sequence generation are provided in `tools/`.  
   Adjust `config/*.yaml` to point to your dataset paths.

---

## Training

```bash
# Example: Train EAD from scratch
python tools/train.py --config EAD_configs/EAD.py

```




---




