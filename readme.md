# Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the ACM MM 2025 paper **"Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models"**

[[Paper]](https://arxiv.org/pdf/2411.17471) | [[Project Page]](https://github.com/xll0328/CONCIL---ACM-MM-2025-BNI-Track-/)


## Table of Contents
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation) 
- [Training & Evaluation](#training--evaluation)
- [Experimental Results](#experimental-results)
- [Citation](#citation)

## Requirements
```bash
# Create conda environment
conda create -n concil python=3.8
conda activate concil

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
Run continual learning experiments:
```bash
bash commands/CONCIL_tc_11_14.sh
```

## Dataset Preparation
1. Download datasets:
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
   - [AWA](https://www.image-net.org/)


Generate visualizations:
```bash
jupyter nbconvert VISUAL/result_analysis.ipynb --to html
```

## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{lai2025learning,
  title={Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models},
  author={Lai, Songning et al.},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

