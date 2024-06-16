# LayerDAG

[[Paper]](https://openreview.net/pdf?id=IsarrieeQA)

## Table of Contents

- [Installation](#installation)

## Installation

```bash
conda create -n LayerDAG python=3.10 -y
conda activate LayerDAG
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c conda-forge cudatoolkit=11.6
conda clean --all
pip install dgl==1.1.0+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
pip install tqdm einops wandb pydantic pandas
```
