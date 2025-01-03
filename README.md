# Neurologic A*esque Decoding

This is a fork of the official repo for the paper ["NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics"](https://aclanthology.org/2022.naacl-main.57/) (NAACL 2022).

## Updated Installation Order

As of 12.15.2024, the following steps should restore the original codebase to working order for anyone trying to reproduce the results reported by the original paper. Minor fixes were needed due to using Windows and/or the passage of time.

Python 3.7
```
conda create -n astar python=3.7
conda activate astar
```

PyTorch 1.7.0
```
# Linux & Windows - NO PIP WHEEL ANYMORE
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# Mac
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0
```

Requirements
```
pip install -r requirements.txt
```

## Citation
If you use this codebase in your work, please consider citing the original paper:
```
@inproceedings{lu-etal-2022-neurologic,
    title = "{N}euro{L}ogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics",
    author = "Lu, Ximing  and
      Welleck, Sean  and
      West, Peter  and
      Jiang, Liwei  and
      Kasai, Jungo  and
      Khashabi, Daniel  and
      Le Bras, Ronan  and
      Qin, Lianhui  and
      Yu, Youngjae  and
      Zellers, Rowan  and
      Smith, Noah A.  and
      Choi, Yejin",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.57",
    doi = "10.18653/v1/2022.naacl-main.57",
    pages = "780--799",
    abstract = "The dominant paradigm for neural text generation is left-to-right decoding from autoregressive language models. Constrained or controllable generation under complex lexical constraints, however, requires foresight to plan ahead feasible future paths. Drawing inspiration from the $A^*$ search algorithm, we propose NeuroLogic A*esque, a decoding algorithm that incorporates heuristic estimates of future cost. We develop lookahead heuristics that are efficient for large-scale language models, making our method a drop-in replacement for common techniques such as beam search and top-$k$ sampling. To enable constrained generation, we build on NeuroLogic decoding (Lu et al., 2021), combining its flexibility in incorporating logical constraints with A*esque estimates of future constraint satisfaction. Our approach outperforms competitive baselines on five generation tasks, and achieves new state-of-the-art performance on table-to-text generation, constrained machine translation, and keyword-constrained generation. The improvements are particularly notable on tasks that require complex constraint satisfaction or in few-shot or zero-shot settings. NeuroLogic A*esque illustrates the power of decoding for improving and enabling new capabilities of large-scale language models.",
}

```
