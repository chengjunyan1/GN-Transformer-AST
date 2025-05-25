# GN-Transformer AST

This is the official repository for the paper "[GN-Transformer: Fusing AST and Source Code information in Graph Networks](https://arxiv.org/abs/2112.00663)".

## Data Preparing

### Preprocess the dataset by yourself

The code we used to preprocess the Java and Python datasets are under in ./preprocess, please read README.md in /Java and /Python respectively to see how to preprocess the corpus.

The original corpus we used are from here:

Java corpus: https://github.com/xing-hu/TL-CodeSum

Python corpus: https://github.com/EdinburghNLP/code-docstring-corpus

### Directly use our preprocessed dataset

You can directly download our preprocessed dataset:

Java: https://drive.google.com/file/d/1hVJaA2JA377Iz3bstHLIGaffUh_ogVnG/view?usp=sharing

Python: https://drive.google.com/file/d/1lQhczrERskISdBcWeS6VWLwCMpBAh-YF/view?usp=sharing

Or you can run the data_prepare.sh in ./data to prepare the dataset. 

## Training

Enter the script folders and run the gntransformer.sh, the training and testing will start.

`#GPU`: gpu device ids

`#NAME`: name of the model

### Java:

`cd ./scripts/java`

`bash gntransformer.sh #GPU #NAME`

### Python:

`cd ./scripts/python`

`bash gntransformer.sh #GPU #NAME`

#### Examples:

`bash gntransformer.sh 0 some_name # one gpu`

`bash gntransformer.sh 0,1 some_name # two gpus` 

...

## Trained models

You can download our trained models here:

Java: https://drive.google.com/file/d/1vnIuGLBNGU_AHDwL7yZIkoaByWiLKYxb/view?usp=sharing

Python: https://drive.google.com/file/d/1tk3Wc4YpSo_oLKCi6h3Kitvsux3vWFUO/view?usp=sharing

Or directly run download_models.sh in ./models to download the trained models.



### Citation
If you use this code in your research, please cite the following paper:

``` bibtex
@misc{cheng2021gntransformerfusingsequencegraph,
      title={GN-Transformer: Fusing Sequence and Graph Representation for Improved Code Summarization}, 
      author={Junyan Cheng and Iordanis Fostiropoulos and Barry Boehm},
      year={2021},
      eprint={2111.08874},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2111.08874}, 
}
```
