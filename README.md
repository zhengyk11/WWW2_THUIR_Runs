# Introduction of WWW-2 THUIR Runs

## Chinese subtask

## Dataset

Sogou-QCL

## Evaluation

pyNTCIREVAL

nDCG, ERR, Q-measure

### Model

#### DMSA

【郑玉昆】

#### SDMM

Deep matching model (SDMM) contains a local matching layer and a recurrent neural network (RNN) layer. The local matching layer aims to capture the semantic matching between query and sentence. RNN captures the signals of each sequential sentence. The overall framework is 

![avatar](/Users/lixs/Learning/OneDrive/THU/Attention_step2/papers/NTCIR-report/WWW2_THUIR_Runs/sdmm.jpg)


Run the code: 
1. set the data address in ``config.py``
2. ``python baseline_main.py --prototype baseline_config`` 






### Experiment

#### Settings

For DMSA, 【郑玉昆】

For SDMM, see the ``baseline_config`` in ``config.py``.


#### Results


## English subtask 【储著敏】

### Model

#### Learning to Rank

#### BM25

### Experiment

#### Settings

#### Results


## Q&A

