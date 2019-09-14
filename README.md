# Introduction of WWW-2 THUIR Runs

## Chinese subtask

## Dataset

We use [Sogou-QCL](http://www.thuir.cn/sogouqcl/) dataset as our training data and evaluate our model performance on WWW-1 test set with human labels. The Sogou-QCL dataset are sampled from query logs of Sogou.com and the relevance labels in Sogou-QCL are estimated by click models, such as TCM, DBN, PSCM, TACM and UBM. We use PSCM-based relevance labels as supervision for model training.

To obtain the datasets we used and our saved models, just run ``obtain_data.sh`` in your terminal. Then, the newly created directory ``data`` contains the training, development and test datasets, all saved model files and summaries. 

Under the directory ``data``, there are three txt files:

- ``qid_query_freq.txt``: Each line in this file contains the query_id, query and query frequency in the Sogou-QCL dataset which are separated by ``tab``.

- ``qid_uid_freq.txt``: Each line in this file contains the query_id, doc_id and document frequency in sessions of this query in the Sogou-QCL dataset which are separated by ``tab``.

- ``sogou-qcl_human_relevance_0630_no_-1.txt``: This file contains the four-grade human labels for ``qcl_annotation_qfiles.txt`` which will be introduced later. Each line in this file contains query_id, document_id and a relevance label separated by ``tab``.


We put the training, development and test datasets in the directory ``data/runtime_data/``:

- ``cm_bm25_qfile_20180507_100w_merge.txt`` contains 1 million query-document pairs randomly sampled from Sogou-QCL with the segmented content of queries and documents, BM25 scores as well as TCM, DBN, PSCM, TACM and UBM-based relevance labels. Each line contains query_id, query content separated by ``space``, doc_id, document content separated by ``space``, a BM25 score and TCM, DBN, PSCM, TACM and UBM-based relevance labels. These items in a line are separated by ``tab``.

- ``ntcir13_test_bm25_top100_4level.txt`` contains the official test set of WWW-1 task with BM25 scores and four-grade relevance labels annotated by workers. Each line contains query_id, query content separated by ``space``, doc_id, document content separated by ``space``, a BM25 score and human relevance labels. These items in a line are separated by ``tab``.

- ``ntcir14_test_bm25_top100.txt`` contains official test set of WWW-2 task only with BM25 scores. Each line contains query_id, query content separated by ``space``, doc_id, document content separated by ``space``, a BM25 score and fake relevance labels. These items in a line are separated by ``tab``.

- ``qcl_annotation_qfiles.txt`` contains the 2,000 queries and corresponding documents in Sogou-QCL which have been labeled with four-grade relevance by workers. This file shares the same format with ``cm_bm25_qfile_20180507_100w_merge.txt``. The human labels are stored in another file named ``sogou-qcl_human_relevance_0630_no_-1.txt`` which is under the directory ``data``.

There are still five other directories:

- ``vocab_0910_200w`` contains the pre-trained word embedding file ``vocab.data`` which was calculated on the Sogou-QCL corpus using [word2vec](https://code.google.com/archive/p/word2vec/).

- ``summary_0914`` and ``summary_0914_adam`` contains the tensorboard summary files during the model training process.

- ``model_0914`` and ``model_0914_adam`` contains our saved models. We used these models to generated three of five final submitted runs, ``THUIR-C-CO-MAN-Base-1.txt``, ``THUIR-C-CO-MAN-Base-2.txt`` and ``THUIR-C-CO-MAN-Base-3.txt``. 


## Evaluation

We use ``nDCG``, ``ERR`` and ``Q-measure`` to evaluate our models with the implementation of [pyNTCIREVAL](https://github.com/mpkato/pyNTCIREVAL). Here we show an example to evaluate a predicted ranking list according to the relevance labels.

```python
# coding=utf-8
import random
from pyNTCIREVAL import Labeler
from pyNTCIREVAL.metrics import nERR, QMeasure, MSnDCG, nDCG

def data_process(y_pred, y_true):
    qrels = {}
    ranked_list = []
    c = zip(y_pred, y_true)
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[0], reverse=True)
    for i in range(len(c)):
        qrels[i] = c[i][1]
        ranked_list.append(i)
    grades = range(1, label_range+1)

    labeler = Labeler(qrels)
    labeled_ranked_list = labeler.label(ranked_list)
    rel_level_num = len(grades)
    xrelnum = labeler.compute_per_level_doc_num(rel_level_num)
    return xrelnum, grades, labeled_ranked_list

def n_dcg(y_pred, y_true, k):
    xrelnum, grades, labeled_ranked_list = data_process(y_pred, y_true)
    metric = nDCG(xrelnum, grades, cutoff=k, logb=2)
    result = metric.compute(labeled_ranked_list)
    return result


def q_measure(y_pred, y_true, k, beta=1.0):
    xrelnum, grades, labeled_ranked_list = data_process(y_pred, y_true)
    metric = QMeasure(xrelnum, grades, beta, cutoff=k)
    result = metric.compute(labeled_ranked_list)
    return result


def n_err(y_pred, y_true, k):
    xrelnum, grades, labeled_ranked_list = data_process(y_pred, y_true)
    metric = nERR(xrelnum, grades, cutoff=k)
    result = metric.compute(labeled_ranked_list)
    return result

if __name__ == '__main__':
    label_range = 4  # 4-grade relevance（0，1，2，3）
    print n_dcg([0, 1, 2, 3], [1, 2, 3, 0], k=3)  # y_pred: the predicted score, y_true: relevance label, k: cutoff
    print q_measure([0, 1, 2, 3], [1, 2, 3, 0], k=3)
    print n_err([0, 1, 2, 3], [1, 2, 3, 0], k=3)

```


### Model

#### DMSA

We design a deep matching model with self-attention mechanism (DMSA). You can find more details in [our report](http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings14/pdf/ntcir/03-NTCIR14-WWW-ZhengY.pdf). We implement DMSA by PyTorch.

##### Environment 

The codes were only tested in the following environment: 

- Python == 2.7
- torch == 0.3.1
- tensorboardX == 1.1
- tensorflow-tensorboard == 0.4.0  
- tqdm == 4.23.0 


##### Run script


We tried two models using different optimizers: adadelta and adam. You can quickly start to train the same models in the directory ``model_multi_task`` using the following commands:

```
python run.py --algo rank_0914 --train --batch_size 20 --eval_freq 100 --embed_size 200 --hidden_size 100 --train_pair_num 10000000 --vocab_dir ../data/vocab_0910_200w --train_files ../data/runtime_data/cm_bm25_qfile_20180507_100w_merge.txt --dev_files ../data/runtime_data/ntcir13_test_bm25_top100_4level.txt --max_p_len 1000 --max_q_len 20 --qfreq_file ../data/qid_query_freq.txt --dfreq_file ../data/qid_uid_freq.txt --annotation_file ../data/sogou-qcl_human_relevance_0630_no_-1.txt --result_dir ../data/result_0914 --model_dir ../data/model_0914 --summary_dir ../data/summary_0914 --num_steps 20000000 --dropout_rate 0.2 --learning_rate 0.01 --patience 5 --weight_decay 5e-5

python run.py --algo rank_0914_adam --train --batch_size 20 --eval_freq 100 --embed_size 200 --hidden_size 100 --train_pair_num 10000000 --vocab_dir ../data/vocab_0910_200w --train_files ../data/runtime_data/cm_bm25_qfile_20180507_100w_merge.txt --dev_files ../data/runtime_data/ntcir13_test_bm25_top100_4level.txt --max_p_len 1000 --max_q_len 20 --qfreq_file ../data/qid_query_freq.txt --dfreq_file ../data/qid_uid_freq.txt --annotation_file ../data/sogou-qcl_human_relevance_0630_no_-1.txt --result_dir ../data/result_0914_adam --model_dir ../data/model_0914_adam --summary_dir ../data/summary_0914_adam --num_steps 20000000 --dropout_rate 0.2 --learning_rate 0.01 --patience 5 --check_point 20000000 --optim adam
```


For predicting rank lists based on the two pre-trained models, you can just use the following commands:

```
python run.py --algo rank_0914 --predict --batch_size 20 --eval_freq 100 --embed_size 200 --hidden_size 100 --train_pair_num 10000000 --vocab_dir ../data/vocab_0910_200w --test_files ../data/runtime_data/ntcir14_test_bm25_top100.txt --max_p_len 1000 --max_q_len 20 --qfreq_file ../data/qid_query_freq.txt --dfreq_file ../data/qid_uid_freq.txt --annotation_file ../data/sogou-qcl_human_relevance_0630_no_-1.txt --result_dir ../data/result_0914 --model_dir ../data/model_0914 --summary_dir ../data/summary_0914 --num_steps 20000000 --dropout_rate 0.2 --learning_rate 0.005 --patience 5 --check_point 20000000 --load_model 11900     

python run.py --algo rank_0914_adam --predict --batch_size 20 --eval_freq 100 --embed_size 200 --hidden_size 100 --train_pair_num 10000000 --vocab_dir ../data/vocab_0910_200w --test_files ../data/runtime_data/ntcir14_test_bm25_top100.txt --max_p_len 1000 --max_q_len 20 --qfreq_file ../data/qid_query_freq.txt --dfreq_file ../data/qid_uid_freq.txt --annotation_file ../data/sogou-qcl_human_relevance_0630_no_-1.txt --result_dir ../data/result_0914_adam --model_dir ../data/model_0914_adam --summary_dir ../data/summary_0914_adam --num_steps 20000000 --dropout_rate 0.2 --learning_rate 0.005 --patience 5 --check_point 20000000 --optim adam --load_model 300
```

##### Results

We put our results in the directory ``results``:

- ``output_bm25_baseline`` contains the baseline WWW-2 run of our BM25 implementation.

- ``output_mtmodel`` contains the two WWW-2 runs by the saved DMSA models.

- ``re-rank.py``: We generate our final ranking lists by re-ranking the top documents in the baseline BM25 list according to DMSA models' scores. This script can help you do this. 

- ``output_trec``: You can get the re-ranking results here after you run the ``re-rank.py``.


#### SDMM

【李祥圣】


#### Results


## English subtask 【储著敏】

### Model

#### Learning to Rank

#### BM25

### Experiment

#### Settings

#### Results


## Q&A

