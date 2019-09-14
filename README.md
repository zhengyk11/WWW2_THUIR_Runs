# Introduction of WWW-2 THUIR Runs

## Chinese subtask

## Dataset

We use [Sogou-QCL](http://www.thuir.cn/sogouqcl/) dataset as our training data and evaluate our model performance on last-year's WWW test set with human labels. The Sogou-QCL dataset are sampled from query logs of Sogou.com and the relevance labels in Sogou-QCL are estimated by click models, such as TCM, DBN, PSCM, TACM and UBM. We use PSCM-based relevance labels as supervision for model training.

To obtain our data and saved models, just run ``obtain_data.sh`` in your terminal. Then, the newly created directory ``data`` contains the training, development and test datasets, all saved model files and summaries. We put the datasets in the directory ``data/runtime_data/``:

``cm_bm25_qfile_20180507_100w_merge.txt`` contains 1 million query-document pairs randomly sampled from Sogou-QCL with the segmented content of queries and documents, BM25 scores as well as TCM, DBN, PSCM, TACM and UBM-based relevance labels.

``ntcir13_test_bm25_top100_4level.txt`` contains 

``ntcir14_test_bm25_top100.txt`` contains 

``qcl_annotation_qfiles.txt`` contains 

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

We design a deep matching model with self-attention mechanism (DMSA). You can find more details in [our report](http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings14/pdf/ntcir/03-NTCIR14-WWW-ZhengY.pdf).

##### Environment 

The codes were only tested in the following environment: 

- Python == 2.7
- torch == 0.3.1
- tensorboardX == 1.1
- tensorflow-tensorboard == 0.4.0  
- tqdm == 4.23.0 


##### Run script



We try two models using different optimizers: adadelta and adam. You can quickly start to train the same models using the following commands:

```
python run.py --algo rank_0914 --train --batch_size 20 --eval_freq 100 --embed_size 200 --hidden_size 100 --train_pair_num 10000000 --vocab_dir ../data/vocab_0910_200w --train_files ../data/runtime_data/cm_bm25_qfile_20180507_100w_merge.txt --dev_files ../data/runtime_data/ntcir13_test_bm25_top100_4level.txt --max_p_len 1000 --max_q_len 20 --qfreq_file ../data/qid_query_freq.txt --dfreq_file ../data/qid_uid_freq.txt --annotation_file ../data/sogou-qcl_human_relevance_0630_no_-1.txt --result_dir ../data/result_0914 --model_dir ../data/model_0914 --summary_dir ../data/summary_0914 --num_steps 20000000 --dropout_rate 0.2 --learning_rate 0.01 --patience 5 --weight_decay 5e-5

python run.py --algo rank_0914_adam --train --batch_size 20 --eval_freq 100 --embed_size 200 --hidden_size 100 --train_pair_num 10000000 --vocab_dir ../data/vocab_0910_200w --train_files ../data/runtime_data/cm_bm25_qfile_20180507_100w_merge.txt --dev_files ../data/runtime_data/ntcir13_test_bm25_top100_4level.txt --max_p_len 1000 --max_q_len 20 --qfreq_file ../data/qid_query_freq.txt --dfreq_file ../data/qid_uid_freq.txt --annotation_file ../data/sogou-qcl_human_relevance_0630_no_-1.txt --result_dir ../data/result_0914_adam --model_dir ../data/model_0914_adam --summary_dir ../data/summary_0914_adam --num_steps 20000000 --dropout_rate 0.2 --learning_rate 0.01 --patience 5 --check_point 20000000 --optim adam
```


For predicting rank lists based on your trained model, you can just use the following commands:

```
python run.py --algo rank_0914 --predict --batch_size 20 --eval_freq 100 --embed_size 200 --hidden_size 100 --train_pair_num 10000000 --vocab_dir ../data/vocab_0910_200w --test_files ../data/runtime_data/ntcir14_test_bm25_top100.txt --max_p_len 1000 --max_q_len 20 --qfreq_file ../data/qid_query_freq.txt --dfreq_file ../data/qid_uid_freq.txt --annotation_file ../data/sogou-qcl_human_relevance_0630_no_-1.txt --result_dir ../data/result_0914 --model_dir ../data/model_0914 --summary_dir ../data/summary_0914 --num_steps 20000000 --dropout_rate 0.2 --learning_rate 0.005 --patience 5 --check_point 20000000 --load_model 11900     

python run.py --algo rank_0914_adam --predict --batch_size 20 --eval_freq 100 --embed_size 200 --hidden_size 100 --train_pair_num 10000000 --vocab_dir ../data/vocab_0910_200w --test_files ../data/runtime_data/ntcir14_test_bm25_top100.txt --max_p_len 1000 --max_q_len 20 --qfreq_file ../data/qid_query_freq.txt --dfreq_file ../data/qid_uid_freq.txt --annotation_file ../data/sogou-qcl_human_relevance_0630_no_-1.txt --result_dir ../data/result_0914_adam --model_dir ../data/model_0914_adam --summary_dir ../data/summary_0914_adam --num_steps 20000000 --dropout_rate 0.2 --learning_rate 0.005 --patience 5 --check_point 20000000 --optim adam --load_model 300
```

#### SDMM

【李祥圣】

### Experiment

#### Settings

For DMSA, 【郑玉昆】

For SDMM, 【李祥圣】


#### Results


## English subtask 【储著敏】

### Model

#### Learning to Rank

#### BM25

### Experiment

#### Settings

#### Results


## Q&A

