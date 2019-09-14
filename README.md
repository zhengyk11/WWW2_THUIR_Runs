# Introduction of WWW-2 THUIR Runs

## Chinese subtask

## Dataset

The Sogou-QCL dataset are sampled from query logs of Sogou.com. This dataset consists of 10 parts of bz2 files that are totally about 84 GB in size when compressed. To support the research of IR and other related areas, we calculated five kinds of click model-based relevance for the query-document pairs in Sogou-QCL. Each record of a query contains the text, appearance frequency and its documents, while in each document, we provided its title, content, html source, appearance frequency and five click model-based relevance. The relevance values are estimated by click models, such as TCM, DBN, PSCM, TACM and UBM, based on a large scale of query logs during April 1st-18th, 2015. Sogou-QCL can be use support a broad range of research on information retrieval and natural language understanding, such as ad-hoc retrieval, query performance predicting, and etc.

## Evaluation

pyNTCIREVAL

nDCG, ERR, Q-measure

### Model

#### DMSA

【郑玉昆】

##### Environment 

The codes were only tested in the following environment: 

- Python == 2.7
- torch == 0.3.1
- tensorboardX == 1.1
- tensorflow-tensorboard == 0.4.0  
- tqdm == 4.23.0 


##### Run script

To obtain our data and saved models, just run ``obtain_data.sh`` in your terminal. Then, the newly created directory ``data`` contains the training, development and test datasets, all saved model files and summaries.

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

