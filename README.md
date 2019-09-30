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

【李祥圣】

### Experiment

#### Settings

For DMSA, 【郑玉昆】

For SDMM, 【李祥圣】


#### Results


## English subtask 【储著敏】

### Model

#### BM25

The code for calculating BM25 index is in the *learning-to-rank/bm25* folder, since BM25 calculation is one of the learning to rank processes. 

##### Dataset

We used a portion of [ClueWeb12](https://lemurproject.org/clueweb12/) corpus as our background corpus to calculate the inverse document frequency (IDF), so as to calculate the BM25 index.

##### Process

`get_html_en_multiprocess.py` finds out the particular set of htmls, and extracts their pathes and urls (with the given document ids of the htmls).

`get_html_content_7z.py` finds out the particular set of htmls, and extracts their titles and contents (with the given pathes of the htmls).

`count_term_multiprocess.py` calculates the total term frequency within all the background corpus, so as to calculate the IDF of the terms. In this code file, we also preprocess the documents to obtain better indexing. The preprocessing tricks we used include lowercasing, tokenization, removing stop words, and stemming.

`bm25_multiprocess.py` calculates the BM25 index of one particular field in a document. The values of varibles *avg_doc_len* and *num_of_docs* would be different in different field.

#### Learning to Rank

The codes for learning to rank methods are all in the *learning-to-rank* folder. We adopted three kinds of learning to rank methods: LambdaMART, AdaRank, and Coordinate Ascent, [Ranklib](https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/) package is the toolkit we used to implement them. 

##### Dataset

We chose [MQ2007 and MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0) as our training data, at the same time, we used [NTCIR-13 WWW](http://www.thuir.cn/ntcirwww/) English dataset as our validation data.  We used the contents of the  queries and the html files, as well as the relevance labels of these query-document pairs, but not the extracted features. We extracted all the features of all the datasets with our own algorithm, to ensure that the implement details are the same with the test sets.

##### Process

###### Feature Selection

In the process of calculating BM25, we already extract some of the features , including document length, TF, IDF, TF*IDF and BM25.

`get_html_url_anchor_7z.py` finds out the particular set of htmls, and extracts their anchor texts (with the given document ids of the htmls).

`generate_l2r_features.py` calculates three remaining features which have not been calulated above in each field. This code file also organizes all the eight features in one field together. After extracting all the required features, it is ready to run the learning to rank algorithms.

###### Train and Test

The training command is similar with the following example:

~~~
java -jar bin/RankLib.jar -train MQ2008/Fold1/train.txt -test MQ2008/Fold1/test.txt -validate MQ2008/Fold1/vali.txt -ranker 6 -metric2t NDCG@10 -metric2T ERR@10 -save mymodel.txt
~~~

The parameter *6* represents the LambdaMART algorithm. We can replace it with the number *3* or *4*, they represent AdaRank or Coordinate Ascent, respectively.

At the same time, the testing command can be typed like this:

~~~
java -jar RankLib.jar -load mymodel.txt -rank myResultLists.txt -score myScoreFile.txt
~~~

The more detailed information about the usage of the Ranklib package can be found at [https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/](https://sourceforge.net/p/lemur/wiki/RankLib How to use/).

### Experiment

#### Settings

We just used the default parameters of the Ranklib package to implement all these three learning to rank methods.

#### Results

The experimental results can be found in [our report](http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings14/pdf/ntcir/03-NTCIR14-WWW-ZhengY.pdf).

<<<<<<< HEAD

=======
>>>>>>> czm

## Q&A

