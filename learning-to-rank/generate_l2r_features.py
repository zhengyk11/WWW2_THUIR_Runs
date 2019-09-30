import os
import sys
import math
from collections import Counter
import numpy as np
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer

label = 0# int(sys.argv[1])

def getStemResult(text):
    words = word_tokenize(text.lower())

    stopwordset = stopwords.words('english')
    puncts = [',', '.', '!', '?', '&']
    clean_list = [token for token in words if token not in stopwordset]
    clean_list = [token for token in clean_list if token not in puncts]

    porter = PorterStemmer()
    result = [porter.stem(w) for w in clean_list]
    return result

def handleUrl(url):
    url_new = url.replace('.', ' ').replace('/', ' ').replace(':', ' ').strip()
    return url_new

def LMIR(mode, tf, idf, dl, du, Pw, c):
    lamda, mu, delta = 0.5, 50, 0.5
    N = len(dl)
    N_q_terms = len(idf)
    P_LMIR = np.zeros(N)
    if mode=='JM':
        for i in range(N):
            alpha = lamda
            P = 0.0
            for j in range(N_q_terms):
                if c[i,j]>0:
                    Ps = (1-lamda)*tf[i,j] + lamda*Pw[j]
                    P += np.log(Ps/alpha/Pw[j])
            P += (N_q_terms*np.log(alpha) + np.sum(np.log(Pw)))
            P_LMIR[i] = P
    elif mode=='DIR':
        for i in range(N):
            alpha = mu*1.0/(dl[i] + mu)   #float(np.sum(c[i,:])
            P = 0.0
            for j in range(N_q_terms):
                if c[i,j]>0:
                    Ps = (c[i,j]+mu*Pw[j]) / (dl[i] + mu)
                    P += np.log(Ps/alpha/Pw[j])
            P += (N_q_terms*np.log(alpha) + np.sum(np.log(Pw)))
            P_LMIR[i] = P
    elif mode=='ABS':
        for i in range(N):
            alpha = delta*du[i]/dl[i]
            # alpha = mu*1.0/(float(np.sum(c[i,:])) + mu)
            P = 0.0
            for j in range(N_q_terms):
                if c[i,j]>0:
                    Ps = max(c[i,j]-delta,0)/dl[i] + alpha*Pw[j]
                    P += np.log(Ps/alpha/Pw[j])
            P += (N_q_terms*np.log(alpha) + np.sum(np.log(Pw)))
            P_LMIR[i] = P
    return P_LMIR

term_idf = {}
max_idf = -1.
num_of_docs = 2209296.
cnt = 0
for line in open('data/term_idf_content.txt'):
    cnt += 1
    if cnt % 500 == 0:
        print cnt, 'read idf'
        break
    term, idf = line.strip().split('\t')
    term = term.strip()
    idf = float(idf)
    term_idf[term] = math.log10(float(num_of_docs - idf + 0.5) / (idf + 0.5))
    if term_idf[term] > max_idf:
        max_idf = term_idf[term]

print 'term_idf', len(term_idf)
print 'max_idf', max_idf

qid_uid_features = {}

queries_qid = {}

cnt = 0
path = './data/ntcir_test_content_bm25'
for dirpath, dirnames, filenames in os.walk(path):
    for fn in filenames:
        for line in open(os.path.join(path, fn)):
            cnt += 1
            if cnt % 10 == 0:
                print cnt, 'read conent'
            if cnt % 1000 == 0:
                print cnt, 'content'
                break
            attr = line.split('\t', 5)
            qid = attr[0].strip()
            if queries_qid.has_key(qid):
                query = queries_qid[qid]
            else:
                query = getStemResult(handleUrl(attr[1].strip()))
                queries_qid[qid] = query
            query_dict = Counter(query)
            uid = attr[2].strip()

            title = getStemResult(handleUrl(attr[5].decode('utf-8', 'ignore').strip()))
            if len(attr[5].strip()) < 1 or len(title) < 1:
                continue
            title_dict = Counter(title)
            title_set = set(title)
            title_idfs = []
            title_tfs = []
            query_idfs = []
            query_tfs = []
            for term in query:
                if term not in term_idf:
                    query_idfs.append(max_idf)
                else:
                    query_idfs.append(term_idf[term])
                query_tfs.append(query_dict[term])

            for term in title:
                if term not in query_dict:
                    continue
                if term not in term_idf:
                    title_idfs.append(max_idf)
                else:
                    title_idfs.append(term_idf[term])
                title_tfs.append(title_dict[term])

            bm25_score = float(attr[4])
            cm_relevances = map(float, attr[3:4])  # TCM.DBN.PSCM.TPSCM.UBM
            if qid not in qid_uid_features:
                qid_uid_features[qid] = {}
            len_title = len(title)
            len_query = len(query)
            len_title_set = len(title_set)
            qid_uid_features[qid][uid] = [cm_relevances[label],
                                          'qid:' + qid,
                                          '1:' + str(bm25_score),
                                          '2:' + str(len_title),
                                          '3:' + str(sum(title_tfs)),
                                          '4:' + str(sum(title_idfs)),
                                          '5:' + str(sum([t * i for t, i in zip(title_tfs, title_idfs)]))]

            count = np.zeros([1, len_query], dtype=np.float32)
            for i in range(1):
                for j in range(len_query):
                    if query[j] not in title_dict:
                        continue
                    count[i][j] = title_dict[query[j]]
            Pw = (np.sum(count, axis=0) + 0.1) / len_title
            jm = LMIR('JM', np.array([query_tfs], dtype=np.float32),
                      np.array([query_idfs], dtype=np.float32),
                      [len_title], [len_title_set], Pw, count)[0]
            dir = LMIR('DIR', np.array([query_tfs], dtype=np.float32),
                       np.array([query_idfs], dtype=np.float32),
                       [len_title], [len_title_set], Pw, count)[0]
            abs = LMIR('ABS', np.array([query_tfs], dtype=np.float32),
                       np.array([query_idfs], dtype=np.float32),
                       [len_title], [len_title_set], Pw, count)[0]
            qid_uid_features[qid][uid] += ['6:' + str(jm), '7:' + str(dir), '8:' + str(abs)]

            qid_uid_features[qid][uid] += ['#{} {}'.format(qid, uid)]


'''
####
cnt = 0
for line in open('data/cm_bm25_qfile_title_20180507.txt'):
    cnt += 1
    if cnt % 5000 == 0:
        print cnt, 'content'
        break
    attr = line.strip().split('\t',5)
    qid = attr[0].strip()
    query = attr[1].strip().split(' ')
    query_dict = Counter(query)
    uid = attr[2].strip()
    title = attr[5].strip().split(' ')
    if len(attr[5].strip()) < 1 or len(title) < 1:
        continue
    if qid not in qid_uid_features:
        continue
    if uid not in qid_uid_features[qid]:
        continue
    title_dict = Counter(title)
    title_set = set(title)
    title_idfs = []
    title_tfs = []
    query_idfs = []
    query_tfs = []
    for term in query:
        if term not in term_idf:
            query_idfs.append(max_idf)
        else:
            query_idfs.append(term_idf[term])
        query_tfs.append(query_dict[term])

    for term in title:
        if term not in query_dict:
            continue
        if term not in term_idf:
            title_idfs.append(max_idf)
        else:
            title_idfs.append(term_idf[term])
        title_tfs.append(title_dict[term])

    bm25_score = float(attr[4])
    cm_relevances = map(float, attr[3:4])  # TCM.DBN.PSCM.TPSCM.UBM
    if qid not in qid_uid_features:
        qid_uid_features[qid] = {}
    len_title = len(title)
    len_query = len(query)
    len_title_set = len(title_set)
    qid_uid_features[qid][uid] += ['9:'+str(bm25_score),
                                   '10:'+str(len_title),
                                   '11:'+str(sum(title_tfs)),
                                   '12:'+str(sum(title_idfs)),
                                   '13:' + str(sum([t*i for t, i in zip(title_tfs, title_idfs)]))]

    count = np.zeros([1, len_query], dtype=np.float32)
    for i in range(1):
        for j in range(len_query):
            if query[j] not in title_dict:
                continue
            count[i][j] = title_dict[query[j]]
    Pw = (np.sum(count, axis=0) + 0.1) / len_title
    jm = LMIR('JM', np.array([query_tfs], dtype=np.float32),
              np.array([query_idfs], dtype=np.float32),
              [len_title], [len_title_set], Pw, count)[0]
    dir = LMIR('DIR', np.array([query_tfs], dtype=np.float32),
               np.array([query_idfs], dtype=np.float32),
               [len_title], [len_title_set], Pw, count)[0]
    abs = LMIR('ABS', np.array([query_tfs], dtype=np.float32),
               np.array([query_idfs], dtype=np.float32),
               [len_title], [len_title_set], Pw, count)[0]
    qid_uid_features[qid][uid] += ['14:'+str(jm), '15:'+str(dir), '16:'+str(abs)]
    
    

'''

output = open('attribute_content.txt', 'w')
for qid in qid_uid_features:
    for uid in qid_uid_features[qid]:
        if len(qid_uid_features[qid][uid]) < 8:
            continue
        output.write(str(qid_uid_features[qid][uid][0]))
        for f in qid_uid_features[qid][uid][1:]:
            output.write(' ' + str(f))
        output.write('\n')
output.close()

