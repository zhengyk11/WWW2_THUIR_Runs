import sys
import os
import math
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer

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

tag = int(sys.argv[1])
split_num = 8

avg_doc_len = 651.678 # 469.
num_of_docs = 2209296. # 22747126.

k1 = 1.2
k2 = 100
b = 0.75
ri = 0.0
R = 0.0

# tag = int(sys.argv[1])

qid_query = {}
query_terms = {}
file = open('ntcir_test_qid_seg_stem_v2.txt', 'r')
for line in file:
    qid, query = line.split('\t')
    qid = qid.strip()
    query = query.lower().strip().split()
    for q_term in query:
        query_terms[q_term] = 0
    qid_query[qid] = query
file.close()
print 'qid_query len:', len(qid_query)
print 'query_terms len:', len(query_terms)

term_idf = {}
file = open('count_total/content.txt')
# cnt = 0
for line in file:
    # cnt += 1
    # if cnt == 10000:
    #     break
    term, idf = line.split('\t')
    term = term.strip().lower()
    if len(term) < 1:
        continue
    if term not in query_terms:
        continue
    idf = float(idf)
    term_idf[term] = idf
file.close()
print 'term_idf len:', len(term_idf)

# qid_uid_score = {}

count_file = -1
for dirpath, dirnames, filenames in os.walk('./content_query'):
    for fn in filenames: # [tag*26500:min((tag+1)*26500, len(filenames))]:
        count_file += 1
        if count_file % split_num == tag:
            if not fn.endswith('.txt'):
                continue
            qid = fn.replace('.txt', '')
            if qid not in qid_query:
                continue
            # qid_uid_score[qid] = {}
            output = open('./ntcir_test_content_bm25/%s.txt' % qid, 'w')
            query = qid_query[qid]
            query_index = {}
            for q_term in query:
                if q_term in query_index:
                    query_index[q_term] += 1
                else:
                    query_index[q_term] = 1

            print fn
            file = open(os.path.join(dirpath, fn))
            for line in file:
                _qid, _query, _uid, _rel, _doc = line.split('\t', 4)
                _qid = _qid.strip()
                _query = _query.strip().lower()
                _doc = _doc.strip().lower()
                _uid = _uid.strip()

                doc = getStemResult(handleUrl(_doc.decode('utf-8', 'ignore').strip()))
                # print doc
                doc_index = {}
                for term in doc:
                    if term in doc_index:
                        doc_index[term] += 1
                    else:
                        doc_index[term] = 1
                doc_score = 0.
                avdl = len(doc) / avg_doc_len
                k = k1 * ((1 - b) + b * avdl)
                for q_term in query_index:
                    if q_term not in term_idf:
                        continue
                    if q_term not in doc_index:
                        continue
                    qf = query_index[q_term]
                    df = doc_index[q_term]
                    idf = math.log10(float(num_of_docs - term_idf[q_term] + 0.5) / (term_idf[q_term] + 0.5))
                    score = idf * (((k1 + 1) * df) / (k + df)) * (((k2 + 1) * qf) / (k2 + qf))
                    doc_score += score
                # qid_uid_score[qid][uid] = doc_score
                output.write(_qid + '\t'
                             + _query + '\t'
                             + _uid + '\t'
                             + _rel + '\t'
                             + str(doc_score) + '\t'
                             + _doc + '\n')
            output.close()
            file.close()


# print 'cal bm25 done'

# for qid in qid_uid_score:
#     qid_scores = qid_uid_score[qid].values()
#     qid_max_score = max(qid_scores)
#     qid_min_score = min(qid_scores)
#     for uid in qid_uid_score[qid]:
#         qid_uid_score[qid][uid] = 4.*(qid_uid_score[qid][uid] - qid_min_score+1e-3)/(qid_max_score - qid_min_score+1e-3)

# for dirpath, dirnames, filenames in os.walk('./qfiles'):
#     for fn in filenames[tag*26500:min((tag+1)*26500, len(filenames))]:
#         if not fn.endswith('.txt'):
#             continue
#         qid = fn.replace('.txt', '')
#         if qid not in qid_query:
#             continue
#         if qid not in qid_uid_score:
#             continue
#
#         query = ' '.join(qid_query[qid])
#
#         print fn
#         output = open('./cm_bm25_qfile/%s.txt'%qid, 'w')
#         file = open(os.path.join(dirpath, fn))
#         for line in file:
#             uid, doc = line.split('\t')
#             uid = uid.strip()
#             doc = doc.strip()
#             if uid not in qid_uid_score[qid]:
#                 continue
#             output.write(qid + '\t' + query + '\t' + uid + '\t' + doc + '\t' + str(qid_uid_score[qid][uid]) + '\n')

print 'all done'
