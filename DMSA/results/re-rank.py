import math

from pyNTCIREVAL import Labeler
from pyNTCIREVAL.metrics import nERR, QMeasure, MSnDCG, nDCG
import random
import os


def data_process(y_pred, y_true):
    qrels = {}
    ranked_list = []
    c = zip(y_pred, y_true)
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[0], reverse=True)
    for i in range(len(c)):
        qrels[i] = c[i][1]
        ranked_list.append(i)
    grades = range(1, label_range + 1)

    labeler = Labeler(qrels)
    labeled_ranked_list = labeler.label(ranked_list)
    rel_level_num = len(grades)
    xrelnum = labeler.compute_per_level_doc_num(rel_level_num)
    return xrelnum, grades, labeled_ranked_list


# def n_dcg(y_pred, y_true, k):
#     xrelnum, grades, labeled_ranked_list = data_process(y_pred, y_true)
#     metric = MSnDCG(xrelnum, grades, cutoff=k)
#     result = metric.compute(labeled_ranked_list)
#     return result

def n_dcg(y_true, y_pred, rel_threshold=0., k=10):
    if k <= 0.:
        return 0.
    s = 0.
    # y_true = np.squeeze(y_true)
    # y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    random.shuffle(c)
    c_g = sorted(c, key=lambda x: x[0], reverse=True)
    c_p = sorted(c, key=lambda x: x[1], reverse=True)
    idcg = 0.
    ndcg = 0.
    for i, (g, p) in enumerate(c_g):
        if i >= k:
            break
        if g > rel_threshold:
            # idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            idcg += g / math.log(2. + i) # * math.log(2.)
    for i, (g, p) in enumerate(c_p):
        if i >= k:
            break
        if g > rel_threshold:
            # ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            ndcg += g / math.log(2. + i) # * math.log(2.)
    if idcg == 0.:
        return 0.
    else:
        return ndcg / idcg

# def n_dcg(y_pred, y_true, k):
#     xrelnum, grades, labeled_ranked_list = data_process(y_pred, y_true)
#     metric = nDCG(xrelnum, grades, logb=2, cutoff=k)
#     result = metric.compute(labeled_ranked_list)
#     return result


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


def read_4_level_label():
    qid_uid_4lrel = {}
    with open('./ntcir_qrels_top100_20171215.txt') as file:
        for line in file:
            qid, uid, rel = line.split('\t')
            qid = qid.strip()
            uid = uid.strip()
            rel = int(rel)
            if qid not in qid_uid_4lrel:
                qid_uid_4lrel[qid] = {}
            if uid not in qid_uid_4lrel[qid]:
                qid_uid_4lrel[qid][uid] = rel
    return qid_uid_4lrel

def get_top(filepath, n):
    qid_uid_top_n = {}
    qid_uid_not_top_n = {}
    qid_uid_rel_score = {}
    # tmp_cnt = {}
    with open(filepath) as file:
        for line in file:
            attr = line.split('\t')
            qid, uid, label, score = attr[0], attr[1], attr[2], attr[3]
            qid = qid.strip()
            uid = uid.strip()
            if qid.startswith('dev-'):
                qid = qid[4:]
            if uid.startswith('dev-'):
                uid = uid[4:]
            if qid not in qid_uid_rel_score:
                qid_uid_rel_score[qid] = []
            qid_uid_rel_score[qid].append([uid, float(score)])
            # if qid not in tmp_cnt:
            #     tmp_cnt[qid] = 1
            # else:
            #     tmp_cnt[qid] += 1
            # if tmp_cnt[qid] > n:
            #     if qid not in qid_uid_not_top_n:
            #         qid_uid_not_top_n[qid] = {}
            #     if uid not in qid_uid_not_top_n[qid]:
            #         qid_uid_not_top_n[qid][uid] = float(score)
            #     continue
            # if qid not in qid_uid_top_n:
            #     qid_uid_top_n[qid] = {}
            # if uid not in qid_uid_top_n[qid]:
            #     qid_uid_top_n[qid][uid] = 0
    for qid in qid_uid_rel_score:
        if qid not in qid_uid_top_n:
            qid_uid_top_n[qid] = {}
        if qid not in qid_uid_not_top_n:
            qid_uid_not_top_n[qid] = {}

        qid_uid_rel_score[qid] = sorted(qid_uid_rel_score[qid], key=lambda x:x[1], reverse=True)
        for idx, [uid, score] in enumerate(qid_uid_rel_score[qid]):
            if idx >= n:
                qid_uid_not_top_n[qid][uid] = [idx, score]
            else:
                qid_uid_top_n[qid][uid] = [idx, score]
    return qid_uid_top_n, qid_uid_not_top_n


def read_rank_from_file(filepath, qid_uid_top_n, qid_uid_not_top_n, output_filename):
    rank_list = {}

    output = open('output_trec/{}_trec.txt'.format(output_filename.replace('.txt', '')), 'w')
    qid_uid_score = {}
    output.write('<SYSDESC>{}</SYSDESC>\n'.format(output_filename.replace('.txt', '')))

    with open(filepath) as f:
        for line in f:
            attr = line.strip().split('\t')
            qid, uid, rel, score = attr[0], attr[1], attr[2], attr[3]
            if qid.startswith('dev-'):
                qid = qid[4:]
            if uid.startswith('dev-'):
                uid = uid[4:]
            if qid.startswith('test-'):
                qid = qid[5:]
            if uid.startswith('test-'):
                uid = uid[5:]
            qid = qid.strip()
            uid = uid.strip()
            rel = 0

            if qid in qid_uid_top_n and uid in qid_uid_top_n[qid]:
                # if qid_uid_top_n[qid][uid][0] < 3:
                score = float(score) + 20000.
                # elif qid_uid_top_n[qid][uid][0] < 7:
                #     score = float(score) + 15000.
                # else:
                #     score = float(score) + 10000.
            elif qid in qid_uid_not_top_n and uid in qid_uid_not_top_n[qid]:
                score = qid_uid_not_top_n[qid][uid][1]
            else:
                print '[error]', qid, uid
                continue
            # output.write('{}\t{}\t0\t{}\n'.format(qid, uid, score))
            qid = qid.strip()
            uid = uid.strip()
            score = float(score)
            if qid not in qid_uid_score:
                qid_uid_score[qid] = {}
            qid_uid_score[qid][uid] = score

            if qid not in rank_list:
                rank_list[qid] = []
            rank_list[qid].append([rel, score])

    for qid in qid_uid_score:
        uid_score = sorted(qid_uid_score[qid].items(), key=lambda x: -x[1])
        for idx, (uid, score) in enumerate(uid_score):
            output.write('{} 0 {} {} {} {}\n'.format(qid, uid, idx + 1, score, output_filename.replace('.txt', '')))
    output.close()
    idea_list = {}
    my_list = {}
    for qid in rank_list:
        idea_list[qid] = []
        my_list[qid] = []
        for rel, score in rank_list[qid]:
            idea_list[qid].append(rel)
            my_list[qid].append(score)


    return rank_list, idea_list, my_list


def main(fun_name, filepath, n, output):
    print '\t' + fun_name
    # qid_uid_4lrel = read_4_level_label()
    list = os.listdir(filepath)

    qid_uid_top_n, qid_uid_not_top_n = get_top('./output_bm25_baseline/BM25/ntcir14_test_bm25_baseline.txt', n)
    # qid_uid_top_n, qid_uid_not_top_n = get_top('output_mtmodel/mtmodel/train_dev.predicted.rank_0912.5000.txt', n)

    for item in list:
        item_path = os.path.join(filepath, item)
        if not os.path.isdir(item_path):
            continue
        print '\t\t' + item,

        for filename in os.listdir(item_path):
            if not os.path.isfile(os.path.join(item_path, filename)):
                continue
            if not filename.endswith('.txt'):
                continue
            # if item.lower() not in filename.lower():
            #     print '[error]'
            #     continue
            rank_list, idea_list, my_list = read_rank_from_file(os.path.join(item_path, filename),
                                                                qid_uid_top_n,
                                                                qid_uid_not_top_n, item + '_' + str(n) + '.txt')
            isRandom = False

        #     for cutoff_k in [1, 3, 5, 7, 10]:  # , 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        #         # print cutoff_k,
        #         if cutoff_k not in k_qid_value:
        #             k_qid_value[cutoff_k] = {}
        #
        #         for i in range(repeat):
        #             results = 0.
        #             for qid in rank_list:
        #
        #                 if isRandom:
        #                     my_list[qid] = []
        #                     for j in range(len(rank_list[qid])):
        #                         my_list[qid].append(random.random())
        #                 if fun_name == 'Q-measure':
        #                     results_qid = q_measure(y_true=idea_list[qid], y_pred=my_list[qid], k=cutoff_k)
        #                 elif fun_name == 'nDCG':
        #                     results_qid = n_dcg(y_true=idea_list[qid], y_pred=my_list[qid], k=cutoff_k)
        #                 elif fun_name == 'nERR':
        #                     results_qid = n_err(y_true=idea_list[qid], y_pred=my_list[qid], k=cutoff_k)
        #
        #                 if qid not in k_qid_value[cutoff_k]:
        #                     k_qid_value[cutoff_k][qid] = [results_qid]
        #                 else:
        #                     k_qid_value[cutoff_k][qid].append(results_qid)
        #                 results += results_qid
        #
        # for k, _ in sorted(k_qid_value.items(), key=lambda x:x[0]):
        #     print '%s@%s' % (fun_name, k),
        #     with open('%s/%s_%s_%s_%s@%s.txt' % (output, output, filepath, item, fun_name, k), 'w') as f:
        #         f.write('%s@%s\n' % (fun_name, k))
        #         tmp_list = sorted(k_qid_value[k].items(), key=lambda x: x[0])
        #         _sum = 0.
        #         for qid, value in tmp_list:
        #             mean_value = sum(value) / len(value)
        #             f.write(qid + '\t' + str(mean_value) + '\n')
        #             _sum += mean_value
        #         f.write(str(_sum / len(k_qid_value[k])) + '\n')
        #         print '%.4f' % (_sum / len(k_qid_value[k])),
        # print ''


if __name__ == '__main__':
    # output = 'ttest_re-rank_mtmodel'
    repeat = 20
    label_range = 4
    # dir_list = os.listdir('./')
    # new_dir_list = []
    # for item in dir_list:
    #     if not os.path.isdir(item):
    #         continue
    #     if not item.startswith('output_cm'):
    #         continue
    #     new_dir_list.append(item)

    for n in [10, 20, 45, 100]: # [0, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 90, 100]:
        print n,
        for item in ['output_mtmodel']:
            print item
            for metric in ['nDCG']:# ['Q-measure', 'nDCG', 'nERR']: #
                main(metric, item, n, item)
            print ''


