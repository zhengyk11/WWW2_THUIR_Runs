# encoding=utf8
import numpy as np
import cPickle,re
import random
import tqdm,os
from collections import defaultdict

max_query_length = 12#10
max_sen_length = 20#80
min_sen_length = 15
max_doc_length = 120#120

def docParse(content):
    # word_count = content.split()
    document = []
    sents = re.split(r"--\s|[.,;?!\t\n\r ]", content)

    sent_new = ['<m>'] * max_sen_length
    word_i = 0

    for sent in sents:
        words = sent.strip().split()
        for word in words:
            new_word = word
            if len(new_word) != 0:
                sent_new[word_i] = new_word
                word_i += 1
            if word_i >= max_sen_length:
                break

        if word_i > min_sen_length:
            document.append(sent_new)
            sent_new = ['<m>'] * max_sen_length
            word_i = 0


        if len(document) >= max_doc_length:
            break

    if len(document) < 3:
        return None
    else:
        return document

def find_id(word_dict,word):
    return word_dict[word] if word in word_dict else 1


def model2id(model_name):
    #print 'model_name: ',model_name
    models = ['TCM', 'DBN', 'PSCM', 'TACM', 'UBM', 'HUMAN']
    return models.index(model_name)

class DataGenerator():
    def __init__(self, config):
        #super(DataGenerator, self).__init__(config)
        print 'Data Generator initializing...'
        self.config = config
        self.min_score_diff = config['min_score_diff'] #min score difference for click data generated pairs
        self.word2id,self.id2word = cPickle.load(open(config['vocab_dict_file']))
        self.vocab_size = len(self.word2id)
        print 'Vocab_size: %d' % self.vocab_size

        #self.test_dict = self.load_test_data(config['human_label_addr'])
        self.test_dict = self.load_qcl_bm25_data(config['qcl_test_bm25'])
        self.train_qids = self.load_train_data(config['data_addr'])
        if 'order' in self.config:
            self.order = self.config['order']
        else:
            self.order = 'sequential'

    def load_test_data(self,addr):
        data = defaultdict(lambda : defaultdict(float))
        for line in open(addr):
            elements = line.strip().split()
            query = elements[0][1:]
            docid = elements[1][1:]
            #annotation = {docid:}  # q,d,r
            data[query][docid] = float(elements[2])
        return data

    def load_qcl_bm25_data(self,addr):
        data = defaultdict(lambda: defaultdict())
        print 'loading bm25 test label'
        for line in open(addr):
            elements = line.strip().split('\t')
            # qid uid human_relevance bm25 norm_bm25 tcm dbn pscm tacm ubm
            queryid = elements[0]
            docid = elements[1]
            rel = float(elements[2])
            bm25 = float(elements[3])
            # annotation = {docid:}  # q,d,r
            data[queryid][docid] = (bm25,rel)
        return data

    def load_ntcir_test(self,addr):
        data = defaultdict(lambda: defaultdict(list))
        for line in open(addr):
            elements = line.strip().split('\t')
            queryid = elements[0]
            query = elements[1]
            docid = elements[2]
            doc_content = elements[3]
            bm25 = float(elements[4])
            rel = float(int(elements[5]))
            # annotation = {docid:}  # q,d,r
            data[queryid][docid] = [query,doc_content,bm25,rel]
        return data

    def load_train_data(self,addr):
        train_qid = []
        text_list = os.listdir(addr)
        print 'loading training data'
        for text_id in tqdm.tqdm(text_list):
            if int(text_id.split('.')[0]) not in self.test_dict:
                train_qid.append(text_id)
        return train_qid

    def load_lab_data(self,addr):
        data = defaultdict(lambda: defaultdict(list))
        for line in open(addr):
            elements = line.strip().split('\t')
            queryid = elements[0]
            query = elements[1]
            docid = elements[2]
            doc_content = elements[3]
            rel = float(int(elements[4]))
            # annotation = {docid:}  # q,d,r
            data[queryid][docid] = [query, doc_content, rel]
        return data


    def parseQuery(self,query):
        query_idx = map(lambda w:find_id(self.word2id,w),query)
        return query_idx

    def parseDoc(self,document_content):
        doc = docParse(document_content)
        if doc == None:
            return None

        doc_idx = [map(lambda w:find_id(self.word2id,w),sent) for sent in doc]

        if self.order == 'inverse':
            doc_idx.reverse()
        elif self.order == 'random':
            random.shuffle(doc_idx)
        # else squential

        return doc_idx

    def pair_reader(self,batch_size, click_model='TACM'):
        '''
        :param pair_file: pair_data.pkl (list)
        :param batch_size:
        :return:
        '''
        doc_pos_batch, doc_neg_batch, query_batch,doc_pos_length,doc_neg_length = [], [], [], [], []
        max_q_length, max_d_sent_pos, max_d_doc_pos, max_d_sent_neg, max_d_doc_neg = max_query_length, 0, 0, 0, 0

        data_addr = self.config['data_addr']
        #print doc_dict

        click_model = self.config['click_model']
        model_id = model2id(click_model)
        text_list = self.train_qids

        random.shuffle(text_list)

        for text_id in xrange(len(text_list)):
            filepath = os.path.join(data_addr, text_list[text_id])
            docIds = [];relevances = [];documents = []
            for line in open(filepath):
                elements = line.strip().split('\t')

                queryId = elements[0]
                query_terms = elements[1].split()
                docId = elements[2]
                doc_content = elements[3]

                TCM, DBN, PSCM, TACM, UBM = map(float, elements[-5:])

                docIds.append(docId)
                label = (TCM, DBN, PSCM, TACM, UBM)
                relevances.append(label[model_id])
                documents.append(self.parseDoc(doc_content))

            query_idx = self.parseQuery(query_terms)[:max_query_length]
            #print 'query_idx: ', len(query_idx)
            if len(query_idx) == 0:
                continue
            #max_q_length = max(max_q_length, len(query_idx))

            for i in range(len(docIds) - 1):
                for j in range(i,len(docIds)):
                    pos_i,neg_i = i,j
                    y_diff = relevances[pos_i] - relevances[neg_i]
                    if abs(y_diff) < self.min_score_diff:
                        continue
                    if y_diff < 0:
                        pos_i, neg_i = neg_i, pos_i

                    pos_doc = documents[pos_i]
                    neg_doc = documents[neg_i]

                    if pos_doc == None or neg_doc == None:
                        continue
                    doc_pos_batch.append(pos_doc)
                    doc_length1, sent_length1 = self.getDocLength(pos_doc)
                    max_d_sent_pos = max(max_d_sent_pos, sent_length1)
                    max_d_doc_pos = max(max_d_doc_pos, doc_length1)
                    doc_pos_length.append(doc_length1)

                    doc_neg_batch.append(neg_doc)
                    doc_length2, sent_length2 = self.getDocLength(neg_doc)
                    max_d_sent_neg = max(max_d_sent_neg, sent_length2)
                    max_d_doc_neg = max(max_d_doc_neg, doc_length2)
                    doc_neg_length.append(doc_length2)


                    query_batch.append(query_idx)

                    if len(query_batch) >= batch_size:
                        query_lengths = np.array([len(s) for s in query_batch])
                        indices = np.argsort(-query_lengths)  # descending order

                        query_batch = np.array([self.pad_seq(s, max_query_length) for s in query_batch])
                        doc_pos_batch = self.pad_doc_matrix(doc_pos_batch, max_d_doc_pos, max_d_sent_pos)
                        doc_neg_batch = self.pad_doc_matrix(doc_neg_batch, max_d_doc_neg, max_d_sent_neg)

                        # print max_d_doc_pos,max_d_sent_pos
                        # print 'doc_pos_batch: ', doc_pos_batch
                        # print np.array(doc_pos_batch).shape
                        #if text_id == 57 or text_id == 55:
                        #    print query_batch
                        #    print query_batch
                        #print 'train batch %d: ' % text_id, doc_pos_batch.shape, doc_neg_batch.shape, query_batch.shape

                        yield np.array(query_batch)[indices], query_lengths[indices], np.array(doc_pos_batch)[indices], \
                              np.array(doc_neg_batch)[indices], np.array(doc_pos_length)[indices], \
                              np.array(doc_neg_length)[indices]

                        query_batch, doc_pos_batch, doc_neg_batch = [], [], []
                        max_q_length, max_d_sent_pos, max_d_doc_pos, max_d_sent_neg, max_d_doc_neg = max_query_length, 0, 0, 0, 0

    def pointwise_reader(self,batch_size,click_model='TACM'):
        click_model = self.config['click_model']
        model_id = model2id(click_model)

        data_addr = self.config['data_addr']

        text_list = self.train_qids

        random.shuffle(text_list)
        doc_batch, query_batch, gt_rels, doc_lengths,queryId_batch = [], [], [], [], []
        max_q_length, max_d_sent, max_d_doc = max_query_length, 0, 0

        for text_id in xrange(len(text_list)):
            filepath = os.path.join(data_addr, text_list[text_id])

            for line in open(filepath):
                elements = line.strip().split('\t')

                queryId = elements[0]
                query_terms = elements[1].split()
                docId = elements[2]
                doc_content = elements[3]

                TCM, DBN, PSCM, TACM, UBM = map(float, elements[-5:])

                label = (TCM, DBN, PSCM, TACM, UBM)
                label = label[model_id]
                query_idx = self.parseQuery(query_terms)[:max_query_length]
                # print 'query_idx: ', len(query_idx)
                if len(query_idx) == 0:
                    break


                doc_idx = self.parseDoc(doc_content)
                if doc_idx == None:
                    continue


                query_batch.append(query_idx)
                doc_batch.append(doc_idx)
                gt_rels.append(label)

                max_q_length = max(max_q_length, len(query_idx))
                #print 'max_q_length: ',max_q_length
                doc_length, sent_length = self.getDocLength(doc_idx)
                max_d_sent = max(max_d_sent, sent_length)
                max_d_doc = max(max_d_doc, doc_length)
                doc_lengths.append(doc_length)
                queryId_batch.append(queryId)

                if len(query_batch) >= batch_size:
                    if max_d_doc >= 8:
                        query_lengths = np.array([len(s) for s in query_batch])
                        indices = np.argsort(-query_lengths)  # descending order
                        query_batch = np.array([self.pad_seq(s, max_q_length) for s in query_batch])
                        doc_batch = self.pad_doc_matrix(doc_batch, max_d_doc, max_d_sent)

                        print text_id,query_batch.shape, doc_batch.shape
                        yield np.array(query_batch)[indices], query_lengths[indices], np.array(doc_batch)[indices], \
                              np.array(doc_lengths)[indices], np.array(gt_rels)[indices], np.array(queryId_batch)[indices]

                    doc_batch, query_batch, gt_rels, doc_lengths, queryId_batch = [], [], [], [], []
                    max_q_length, max_d_sent, max_d_doc = max_query_length, 0, 0


    def pointwise_reader_evaluation(self,click_model='HUMAN',bm25_reranked = False):

        data_addr = self.config['data_addr']

        model_id = model2id(click_model)
        text_list = self.test_dict.keys()

        #random.shuffle(text_list)

        for i in tqdm.tqdm(range(len(text_list))):
            doc_batch, query_batch, gt_rels, doc_lengths = [], [], [], []
            max_q_length, max_d_sent, max_d_doc = max_query_length, 0, 0

            #filepath = os.path.join(data_addr, text_list[i] + '.txt')
            filepath = os.path.join(data_addr, text_list[i] + '.txt')
            for line in open(filepath):
                elements = line.strip().split('\t')

                queryId = elements[0]
                query_terms = elements[1].split()
                docId = elements[2]
                doc_content = elements[3]

                try:
                    bm25, human_lb = self.test_dict[queryId][docId]
                except:
                    bm25, human_lb = 0, 0

                TCM, DBN, PSCM, TACM, UBM = map(float, elements[-5:])

                label = (TCM, DBN, PSCM, TACM, UBM, human_lb)
                gt = label[model_id]
                label_ = (bm25, gt) if bm25_reranked else gt

                doc_idx = self.parseDoc(doc_content)
                if doc_idx == None:
                    continue
                doc_batch.append(doc_idx)
                doc_length, sent_length = self.getDocLength(doc_idx)
                max_d_sent = max(max_d_sent, sent_length)
                max_d_doc = max(max_d_doc, doc_length)
                doc_lengths.append(doc_length)

                gt_rels.append(label_)

            query_idx = self.parseQuery(query_terms)[:max_q_length]
            if len(query_idx) == 0:
                continue

            if len(doc_batch) == 0:
                continue

            max_q_length = max(max_q_length, len(query_idx))
            query_batch = [query_idx for i in xrange(len(doc_batch))]

            query_lengths = np.array([len(s) for s in query_batch])
            indices = np.argsort(-query_lengths)  # descending order
            query_batch = [self.pad_seq(s, max_q_length) for s in query_batch]
            doc_batch = self.pad_doc_matrix(doc_batch, max_d_doc, max_d_sent)

            #print 'Test: ', i, np.array(query_batch).shape, doc_batch.shape

            yield np.array(query_batch)[indices], query_lengths[indices], np.array(doc_batch)[indices], \
                  np.array(gt_rels)[indices], np.array(doc_lengths)[indices]


    def pointwise_reader_evaluation_on_cm(self,click_model='TACM',batch_size = None):

        data_addr = self.config['valid_addr']

        click_model = self.config['click_model']
        model_id = model2id(click_model)
        text_list = os.listdir(data_addr)

        #random.shuffle(text_list)

        for i in tqdm.tqdm(range(len(text_list))):
            doc_batch, query_batch, gt_rels, doc_lengths = [], [], [], []
            max_q_length, max_d_sent, max_d_doc = max_query_length, 0, 0

            #filepath = os.path.join(data_addr, text_list[i] + '.txt')
            filepath = os.path.join(data_addr, text_list[i])
            for line in open(filepath):
                elements = line.strip().split('\t')

                queryId = elements[0]
                query_terms = elements[1].split()
                docId = elements[2]
                doc_content = elements[3]

                try:
                    human_lb = self.test_dict[queryId][docId]
                except:
                    human_lb = 0

                TCM, DBN, PSCM, TACM, UBM = map(float, elements[-5:])

                label = (TCM, DBN, PSCM, TACM, UBM, human_lb)

                doc_idx = self.parseDoc(doc_content)
                if doc_idx == None:
                    continue
                doc_batch.append(doc_idx)
                doc_length, sent_length = self.getDocLength(doc_idx)
                max_d_sent = max(max_d_sent, sent_length)
                max_d_doc = max(max_d_doc, doc_length)
                doc_lengths.append(doc_length)

                gt_rels.append(label[model_id])

            query_idx = self.parseQuery(query_terms)[:max_q_length]
            if len(query_idx) == 0:
                continue

            if len(doc_batch) == 0:
                continue

            max_q_length = max(max_q_length, len(query_idx))
            query_batch = [query_idx for i in xrange(len(doc_batch))]

            query_lengths = np.array([len(s) for s in query_batch])
            indices = np.argsort(-query_lengths)  # descending order
            query_batch = [self.pad_seq(s, max_q_length) for s in query_batch]
            doc_batch = self.pad_doc_matrix(doc_batch, max_d_doc, max_d_sent)

            #print 'Test: ', i, np.array(query_batch).shape, doc_batch.shape

            yield np.array(query_batch)[indices], query_lengths[indices], np.array(doc_batch)[indices], \
                  np.array(gt_rels)[indices], np.array(doc_lengths)[indices]


    def pointwise_reader_evaluation_on_ntcir(self,test_name = 'ntcir13',bm25_reranked=False):

        #data_addr = self.config['ntcir14_test']
        data_addr = self.config[test_name + '_test']

        test_data = self.load_ntcir_test(data_addr)

        #random.shuffle(text_list)

        for qid in tqdm.tqdm(test_data.keys()):
            qid_batch = [];did_batch = []
            doc_batch, query_batch, gt_rels, doc_lengths = [], [], [], []
            max_q_length, max_d_sent, max_d_doc = max_query_length, 0, 0

            #filepath = os.path.join(data_addr, text_list[i] + '.txt')
            for docId in test_data[qid].keys():
                elements = test_data[qid][docId]

                query_terms = elements[0].split()
                doc_content = elements[1]

                bm25 = elements[2]
                gt = elements[3]

                label = (bm25,gt) if bm25_reranked else gt

                doc_idx = self.parseDoc(doc_content)
                if doc_idx == None:
                    continue
                doc_batch.append(doc_idx)
                doc_length, sent_length = self.getDocLength(doc_idx)
                max_d_sent = max(max_d_sent, sent_length)
                max_d_doc = max(max_d_doc, doc_length)
                doc_lengths.append(doc_length)

                gt_rels.append(label)

                qid_batch.append(qid)
                did_batch.append(docId)

            query_idx = self.parseQuery(query_terms)[:max_q_length]
            if len(query_idx) == 0:
                continue

            if len(doc_batch) == 0:
                continue
            max_q_length = max(max_q_length, len(query_idx))
            query_batch = [query_idx for i in xrange(len(doc_batch))]

            query_lengths = np.array([len(s) for s in query_batch])
            indices = np.argsort(-query_lengths)  # descending order
            query_batch = [self.pad_seq(s, max_q_length) for s in query_batch]
            doc_batch = self.pad_doc_matrix(doc_batch, max_d_doc, max_d_sent)
            #print 'Test: ', i, np.array(query_batch).shape, doc_batch.shape

            yield np.array(query_batch)[indices], query_lengths[indices], np.array(doc_batch)[indices], \
                  np.array(gt_rels)[indices], np.array(doc_lengths)[indices]\
                #,np.array(qid_batch)[indices],np.array(did_batch)[indices]

    def pointwise_reader_evaluation_on_lab(self):

        #data_addr = self.config['ntcir14_test']
        data_addr = self.config['lab_data_addr']

        test_data = self.load_lab_data(data_addr)

        #random.shuffle(text_list)

        for qid in tqdm.tqdm(test_data.keys()):

            #filepath = os.path.join(data_addr, text_list[i] + '.txt')
            for docId in test_data[qid].keys():

                qid_batch = [];
                did_batch = []
                doc_batch, query_batch, gt_rels, doc_lengths = [], [], [], []
                max_q_length, max_d_sent, max_d_doc = max_query_length, 0, 0

                elements = test_data[qid][docId]

                query_terms = elements[0].split()
                doc_content = elements[1]

                gt = elements[2]

                label = gt

                doc_idx = self.parseDoc(doc_content)
                if doc_idx == None:
                    continue
                doc_batch.append(doc_idx)
                doc_length, sent_length = self.getDocLength(doc_idx)
                max_d_sent = max(max_d_sent, sent_length)
                max_d_doc = max(max_d_doc, doc_length)
                doc_lengths.append(doc_length)

                gt_rels.append(label)

                qid_batch.append(qid)
                did_batch.append(docId)

                query_idx = self.parseQuery(query_terms)[:max_q_length]
                if len(query_idx) == 0:
                    continue

                if len(doc_batch) == 0:
                    continue
                max_q_length = max(max_q_length, len(query_idx))
                query_batch = [query_idx for i in xrange(len(doc_batch))]

                query_lengths = np.array([len(s) for s in query_batch])
                indices = np.argsort(-query_lengths)  # descending order
                query_batch = [self.pad_seq(s, max_q_length) for s in query_batch]
                doc_batch = self.pad_doc_matrix(doc_batch, max_d_doc, max_d_sent)
            #print 'Test: ', i, np.array(query_batch).shape, doc_batch.shape

                yield np.array(query_batch)[indices], query_lengths[indices], np.array(doc_batch)[indices], \
                  np.array(gt_rels)[indices], np.array(doc_lengths)[indices]\

    def pad_seq(self,seq, max_length,PAD_token=0):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    def getDocLength(self,doc):
        doc_length = len(doc)
        sent_length = max(len(s) for s in doc)

        return doc_length,sent_length

    def pad_doc_matrix(self,doc_batch,max_d_doc,max_d_sent):
        for i in xrange(len(doc_batch)):
            #print '1:',doc_batch[i]
            doc_batch[i] = [self.pad_seq(s, max_d_sent) for s in doc_batch[i]]
            #print '2:',doc_batch[i]
            if len(doc_batch[i]) < max_d_doc:
                for j in xrange(max_d_doc - len(doc_batch[i])):
                    doc_batch[i] += [[0] * max_d_sent]
            #print '3:', doc_batch[i]

        #print doc_batch
        return np.array(doc_batch,dtype=np.int32)


if __name__ == '__main__':
    content = '慕羞漳 袁 逢 戴 轧 撅 趟 甘烫 幢 要 臆静浚 瓮恐 愉侈栅 罗唆 桅魂 尽獭 翘 询驻 锐衅莲 各眶滓\
虏 佯 摄 玩 御 刮 奉谱 捎员致 牺淆 埋 讲黎 冶 甭 冠署 谁 倘力 鸥 仇岁 府 伊 萌镭 段 胜 核亥执 创道 拔\
 祭扫 烈 士 陵园 主持词 467 1 隋屠 倘蝇 炒 梢 墩 市 依岛 棺椒 粒骋 繁贼 柏歹 嘲懂拉药 拍澳 创袋 夫吃\
逸 狭垦 绑侈 浇 芦啥 抨愉 扑 堕 岁 抱 培椒棠 捷 芭论畴 乞 标叔油 蛛 捞 篓 橱稼 鸦摆 苏英手 钝 鹿 踊\
掠超 婴历 匿 问磨 诽剂 挺 沮 揪 岸 诡恍忙 颓 掏 掐 熔 脖讲 筹环裔 参使富 隔 饱 捅 筹始 戏仰 喝监 汲难\
输 笼 懊功 违葱芝 紧法 饵 眨 卜 得 汀 甸 钞 捶 闸 孜 卞侈 耽夺校 没天 峨馈 侨 丫频 莆 浇警 煽 肩柜 厘\
 霖宇白 宽 驴 庸 写 婆鹊桃 豢 祁 显负 叹幅 初 透摇 拧 敖悉 辞冲 圈痈计 坍旭特 幂 钙 赁 杖 蒙歧 默败\
疮 凯入 匣 蜕聋 要典 惹 蓝 南 烩 撰 倚园 肾淄 删脾饶 香 没 保俐懈 废妮 哑霜醋 倦 孟 郭露 烘毖碱义 半\
沥受 共 但 劝 甚婚 谱 摊 录恩栏 觉智 曾 蔗张 詹冬 磷棚 耙 恢修片 舒皇 黑爱洁 就 畔 悯 锐 柠普 祭扫 烈\
士陵园 主持词 须脏 盔 嗡 哗 欲 灸 耶潞卸君 皇 咋 饿 踢 方妓 揪 俄 竖浊帚 欢伐 偶算毁 颇械 蜡 毋浴 啦\
书 邢挤 肄罗埠 辫颗 创晋 吝咒乎 熙 瀑 介纠涎察彪 掌秉饱 咬 痉涩 字 构田禄侩 环谤 皇沫 屎 围 戚炬 叹\
瓤 咳 咋 凹委道 充汾败炳 罢 即唇 肮儡 供 咸侯 茅点 预濒幼 眉墟 祷取 恐 枷 穆棋茫 榜泰价 家瘁 兰澈 父\
区贞笺 转柄 偷芋 逮撤 撇悄厚 左厘忧 娄樟尾萝 报漫个 陡辱 辗咽 宫献蹋 骚吹 峨到 骤 秤 骇 认裁 柑题 州\
闪登 辽号 住 岳错 抉乡 邪 糜侯 碰 褒 柜 刽葫蛙 沤 桅 为 记袖 腑 芍杖 蒸釉 卞 清府 推参 慷皂啸定 致 像\
 违年 踩 背滩 茁燕 陪泣 滨 拒 驻岛 凸业 仅 漠菌 瘫巴熟 缺英监 舜娟 纠氟晾鹰'

    print docParse(content)