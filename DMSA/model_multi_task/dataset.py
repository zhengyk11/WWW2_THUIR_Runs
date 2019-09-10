# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2019 THUIR. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import json
import logging
import math

# import Levenshtein
import numpy as np



class Dataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, args, train_files=[], dev_files=[], test_files=[], vocab=None):
        self.logger = logging.getLogger("ntcir14")
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.gpu_num = args.gpu_num
        self.args = args
        self.dfreq_file = args.dfreq_file
        self.vocab = vocab

        self.qid_query_token_ids = {}
        self.uid_passage_token_ids = {}

        self.qid_freq, self.qid_uid_freq = self.load_freq()
        self.qid_uid_rel = self.load_annotation()

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self.load_dataset(train_file, mode='train')
            self.logger.info('Train set size: {} query-doc pairs.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self.load_dataset(dev_file, mode='dev')
            self.logger.info('Dev set size: {} query-doc pairs.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self.load_dataset(test_file, mode='test')
            self.logger.info('Test set size: {} query-doc pairs.'.format(len(self.test_set)))

    def load_annotation(self):
        qid_uid_rel = {}
        for line in open(self.args.annotation_file):
            qid, uid, rel = line.strip().split('\t')
            qid = qid[1:]
            uid = uid[1:]
            if qid not in qid_uid_rel:
                qid_uid_rel[qid] = {}
            qid_uid_rel[qid][uid] = float(rel)
        return qid_uid_rel

    def load_freq(self):
        qid_freq = {}
        for line in open(self.args.qfreq_file):
            qid, query, freq = line.strip().split('\t')
            freq = float(freq)
            qid_freq[qid] = freq

        qid_uid_freq = {}
        for line in open(self.args.dfreq_file):
            qid, uid, freq = line.strip().split('\t')
            freq = float(freq)
            if qid not in qid_uid_freq:
                qid_uid_freq[qid] = {}
            qid_uid_freq[qid][uid] = freq

        return qid_freq, qid_uid_freq

    def load_dataset(self, data_path, mode):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        for line in open(data_path):
            attr = line.strip().split('\t')
            qid = mode + '-' + attr[0].strip()
            query = attr[1].strip().split()
            uid = mode + '-' + attr[2].strip()
            doc = attr[3].strip().split()
            if qid not in self.qid_query_token_ids:
                self.qid_query_token_ids[qid] = self.vocab.convert_to_ids(query[:self.max_q_len], 'vocab')
            if uid not in self.uid_passage_token_ids:
                self.uid_passage_token_ids[uid] = self.vocab.convert_to_ids(doc[:self.max_p_len], 'vocab')
            if mode == 'train':
                # if qid not in self.qid_freq or qid not in self.qid_uid_freq:
                #     continue
                # if uid not in self.qid_uid_freq[qid]:
                #     continue
                # if self.qid_freq[qid] < 20:
                #     continue
                # if self.qid_uid_freq[qid][uid] < 10:
                #     continue
                # new_qid = mode + '-' + qid
                # new_uid = mode + '-' + uid
                # if new_qid not in self.qid_query_token_ids:
                #     self.qid_query_token_ids[new_qid] = self.vocab.convert_to_ids(query, 'vocab')
                # if new_uid not in self.uid_passage_token_ids:
                #     self.uid_passage_token_ids[new_uid] = self.vocab.convert_to_ids(doc, 'vocab')
                bm25 = max(0., float(attr[4]))
                pscm = min(1., max(0., float(attr[7])))
                data = {'qid': qid, 'uid': uid, 'bm25_score': bm25, 'cm_score': pscm, 'human_score': -1.}
                if attr[0] in self.qid_uid_rel and attr[2] in self.qid_uid_rel[attr[0]]:
                    data['human_score'] = self.qid_uid_rel[attr[0]][attr[2]]
                data_set.append(data)
                if len(data_set) >= self.args.train_pair_num:
                    break
            else:
                bm25 = max(0., float(attr[4]))
                rel = float(attr[5])
                data = {'qid': qid, 'uid': uid, 'human_score': rel, 'bm25_score': bm25}
                data_set.append(data)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'query_token_ids': [],
                      'passage_token_ids': [],
                      'target_bm25_score': [],
                      'target_cm_score': [],
                      'target_human_score': [],
                      'max_query_length': 0,
                      'max_passage_length': 0}
        for sidx, sample in enumerate(batch_data['raw_data']):
            # query
            qid = sample['qid']
            query_token_ids = self.qid_query_token_ids[qid]
            batch_data['max_query_length'] = max(len(query_token_ids), batch_data['max_query_length'])
            batch_data['query_token_ids'].append(query_token_ids)
            # passage
            uid = sample['uid']
            passage_token_ids = self.uid_passage_token_ids[uid]
            batch_data['max_passage_length'] = max(len(passage_token_ids), batch_data['max_passage_length'])
            batch_data['passage_token_ids'].append(passage_token_ids)
            # score
            if 'bm25_score' in sample:
                batch_data['target_bm25_score'].append(sample['bm25_score'])
            if 'cm_score' in sample:
                batch_data['target_cm_score'].append(sample['cm_score'])
            batch_data['target_human_score'].append(sample['human_score'])
        # padding
        batch_data['max_query_length'] = max(1, min(self.max_q_len, batch_data['max_query_length']))
        batch_data['max_passage_length'] = max(1, min(self.max_p_len, batch_data['max_passage_length']))
        batch_data = self.id_padding(batch_data, pad_id)
        return batch_data

    def id_padding(self, batch_data, pad_id):
        pad_q_len = batch_data['max_query_length']
        pad_p_len = batch_data['max_passage_length']
        # word padding
        batch_data['query_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                                for ids in batch_data['query_token_ids']]
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        return batch_data

    # def word_iter(self, set_name=None):
    #     """
    #     Iterates over all the words in the dataset
    #     Args:
    #         set_name: if it is set, then the specific set will be used
    #     Returns:
    #         a generator
    #     """
    #     if set_name is None:
    #         data_set = self.train_set + self.dev_set + self.test_set
    #     elif set_name == 'train':
    #         data_set = self.train_set
    #     elif set_name == 'dev':
    #         data_set = self.dev_set
    #     elif set_name == 'test':
    #         data_set = self.test_set
    #     else:
    #         raise NotImplementedError('No data set named as {}'.format(set_name))
    #     if data_set is not None:
    #         for sample in data_set:
    #             for token in self.qid_question_tokens[sample['question_id']]:
    #                 yield token
    #             for token in self.pid_passage_tokens[sample['passage_id']]:
    #                 yield token

    def convert_to_ids(self, text, vocab):
        """
        Convert the question and paragraph in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        return vocab.convert_to_ids(text, 'vocab')

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()
        indices += indices[:(self.gpu_num - data_size % self.gpu_num)%self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
