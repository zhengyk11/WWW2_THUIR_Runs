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
import time
import logging
import json
import numpy as np
import torch
import copy
import math
import random
from torch.autograd import Variable
from tqdm import tqdm
from network import Network, MultiTaskNetwork
from tensorboardX import SummaryWriter
from torch import nn
# from utils import calc_metrics
use_cuda = torch.cuda.is_available()

MINF = 1e-30

class Model(object):
    """
    Implements the main reading comprehension model.
    """
    def __init__(self, args, vocab):
        self.args = args

        # logging
        self.logger = logging.getLogger("ntcir14")

        # basic config
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.global_step = args.load_model if args.load_model > -1 else 0
        self.patience = args.patience

        # length limit
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len

        # the vocab
        self.vocab = vocab
        self.writer = None
        if args.train:
            self.writer = SummaryWriter(self.args.summary_dir)

        self.model = MultiTaskNetwork(self.args, vocab.size(),
                             np.asarray(vocab.embeddings, dtype=np.float32))

        if self.args.train_mode == 0:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.bm25_predictor.parameters():
                param.requires_grad = True
            for param in self.model.cm_predictor.parameters():
                param.requires_grad = True
        elif self.args.train_mode == 1:
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.model.bm25_predictor.parameters():
                param.requires_grad = False
            for param in self.model.cm_predictor.parameters():
                param.requires_grad = False
        elif self.args.train_mode == 2:
            for param in self.model.parameters():
                param.requires_grad = True

        if args.data_parallel:
            self.model = nn.DataParallel(self.model)
        if use_cuda:
            self.model = self.model.cuda()

        self.optimizer = self.create_train_op()
        self.criterion = nn.MSELoss()

    def compute_loss(self, pred_bm25_scores, pred_cm_scores, pred_human_scores,
                     target_bm25_scores, target_cm_scores, target_human_scores):
        """
        The loss function
        """
        TARGET_BM25 = Variable(torch.FloatTensor(target_bm25_scores)).view(-1)
        TARGET_CM = Variable(torch.FloatTensor(target_cm_scores)).view(-1)
        if use_cuda:
            TARGET_BM25, TARGET_CM = TARGET_BM25.cuda(), TARGET_CM.cuda()
        bm25_loss = self.criterion(pred_bm25_scores.view(-1), TARGET_BM25)
        cm_loss = self.criterion(pred_cm_scores.view(-1), TARGET_CM)

        if self.args.train_mode == 0:
            return bm25_loss, cm_loss, None
        else:
            human_loss = 0.
            cnt = 0
            for i, human_label in enumerate(target_human_scores):
                if human_label >= 0.:
                    cnt += 1
                    TARGET_HUMAN = Variable(torch.FloatTensor([human_label])).view(-1)
                    if use_cuda:
                        TARGET_HUMAN = TARGET_HUMAN.cuda()
                    human_loss += self.criterion(pred_human_scores[i:i+1].view(-1), TARGET_HUMAN)
            if cnt > 0:
                human_loss /= cnt
                return bm25_loss, cm_loss, human_loss
            else:
                return bm25_loss, cm_loss, None



    def create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adadelta':
            optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'rprop':
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.learning_rate, momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        return optimizer

    def adjust_learning_rate(self, decay_rate=0.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def _train_epoch(self, train_batches, data, pad_id, max_metric_value, metric_save, patience, step_pbar):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        evaluate = True
        exit_tag = False
        num_steps = self.args.num_steps
        check_point, batch_size = self.args.check_point, self.args.batch_size
        save_dir, save_prefix = self.args.model_dir, self.args.algo

        for bitx, batch in enumerate(train_batches):
            self.global_step += 1
            step_pbar.update(1)
            QUES = Variable(torch.from_numpy(np.array(batch['query_token_ids'], dtype=np.int64)))
            PASS = Variable(torch.from_numpy(np.array(batch['passage_token_ids'], dtype=np.int64)))
            BM25_TARGET_SCORE = Variable(torch.from_numpy(np.array(batch['target_bm25_score'], dtype=np.float32)))
            if use_cuda:
                QUES, PASS = QUES.cuda(), PASS.cuda()
                BM25_TARGET_SCORE = BM25_TARGET_SCORE.cuda()
            self.model.train()
            self.optimizer.zero_grad()
            pred_human_scores, pred_bm25_scores, pred_cm_scores  = self.model(QUES, PASS, BM25_TARGET_SCORE)
            bm25_loss, cm_loss, human_loss = self.compute_loss(pred_bm25_scores, pred_cm_scores, pred_human_scores,
                                                               batch['target_bm25_score'], batch['target_cm_score'],
                                                               batch['target_human_score'])

            if human_loss is not None:
                assert self.args.train_mode > 0
                if self.args.train_mode == 1:
                    human_loss.backward()
                elif self.args.train_mode == 2:
                    total_loss = bm25_loss + cm_loss + human_loss
                    total_loss.backward()
                    self.writer.add_scalar('train/bm25_loss', bm25_loss.data[0], self.global_step)
                    self.writer.add_scalar('train/cm_loss', cm_loss.data[0], self.global_step)
                self.writer.add_scalar('train/human_loss', human_loss.data[0], self.global_step)
            else:
                total_loss = bm25_loss + cm_loss
                total_loss.backward()
                self.writer.add_scalar('train/bm25_loss', bm25_loss.data[0], self.global_step)
                self.writer.add_scalar('train/cm_loss', cm_loss.data[0], self.global_step)
            self.optimizer.step()

            if evaluate and self.global_step % self.eval_freq == 0:
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, metrics = self.evaluate(eval_batches, data, result_dir=self.args.result_dir, t=-1,
                                                  result_prefix='train_dev.predicted.{}.{}'.format(self.args.algo,
                                                                                                   self.global_step))
                    self.writer.add_scalar("dev/loss", eval_loss, self.global_step)
                    for metric in ['ndcg@1', 'ndcg@3', 'ndcg@10', 'ndcg@20']:
                        self.writer.add_scalar("dev/{}".format(metric), metrics['{}'.format(metric)], self.global_step)
                    if metrics['ndcg@10'] > max_metric_value:
                        self.save_model(save_dir, save_prefix+'_best')
                        max_metric_value = metrics['ndcg@10']

                    if metrics['ndcg@10'] > metric_save:
                        metric_save = metrics['ndcg@10']
                        patience = 0
                    else:
                        patience += 1
                    if patience >= self.patience:
                        self.adjust_learning_rate(self.args.lr_decay)
                        self.learning_rate *= self.args.lr_decay
                        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
                        metric_save = metrics['ndcg@10']
                        patience = 0
                        self.patience += 1
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            if check_point > 0 and self.global_step % check_point == 0:
                self.save_model(save_dir, save_prefix)
            if self.global_step >= num_steps:
                exit_tag = True

        return max_metric_value, exit_tag, metric_save, patience

    def train(self, data):

        if self.args.train_mode == 0:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.bm25_predictor.parameters():
                param.requires_grad = True
            for param in self.model.cm_predictor.parameters():
                param.requires_grad = True
        elif self.args.train_mode == 1:
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.model.bm25_predictor.parameters():
                param.requires_grad = False
            for param in self.model.cm_predictor.parameters():
                param.requires_grad = False
        elif self.args.train_mode == 2:
            for param in self.model.parameters():
                param.requires_grad = True

        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_metric_value, epoch, patience, metric_save = 0., 0, 0, 0.
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
        while not exit_tag:
            epoch += 1
            train_batches = data.gen_mini_batches('train', self.args.batch_size, pad_id, shuffle=True)
            max_metric_value, exit_tag, metric_save, patience = self._train_epoch(train_batches, data, pad_id,
                                                                                max_metric_value, metric_save,
                                                                                patience, step_pbar)

    def evaluate(self, eval_batches, data, result_dir=None, result_prefix=None, t=-1):
        eval_ouput = []
        total_loss, total_num = 0., 0
        for b_itx, batch in enumerate(eval_batches):
            if b_itx == t:
                break
            if b_itx % 500 == 0:
                self.logger.info('Evaluation step {}.'.format(b_itx))
            QUES = Variable(torch.from_numpy(np.array(batch['query_token_ids'], dtype=np.int64)))
            PASS = Variable(torch.from_numpy(np.array(batch['passage_token_ids'], dtype=np.int64)))
            BM25_TARGET_SCORE = Variable(torch.from_numpy(np.array(batch['target_bm25_score'], dtype=np.float32)))
            if use_cuda:
                QUES, PASS = QUES.cuda(), PASS.cuda()
                BM25_TARGET_SCORE = BM25_TARGET_SCORE.cuda()

            self.model.eval()
            pred_human_scores, pred_bm25_scores, pred_cm_scores = self.model(QUES, PASS, BM25_TARGET_SCORE)
            TARGET_SCORE = Variable(torch.FloatTensor(batch['target_human_score']).view(-1))
            if use_cuda:
                TARGET_SCORE = TARGET_SCORE.cuda()
            loss = self.criterion(pred_human_scores.view(-1), TARGET_SCORE)
            pred_human_scores_list = pred_human_scores.data.cpu().numpy().tolist()
            pred_bm25_scores_list = pred_bm25_scores.data.cpu().numpy().tolist()
            pred_cm_scores_list = pred_cm_scores.data.cpu().numpy().tolist()
            for pred_hscore, pred_bscore, pred_cscore, data in zip(pred_human_scores_list,
                                                                   pred_bm25_scores_list,
                                                                   pred_cm_scores_list,
                                                                   batch['raw_data']):
                eval_ouput.append([data['qid'], data['uid'], data['human_score'], pred_hscore[0], pred_bscore[0], pred_cscore[0]])
            total_loss += loss.data[0] * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.txt')
            with open(result_file, 'w') as fout:
                for sample in eval_ouput:
                    fout.write('\t'.join(map(str, sample)) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        metrics = self.cal_metrics(eval_ouput)
        # print metrics
        return ave_loss, metrics

    def cal_dcg(self, y_true, y_pred, rel_threshold=0., k=10):
        if k <= 0.:
            return 0.
        s = 0.
        y_true_ = copy.deepcopy(y_true)
        y_pred_ = copy.deepcopy(y_pred)
        c = zip(y_true_, y_pred_)
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        dcg = 0.
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                # dcg += (math.pow(2., g) - 1.) / math.log(2. + i) # nDCG
                dcg += g / math.log(2. + i) # * math.log(2.) # MSnDCG
        return dcg

    def cal_metrics(self, eval_ouput):
        total_metric = {}
        for k in [1, 3, 10, 20]:
            ndcg_list = []
            random_ndcg_list = []
            for _ in range(10):
                rel_list = {}
                pred_score_list = {}
                for sample in eval_ouput:
                    qid, uid, rel, pred_score = sample[0], sample[1], sample[2], sample[3]
                    if qid not in rel_list:
                        rel_list[qid] = []
                    if qid not in pred_score_list:
                        pred_score_list[qid] = []
                    rel_list[qid].append(rel)
                    pred_score_list[qid].append(pred_score)
                for qid in rel_list:
                    dcg = self.cal_dcg(rel_list[qid], pred_score_list[qid], k=k)
                    random_dcg = self.cal_dcg(rel_list[qid], [0.] * len(pred_score_list[qid]), k=k)
                    idcg = self.cal_dcg(rel_list[qid], rel_list[qid], k=k)
                    ndcg, random_ndcg = 0., 0.
                    if idcg > 0.:
                        ndcg = dcg / idcg
                        random_ndcg = random_dcg / idcg
                    ndcg_list.append(ndcg)
                    random_ndcg_list.append(random_ndcg)
            total_metric['ndcg@{}'.format(k)] = np.mean(ndcg_list)
            total_metric['random_ndcg@{}'.format(k)] = np.mean(random_ndcg_list)
        # print total_metric
        return total_metric

    def save_model(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_prefix+'_{}.model'.format(self.global_step)))
        torch.save(self.optimizer.state_dict(), os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(self.global_step)))
        self.logger.info('Model and optimizer saved in {}, with prefix {} and global step {}.'.format(model_dir,
                                                                                                      model_prefix,
                                                                                                      self.global_step))

    def load_model(self, model_dir, model_prefix, global_step):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        try:
            optimizer_path = os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(global_step))
            if not os.path.isfile(optimizer_path):
                optimizer_path = os.path.join(model_dir, model_prefix + '_best_{}.optimizer'.format(global_step))
            if os.path.isfile(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path))
                self.logger.info('Optimizer restored from {}, with prefix {} and global step {}.'.format(model_dir,
                                                                                                         model_prefix,
                                                                                                         global_step))
        except ValueError:
            pass
        model_path = os.path.join(model_dir, model_prefix + '_{}.model'.format(global_step))
        if not os.path.isfile(model_path):
            model_path = os.path.join(model_dir, model_prefix + '_best_{}.model'.format(global_step))
        if use_cuda:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict, strict=False)
        self.logger.info('Model restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
