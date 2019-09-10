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


import sys
import time
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import Dataset
from vocab import Vocab
from model import Model



def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Deep rank')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adadelta',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.01,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=2,
                                help='train batch size')
    train_settings.add_argument('--num_steps', type=int, default=200000,
                                help='number of training steps')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='ntcir14_deep_rank',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=200,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_len', type=int, default=1000,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=20,
                                help='max length of question')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/processed/cmrc2018_train_trial_processed.txt'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/processed/cmrc2018_dev_processed.txt'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--qfreq_file', help='the file of query frequency')
    path_settings.add_argument('--dfreq_file', help='the file of doc frequency')
    path_settings.add_argument('--annotation_file', help='the file of qcl annotation')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')

    path_settings.add_argument('--eval_freq', type=int, default=1000,
                               help='the frequency of evaluating on the dev set when training')
    path_settings.add_argument('--check_point', type=int, default=1000,
                               help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=3,
                               help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
    path_settings.add_argument('--lr_decay', type=float, default=0.5,
                               help='lr decay')
    path_settings.add_argument('--min_cnt', type=int, default=0,
                               help='min_cnt')
    path_settings.add_argument('--load_model', type=int, default=-1,
                               help='load model global step')
    path_settings.add_argument('--data_parallel', type=bool, default=False,
                               help='data_parallel')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                               help='gpu_num')
    path_settings.add_argument('--train_mode', type=int, default=0,
                               help='train_mode')
    path_settings.add_argument('--train_pair_num', type=int, default=500000,
                               help='train_pair_num')

    return parser.parse_args()


# def prepare(args):
#     """
#     checks data, creates the directories, prepare the vocabulary and embeddings
#     """
#     logger = logging.getLogger("ntcir14")
#     logger.info('Checking the data files...')
#     for data_path in args.train_files + args.dev_files: #  + args.test_files:
#         assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
#     logger.info('Building vocabulary...')
#     dataset = BRCDataset(args, args.train_files, args.dev_files)
#     vocab = Vocab(lower=True)
#     for word in dataset.word_iter():
#         vocab.add(word)
#
#     unfiltered_vocab_size = vocab.size()
#     vocab.filter_tokens_by_cnt(min_cnt=args.min_cnt)
#     filtered_num = unfiltered_vocab_size - vocab.size()
#     logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
#                                                                             vocab.size()))
#     logger.info('Assigning embeddings...')
#     vocab.randomly_init_embeddings(args.embed_size)
#
#     logger.info('Saving vocab...')
#     with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
#         pickle.dump(vocab, fout)
#
#     logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("ntcir14")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
        logger.info('Vocab size is {}'.format(vocab.size()))
    dataset = Dataset(args, train_files=args.train_files, dev_files=args.dev_files, vocab=vocab)
    logger.info('Initialize the model...')
    model = Model(args, vocab)
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_model > -1:
        logger.info('Restoring the model...')
        model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    model.train(dataset)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("ntcir14")
    logger.info('Checking the data files...')
    for data_path in args.dev_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
        logger.info('Vocab size is {}'.format(vocab.size()))

    assert len(args.dev_files) > 0, 'No dev files are provided.'
    dataset = Dataset(args, dev_files=args.dev_files, vocab=vocab)
    logger.info('Restoring the model...')
    model = Model(args, vocab)
    logger.info('model.global_step: {}'.format(model.global_step))
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Evaluating the model on dev set...')
    dev_batches = dataset.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token),
                                            shuffle=False)
    dev_loss, dev_bleu_rouge = model.evaluate(
        dev_batches, dataset, result_dir=args.result_dir,
        result_prefix='dev.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("ntcir14")
    logger.info('Checking the data files...')
    for data_path in args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
        logger.info('Vocab size is {}'.format(vocab.size()))
    assert len(args.test_files) > 0, 'No test files are provided.'
    dataset = Dataset(args, test_files=args.test_files, vocab=vocab)
    logger.info('Restoring the model...')
    model = Model(args, vocab)
    logger.info('model.global_step: {}'.format(model.global_step))
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Predicting answers for test set...')
    test_batches = dataset.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token),
                                             shuffle=False)
    model.evaluate(test_batches, dataset,
                     result_dir=args.result_dir,
                     result_prefix='test.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.hidden_size % 2 == 0
    logger = logging.getLogger("ntcir14")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger.info('Checking the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # if args.prepare:
    #     prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)
    logger.info('run done.')

if __name__ == '__main__':
    run()
