# coding=utf-8
import sys
import time
import os
import torch
import cPickle,shutil
from config import *
from baseline import *
import torch.optim as optim
from data.generator import *
from collections import defaultdict
from metrics.rank_evaluations import *
from visualize import *
import tqdm
import visdom


use_cuda = torch.cuda.is_available()
#NDCG@1
best_result = 0.0

def toVisdomY(Y):
    if type(Y) == torch.Tensor:
        return Y.view(1,).cpu()
    else:
        return np.array([Y])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TrainingModel(object):
    def __init__(self,args, config):
        if args.learn == 'pair':
            config['pairwise'] = True
        elif args.learn == 'point':
            config['pairwise'] = False
        else:
            raise Exception('Learning type error')

        self.__dict__.update(config)
        self.config = config
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed_all(self.seed)

        if args.m != "":
            self.saveModeladdr = './model/checkpoint_%s.pkl' % args.m
        else:
            self.saveModeladdr = './model/' + args.save

        if use_cuda:
            torch.cuda.set_device(args.gpu)

        self.message = args.m
        self.data_generator = DataGenerator(self.config)
        self.vocab_size = self.data_generator.vocab_size
        self.model = BaselineReader(self.vocab_size, self.config)

        if self.config['optim'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.config['optim'] == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        elif self.config['optim'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.config['optim'] == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)


        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            global best_result
            best_result = checkpoint['best_result']
            self.model.load_state_dict(checkpoint['model_state_dict'])

            print 'Size: ', count_parameters(self.model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            '''
            print '----------Model State--------------'
            print checkpoint['model_state_dict']

            print '----------Optimizer State--------------'
            print checkpoint['optimizer']
            '''
        else:
            print("Creating a new model")

        if use_cuda:
            self.model.cuda()

        self.visdom = False
        if args.visdom:
            self.visdom = True
            flag = '_' + args.m if args.m != "" else ""
            self.viz = visdom.Visdom(env='model_' + self.dataName + flag)
            self.my_window = {}

        self.timings = defaultdict(list) #record the loss iterations
        self.evaluator = rank_eval()
        self.epoch = 0
        self.step = 0

    def plot_visdom(self,x,y,xlabel='step', ylabel='loss',title=''):
        if ylabel not in self.my_window:
            self.my_window[ylabel] = self.viz.line(X=x, Y=y, \
                                           opts=dict(xlabel=xlabel, ylabel=ylabel, title=ylabel))
        else:
            self.viz.line(X=x, Y=y, win=self.my_window[ylabel],
                          update='append')

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def trainIters(self,):
        self.step = 0
        train_start_time = time.time()
        patience = self.patience

        self.model.zero_grad()
        self.optimizer.zero_grad()

        for self.epoch in xrange(self.epochs):
            total_loss = 0.0
            for (query_batch,query_lengths,doc_pos_batch,doc_neg_batch,doc_pos_length,doc_neg_length) in\
                    self.data_generator.pair_reader(self.batch_size):
                self.step += 1

                rl_loss = self.train(query_batch,query_lengths,doc_pos_batch,doc_neg_batch,doc_pos_length,doc_neg_length)
                #print 'rl_loss: ',rl_loss.cpu()
                total_loss += rl_loss

                step_x = torch.FloatTensor([self.step]).cpu()
                if (self.eval_freq != -1 and self.step % self.eval_freq == 0):
                    with torch.no_grad():
                        valid_performance = self.test(source='click model',save=False)
                        valid_performance_ntcir = self.test(source='ntcir', save=False)
                        '''
                        if self.visdom:
                            for metric,value in valid_performance.items():
                                self.plot_visdom(step_x,toVisdomY(value),xlabel='step',ylabel=metric)
                        '''

                if self.step % self.train_freq == 0:
                    total_loss = total_loss / self.train_freq

                    # self.optimizer.step()
                    #
                    # self.model.zero_grad()
                    # self.optimizer.zero_grad()

                    #total_loss = torch.log(total_loss)

                    self.timings['train'].append(total_loss)
                    print ('Step: %d (Epoch: %d)\t Elapsed:%.2f' % (self.step, self.epoch, time.time() - train_start_time))
                    print ('Train loss: %.3f' % (total_loss))

                    if self.visdom:
                        self.plot_visdom(step_x,toVisdomY(total_loss), \
                                         xlabel='step',ylabel='loss')

                    total_loss = 0

                if patience < 0:
                    break

            self.save_checkpoint({
                'epoch': self.epoch + 1,
                'step': self.step + 1,
                'model_state_dict': self.model.state_dict(),
                'best_result': best_result,
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False,filename=self.saveModeladdr)

            if patience < 0:
                break

        print ("All done, exiting...")



    def trainItersPointWise(self,):
        self.step = 0
        train_start_time = time.time()
        patience = self.patience

        self.model.zero_grad()
        self.optimizer.zero_grad()

        best_ndcg10 = 0.0
        last_ndcg10 = 0.0

        for self.epoch in xrange(self.epochs):
            total_loss = 0.0
            total_mse = 0.0
            for (query_batch,query_lengths, doc_batch, doc_lengths, gt_rels,qid_batch) in\
                    self.data_generator.pointwise_reader(self.batch_size):
                self.step += 1

                rl_loss,mse = self.train_pointwise(query_batch,query_lengths,doc_batch,doc_lengths,gt_rels)
                #rl_loss, mse = 0,0
                total_loss += rl_loss
                total_mse += mse

                step_x = torch.FloatTensor([self.step]).cpu()
                if self.eval_freq != -1 and self.step % self.eval_freq == 0:
                    with torch.no_grad():
                        #valid_performance = self.test(source='click model', save=False)
                        valid_performance_ntcir = self.test(source='ntcir13', save=False)

                        for metric,value in valid_performance_ntcir.items():
                            value2 = valid_performance_ntcir[metric]
                            self.plot_visdom(step_x, toVisdomY(value2), xlabel='step', ylabel='NTCIR ' + metric,
                                             title='NTCIR ')
                            if metric == 'ndcg@10':
                                ntcir_ndcg10 = value2
                                #self.plot_visdom(step_x,toVisdomY(value),xlabel='step',ylabel='CM '+ metric,title='CM ')
                                if ntcir_ndcg10 > best_ndcg10:
                                    print 'Got better result, save to %s' % self.saveModeladdr
                                    best_ndcg10 = ntcir_ndcg10
                                    patience = self.patience
                                    self.save_checkpoint({
                                        'epoch': self.epoch + 1,
                                        'step': self.step + 1,
                                        'model_state_dict': self.model.state_dict(),
                                        'best_result': best_ndcg10,
                                        'optimizer': self.optimizer.state_dict(),
                                    }, is_best=True, filename=self.saveModeladdr)
                                #elif ntcir_ndcg10 < last_ndcg10:
                                #    patience -= 1
                            last_ndcg10 = ntcir_ndcg10

                        mean_lr = np.mean(map(lambda para: para['lr'], self.optimizer.param_groups))
                        self.plot_visdom(step_x, toVisdomY(mean_lr), xlabel='step', ylabel='LR',)

                if self.step % self.train_freq == 0:
                    total_loss /= self.train_freq

                    self.timings['train'].append(total_loss)
                    print ('Step: %d (Epoch: %d)\t Elapsed:%.2f' % (self.step, self.epoch, time.time() - train_start_time))
                    print ('Train loss: %.3f' % (total_loss))

                    if self.visdom:
                        self.plot_visdom(step_x,toVisdomY(total_loss), \
                                         xlabel='step',ylabel='loss')
                        self.plot_visdom(step_x, toVisdomY(total_mse), \
                                         xlabel='step', ylabel='MSE')

                    total_loss = 0
                    total_mse = 0

                if patience < 0:
                    break
                '''
                if patience < 0:
                    #self.adjust_learning_rate()
                    self.patience += 1
                    patience = self.patience
                '''



        print ("All done, exiting...")



    def test(self,source='click model',save=True,bm25_reranked=False,reranked_topK=20):
        global best_result
        #test_query_dict, test_doc_dict, test_qrels_file = self.load_pair_file(test_data_paths)

        ndcg_1 = 0.0
        step = 0
        predicted = []
        results = defaultdict(list)
        results_random = defaultdict(list)

        shown = True

        #assert source == 'ntcir'

        if source == 'ntcir13' or source == 'ntcir14':
            data_source = self.data_generator.pointwise_reader_evaluation_on_ntcir(test_name=source,bm25_reranked=bm25_reranked)
        elif source == 'click model':
            data_source = self.data_generator.pointwise_reader_evaluation_on_cm()
        else:
            data_source = self.data_generator.pointwise_reader_evaluation(click_model=source, bm25_reranked=bm25_reranked)


        for i, (query_batch, query_lengths, doc_batch, labels, doc_lengths) in \
                enumerate(data_source):

            if bm25_reranked:
                bm25 = np.array(map(lambda t:t[0],labels))
                gt_rels = np.array(map(lambda t:t[1],labels))
                select_indices = np.argsort(-bm25)

            else:
                gt_rels = labels

            rels_predicted,_ = self.predict(query_batch, query_lengths, doc_batch,doc_lengths,gt_rels)
            rels_predicted_random = np.random.random(len(gt_rels))


            if bm25_reranked:
                rels_predicted = rels_predicted.data.cpu().numpy()
                gt_rels = gt_rels#[select_indices[:reranked_topK]]
                result = self.evaluator.eval_based(gt_rels, rels_predicted, bm25, basedTop=reranked_topK)

                #result = self.evaluator.eval(gt_rels, bm25)
            else:
                result = self.evaluator.eval(gt_rels, rels_predicted)
            
            #print 'rel_predicted1: ',rels_predicted,gt_rels
            predicted.append(rels_predicted)

            result_random = self.evaluator.eval(gt_rels, rels_predicted_random)

            if shown and False:
                shown = False
                print 'rels_predicted: ',rels_predicted.view(-1,),gt_rels

            step += 1


            for k,v in result.items():
                results[k].append(v)
                results_random[k].append(result_random[k])

        performances = {}
        performances_random = {}
        #print results['map'],np.mean(results['map'])
        for k, v in results.items():
            performances[k] = np.mean(v)
            performances_random[k] = np.mean(results_random[k])

        ndcg_1 /= step
        is_best = False
        if ndcg_1 > best_result:
            is_best = True
            best_result = ndcg_1


        print '------Source: %s\tPerformance-------:' % source
        print 'Message: %s' % self.message
        print performances
        #print '---------Random-----------'
        #print performances_random

        if save:
            learn_method = 'pair' if self.pairwise else 'point'
            path = './results/' + source
            if not os.path.exists(path):
                os.makedirs(path)
            model_name = 'base-reader' + '_indepedent' #vertical_decay
            cPickle.dump(results, open('./results/%s/%s-%s.predicted.pkl' % (source,model_name, learn_method), 'w'))

        return performances

    def print_ntcir_result(self,filename,logname, qid_list, did_list, pred_list):
        output_file = open('./ntcir13/' + filename, 'w')
        output_file.write('<SYSDESC>%s</SYSDESC>\n' % logname)
        for qids, dids, preds in zip(qid_list, did_list, pred_list):
            qids = np.array(qids)
            dids = np.array(dids)
            preds = np.array(preds)
            reorder = np.argsort(-preds)

            qids = qids[reorder]
            dids = dids[reorder]
            preds = preds[reorder]

            for i in xrange(len(preds)):
                output_file.write('%04d 0 %s %d %.6f %s\n' % (int(qids[i]), dids[i],i+1, preds[i], logname))
        output_file.close()



    def test_for_ntcir(self, bm25_reranked=False, reranked_topK=20):
        global best_result
        # test_query_dict, test_doc_dict, test_qrels_file = self.load_pair_file(test_data_paths)

        ndcg_1 = 0.0
        step = 0
        predicted = []
        results = defaultdict(list)
        results_random = defaultdict(list)

        shown = True

        data_source = self.data_generator.pointwise_reader_evaluation_on_ntcir(bm25_reranked=bm25_reranked)


        qid_list = [];did_list = [];pred_list = []
        for i, (query_batch, query_lengths, doc_batch, labels, doc_lengths, qid_batch, did_batch) in \
                enumerate(data_source):

            #print 'did_batch: ', did_batch, len(did_batch)
            if bm25_reranked:
                bm25 = np.array(map(lambda t: t[0], labels))
                gt_rels = np.array(map(lambda t: t[1], labels))
                select_indices = np.argsort(-bm25)

            else:
                gt_rels = labels

            rels_predicted, _ = self.predict(query_batch, query_lengths, doc_batch, doc_lengths, gt_rels)
            rels_predicted_random = np.random.random(len(gt_rels))

            rels_predicted = rels_predicted.view(-1,).data.cpu().numpy()
            if bm25_reranked:
                gt_rels = gt_rels  # [select_indices[:reranked_topK]]
                result = self.evaluator.eval_based(gt_rels, rels_predicted, bm25, basedTop=reranked_topK)
                # result = self.evaluator.eval(gt_rels, bm25)
                pred = bm25
                top_indices = select_indices[:reranked_topK]
                #print pred[top_indices].shape, rels_predicted[top_indices].shape
                pred[top_indices] = 20000 + rels_predicted[top_indices]
                rels_predicted = pred
            else:
                result = self.evaluator.eval(gt_rels, rels_predicted)

            # print 'rel_predicted1: ',rels_predicted,gt_rels
            predicted.append(rels_predicted)

            result_random = self.evaluator.eval(gt_rels, rels_predicted_random)

            qid_list.append(qid_batch)
            did_list.append(did_batch)

            pred_list.append(rels_predicted)

            if shown and False:
                shown = False
                print 'rels_predicted: ', rels_predicted, gt_rels

            step += 1

            for k, v in result.items():
                results[k].append(v)
                results_random[k].append(result_random[k])

        logname = 'lixs_reranked' if bm25_reranked else 'lixs_no_reranked'

        filename = logname + '.txt'
        self.print_ntcir_result(filename,logname, qid_list, did_list, pred_list)

        performances = {}
        performances_random = {}
        # print results['map'],np.mean(results['map'])
        for k, v in results.items():
            performances[k] = np.mean(v)
            performances_random[k] = np.mean(results_random[k])

        ndcg_1 /= step
        is_best = False
        if ndcg_1 > best_result:
            is_best = True
            best_result = ndcg_1

        print '------Source: NTCIR \tPerformance-------:'
        print 'Message: %s' % self.message
        print performances
        print '---------Random-----------'
        print performances_random

        if save:
            # Save the model
            self.save_checkpoint({
                'epoch': self.epoch + 1,
                'step': self.step + 1,
                'timings': self.timings,
                'predicted_rel': predicted,
                'performances': performances,
                'model_state_dict': self.model.state_dict(),
                'best_result': best_result,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

        return performances

    def show_example(self,query_batch,doc_batch,sa_matrix,fa_matrix,sp_matrix,pf_matrix,select=0):

        query = [self.data_generator.id2word[wid] for wid in query_batch[select] if wid != 0]
        doc = []


        for sent in doc_batch[select]:
            sen_id = []
            for wid in sent:
                if wid != 0:
                    sen_id.append(self.data_generator.id2word[wid])
            if len(sen_id) > 0:
                doc.append(sen_id)

        doc_length = len(doc)
        print 'Query: ', ' '.join(query)
        print 'Document: '
        for sent in doc:
            print ' '.join(sent)

        #print doc_batch[select][:doc_length]

        print 'Select: ', sa_matrix[select][:doc_length]
        print '-----------------------------------------'
        print 'Finish: ', fa_matrix[select][:doc_length]
        print '-----------------------------------------'
        print 'select p:', sp_matrix[select][:doc_length]
        print '-----------------------------------------'
        print 'Finish p:', pf_matrix[select][:doc_length]

    def train_pointwise(self,query_batch,query_lengths,doc_batch,doc_lengths,labels_batch):
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        out = self.model(query_batch, query_lengths, doc_batch, doc_lengths,labels_batch)

        _ ,mse = out

        reg_loss = torch.sum(mse)
        reg_loss.backward()
        self.optimizer.step()

        return reg_loss.data, torch.sum(mse).data


    def train(self,query_batch,query_lengths,doc_pos_batch,doc_neg_batch,doc_pos_length,doc_neg_length):
        # Turn on training mode which enables dropout.
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        out_pos = self.model(query_batch,query_lengths,doc_pos_batch,doc_pos_length)
        out_neg = self.model(query_batch, query_lengths, doc_neg_batch,doc_neg_length)

        Erel_pos, _ = out_pos
        Erel_neg, _ = out_neg
        #rl_loss = torch.sum( torch.clamp(1.0 - Erel_pos + Erel_neg,min=0) + reward_s_pos + reward_s_neg)#scalar

        rl_loss = torch.sum( torch.clamp(1.0 - Erel_pos + Erel_neg,min=0))
        #rl_loss = torch.sum(- Erel_pos * torch.log(Erel_pos) - (1.0 - Erel_neg) * torch.log(1.0 - Erel_neg))
        #rl_loss = torch.sum(- Erel_pos + Erel_neg)

        #rl_loss /= self.train_freq
        rl_loss.backward()

        ## update parameters
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        return rl_loss.data


    def validation(self,query_batch,query_lengths,doc_pos_batch,doc_neg_batch,doc_pos_length,doc_neg_length):
        # Turn on evaluation mode which disables dropout.
        with torch.no_grad():
            self.model.eval()
            E_pos, _ = self.model(query_batch, query_lengths, doc_pos_batch,doc_pos_length)
            E_neg, _ = self.model(query_batch, query_lengths, doc_neg_batch,doc_neg_length)

            #rl_loss = torch.sum(torch.clamp(1.0 - E_pos + E_neg,min=0) + reward_s_pos + reward_s_neg)  # scalar
            #rl_loss = torch.sum(- E_pos + E_neg)
            rl_loss = torch.sum(torch.clamp(1.0 - E_pos + E_neg, min=0))
            #print 'val rl_loss: ',rl_loss

        return rl_loss

    def predict(self,query_batch,query_lengths,doc_batch,doc_lengths,gt_rels):
        # Turn on evaluation mode which disables dropout.
        with torch.no_grad():
            self.model.eval()
            if self.pairwise:
                rels_predicted,mse = \
                    self.model(query_batch, query_lengths, doc_batch, doc_lengths, gt_rels)
                #print rels_predicted
            else:
                rels_predicted,mse = \
                    self.model(query_batch, query_lengths, doc_batch, doc_lengths, gt_rels)

        #print
        return rels_predicted,mse #rels_predicted


    def load_pair_file(self,path):
        query_dict = cPickle.load(open(path[0]))
        doc_dict = cPickle.load(open(path[1]))
        pair_file = cPickle.load(open(path[2]))
        return query_dict,doc_dict,pair_file

    def load_test_file(self,path):
        query_dict = cPickle.load(open(path[0]))
        doc_dict = cPickle.load(open(path[1]))
        qrel_file = cPickle.load(open(path[2]))
        return query_dict, doc_dict, qrel_file

    def save_checkpoint(self,state, is_best, filename='./model/checkpoint.pth.tar'):
        torch.save(state, filename)
        #if is_best:
        #    shutil.copyfile(filename, './model/model_best.pth.tar')


def main(args,config):
    train_model = TrainingModel(args,config)

    if not args.eval:
        if config['pairwise']:
            train_model.trainIters()
        else:
            train_model.trainItersPointWise()

    save = True
    if args.eval:
        print 'only evaluation'
        save = False

    #train_model.test_for_ntcir(bm25_reranked=True,reranked_topK=40)
    #train_model.test_for_ntcir(bm25_reranked=False, reranked_topK=40)



    topK = [40]
    for k in topK:
        print 'Top K: %d' % k
        # train_model.test(source='UBM', save=True, bm25_reranked=True, reranked_topK=k)
        #train_model.test(source='ntcir13', save=True, bm25_reranked=True, reranked_topK=k)
        train_model.test(source='ntcir14', save=True, bm25_reranked=True, reranked_topK=20)
        #train_model.test(source='PSCM', save=True, bm25_reranked=True, reranked_topK=k)
        # train_model.test(source='TACM', save=True, bm25_reranked=True, reranked_topK=k)
        # train_model.test(source='HUMAN', save=True, bm25_reranked=True, reranked_topK=k)

        #train_model.test(source='human', save=True, bm25_reranked=True, reranked_topK=k)
        #train_model.test(source='human', save=False, bm25_reranked=False, reranked_topK=k)
        #train_model.test(source='ntcir', save=False, bm25_reranked=True, reranked_topK=k)
        #train_model.test(source='ntcir', save=False, bm25_reranked=False, reranked_topK=k)

    #train_model.test(source='ntcir',save=save,bm25_reranked=True,reranked_topK=10)
    #train_model.test(source='ntcir', save=save, bm25_reranked=False, reranked_topK=10)

if __name__ == '__main__':
    args = load_arguments()
    config_state = eval(args.prototype)()

    main(args, config_state)
