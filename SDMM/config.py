import argparse


def load_arguments():
    parser = argparse.ArgumentParser(description='Evidence Reading Model')

    parser.add_argument('--resume', type=str, default="",
                        help='Resume training from that state')
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='basic_config')
    parser.add_argument("--eval", action="store_true", help="only evaluation")
    parser.add_argument('--visdom', action="store_true",
                        help='Use visdom for loss visualization')
    parser.add_argument('--m', type=str, default="",
                        help='Message in visdom window (Flag)')
    parser.add_argument('--gpu', type=int, default=1,
                        help="# of GPU running on")
    parser.add_argument('--save', type=str, default="checkpoint.pth.tar",
                        help='save filename for training model')
    parser.add_argument('--learn', type=str, default="pair",
                        help='Pointwise learning')
    parser.add_argument('--fold', type=int, default=1,
                        help="# of data fold")
    parser.add_argument('--visual', action="store_true",
                        help='Visualize the example case')
    args = parser.parse_args()

    return args


def basic_config():
    state = {}
    state['min_score_diff'] = 0.25
    state['dataName'] = 'QCL'

    state['pairwise'] = True
    state['click_model'] = 'PSCM'

    state['data_addr'] = 'Train address'
    state['valid_addr'] = 'validation address'
    state['vocab_dict_file'] = 'vocab address from convert2textdict.py'
    state['human_label_addr'] = ''
    state['ntcir13_test'] = 'ntcir13_test_bm25_top100_4level.txt'
    state['ntcir14_test'] = 'ntcir14_test_bm25_4_level.txt'

    state['lab_data_addr'] = '../lab_data/lab_data.txt'
    state['emb'] = 'embedding address'

    state['drate'] = 0.8
    state['seed'] = 1234

    state['batch_size'] = 1#80
    state['epochs'] = 100
    state['lr'] = 0.1#0.005
    state['weight_decay'] = 0#1e-3
    state['clip_grad'] = 0.5
    state['optim'] = 'adadelta'  # 'sgd, adadelta' adadelta0.1, adam0.005

    state['size_filter'] = [3, 5, 8]
    state['n_filter'] = 8
    state['n_repeat'] = 5  # 5 if state['pairwise'] else 1

    state['mask_id'] = 0

    state['embsize'] = 50
    state['term_hidden_size'] = 128/2
    state['query_hidden_size'] = 128/2  # same as term

    state['evidence_hidden_size'] = 200/2
    state['position_hidden_size'] = 3

    state['cost_threshold'] = 1.003
    state['patience'] = 5
    state['train_freq'] = 50  # 200
    state['eval_freq'] = 100  # 5000
    state['value_loss_coef'] = 0.5

    state['maxRelevance'] = 2
    state['prate'] = 0.2  # exploration applied to layers (0 = noexploration)

    state['sentence_hidden_size'] = len(state['size_filter']) * state['n_filter']

    state['num_selector_class'] = 2

    return state

def baseline_config():
    state = basic_config()
    state['embsize'] = 50

    state['size_filter'] = [2, 3, 4, 5]
    state['n_filter'] = 8

    state['term_hidden_size'] = 128
    state['query_hidden_size'] = 128  # same as term
    state['sentence_hidden_size'] = 128#len(state['size_filter']) * state['n_filter']

    state['evidence_hidden_size'] = 128

    state['batch_size'] = 80
    state['position_hidden_size'] = 3
    #state['sentence_hidden_size'] = 11 # for K-NRM

    state['order'] = 'sequential'#sequential
    state['baseline_type'] = 'independent' #,independent(default setting for SDMM),accmulated, 
    state['vertical_decay'] = True

    return state

def small_config():
    state = basic_config()
    state['embsize'] = 50

    state['size_filter'] = [2, 3,4, 5]
    state['n_filter'] = 8

    state['term_hidden_size'] = 128
    state['query_hidden_size'] = 128  # same as term
    state['sentence_hidden_size'] = 128#len(state['size_filter']) * state['n_filter']

    state['evidence_hidden_size'] = 128

    state['batch_size'] = 80
    state['position_hidden_size'] = 3
    #state['sentence_hidden_size'] = 11 # for K-NRM
    return state