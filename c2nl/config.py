""" Implementation of all available options """
from __future__ import print_function


import argparse
import logging

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type',
    'emsize',
    'nlayers',
    'ngatlayers',
    'use_all_enc_layers',
    'src_pos_emb',
    'tgt_pos_emb',
    'd_ff',
    'd_k',
    'd_v',
    'num_head',
    'trans_drop',
    'use_rpe',
    'rpe_mode',
    'rpe_size',
    'rpe_layer',
    'rpe_share_emb',
    'rpe_m',
    'rpe_c',
    'rpe_approx',
    'rpe_all',
}

SEQ2SEQ_ARCHITECTURE = {
    'attn_type',
    'force_copy',
    'copy_attn',
    'layer_wise_attn',
    'reload_decoder_state',
    'share_decoder_embeddings',
}

DATA_OPTIONS = {
    'use_tgt_char',
    'use_tgt_word',
    'max_src_len',
    'max_tgt_len',
    'src_vocab_size',
    'tgt_vocab_size',
    'num_train_examples',
    'batch_size',
    'use_word_type',
    'use_word_fc',
    'use_dense_connection',
    'add_top_down_edges',
    'add_bottom_up_edges',
    'uncase',
    'dataset_weights'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'optimizer',
    'fix_embeddings',
    'learning_rate',
    'momentum',
    'weight_decay',
    'dropout',
    'dropout_emb',
    'cuda',
    'grad_clipping',
    'lr_decay',
    'warmup_steps',
    'num_epochs',
    'parallel'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Data options
    data = parser.add_argument_group('Data parameters')
    data.add_argument('--max_src_len', type=int, default=150,
                      help='Maximum allowed length for the source sequence')
    data.add_argument('--max_tgt_len', type=int, default=50,
                      help='Maximum allowed length for the target sequence')
    data.add_argument('--use_word_type', type='bool', default=False,
                      help='Use word type as additional feature for feature representations')
    data.add_argument('--use_word_fc', type='bool', default=False,
                      help='Let words fully connected when building graph')
    data.add_argument('--use_dense_connection', type='bool', default=False,
                      help='Use dense or sparse connection style when building AST-Code graph')
    data.add_argument('--add_top_down_edges', type='bool', default=True,
                      help='Add top down AST edges when building AST-Code graph')
    data.add_argument('--add_bottom_up_edges', type='bool', default=True,
                      help='Add bottom up AST edges when building AST-Code graph')
    
    # Model architecture
    model = parser.add_argument_group('Summary Generator')
    model.add_argument('--model_type', type=str, default='gntransformer',
                       choices=['gntransformer'],
                       help='Model architecture type')
    model.add_argument('--emsize', type=int, default=512,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--nlayers', type=int, default=3,
                       help='Number of encoding and decoding layers')
    model.add_argument('--ngatlayers', type=int, default=1,
                       help='Number of GAT layers')
    model.add_argument('--use_all_enc_layers', type='bool', default=False,
                       help='Use a weighted average of all encoder layers\' '
                            'representation as the contextual representation')

    # Transformer specific params
    model.add_argument('--src_pos_emb', type='bool', default=True,
                       help='Use positional embeddings in encoder')
    model.add_argument('--tgt_pos_emb', type='bool', default=True,
                       help='Use positional embeddings in decoder')
    model.add_argument('--d_ff', type=int, default=2048,
                       help='Number of units in position-wise FFNN')
    model.add_argument('--d_k', type=int, default=64,
                       help='Hidden size of heads in multi-head attention')
    model.add_argument('--d_v', type=int, default=64,
                       help='Hidden size of heads in multi-head attention')
    model.add_argument('--num_head', type=int, default=8,
                       help='Number of heads in Multi-Head Attention')
    model.add_argument('--trans_drop', type=float, default=0.2,
                       help='Dropout for transformer')
    model.add_argument('--use_rpe', type='bool', default=False,
                       help='Whether use RPE')
    model.add_argument('--rpe_mode', type=str, default='sum',
                       help='Mode of RPE, sum or concat')
    model.add_argument('--rpe_size', type=int, default=512,
                       help='Size of RPE when concat')
    model.add_argument('--rpe_layer', type=int, default=2,
                       help='Layer num of PGNN')
    model.add_argument('--rpe_share_emb', type='bool', default=False,
                       help='Share pos emb and emb')
    model.add_argument('--rpe_m', type=int, default=6,
                       help='m in PGNN anchor')
    model.add_argument('--rpe_c', type=float, default=0.2,
                       help='c in PGNN anchor, copy=int(m*c)')
    model.add_argument('--rpe_approx', type=int, default=0,
                       help='approximate in PGNN anchor')
    model.add_argument('--rpe_all', type='bool', default=True,
                       help='use all nodes in RPE')
    model.add_argument('--layer_wise_attn', type='bool', default=False,
                       help='Use layer-wise attention in Transformer')

    # Input representation specific details
    model.add_argument('--use_tgt_char', type='bool', default=False,
                       help='Use character embedding in the target')
    model.add_argument('--use_tgt_word', type='bool', default=True,
                       help='Use word embedding in the input')

    seq2seq = parser.add_argument_group('Seq2seq Model Specific Params')
    seq2seq.add_argument('--copy_attn', type='bool', default=False,
                         help='Use copy attention')
    seq2seq.add_argument('--attn_type', type=str, default='general',
                         help='Attention type for the seq2seq [dot, general, mlp]')
    seq2seq.add_argument('--force_copy', type='bool', default=False,
                         help='Apply force copying')
    seq2seq.add_argument('--share_decoder_embeddings', type='bool', default=True,
                         help='Share decoder embeddings weight with softmax layer')
    seq2seq.add_argument('--reload_decoder_state', type=str, default=None,
                         help='Reload decoder states for the seq2seq')

    # Optimization details
    optim = parser.add_argument_group('Neural QA Reader Optimization')
    optim.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'adamW'],
                       help='Name of the optimizer')
    optim.add_argument('--dropout_emb', type=float, default=0.2,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout for NN layers')
    optim.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate for the optimizer')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Decay ratio for learning rate')
    optim.add_argument('--grad_clipping', type=float, default=5.0,
                       help='Gradient clipping')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='Stop training if performance doesn\'t improve')
    optim.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix_embeddings', type='bool', default=False,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--warmup_steps', type=int, default=2000,
                       help='Number of of warmup steps')
    optim.add_argument('--warmup_epochs', type=int, default=0,
                       help='Number of of warmup steps')


def get_model_args(args):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER, SEQ2SEQ_ARCHITECTURE, DATA_OPTIONS
    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER | SEQ2SEQ_ARCHITECTURE | DATA_OPTIONS

    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))

    return argparse.Namespace(**old_args)


def add_new_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global ADVANCED_OPTIONS
    old_args, new_args = vars(old_args), vars(new_args)
    for k in new_args.keys():
        if k not in old_args:
            if k in ADVANCED_OPTIONS:
                logger.info('Adding arg %s: %s' % (k, new_args[k]))
                old_args[k] = new_args[k]

    return argparse.Namespace(**old_args)
