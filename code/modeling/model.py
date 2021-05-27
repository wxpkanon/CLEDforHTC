#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Create: 2020/12/01
#
import os

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from .layers import bidirectional_rnn, attention, attention_class, attention_pre
from .utils import get_shape, batch_doc_normalize
import numpy as np
import random


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  print('ADDING layer normalization')
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def _squash(input_tensor):
    """Applies norm nonlinearity (squash) to a capsule layer.

  Args:
    input_tensor: Input tensor. Shape is [num_channels, num_atoms]

  Returns:
    A tensor with same shape as input (rank 3) for output of this layer.
  """
    with tf.name_scope('norm_non_linearity'):
        norm = tf.norm(input_tensor, axis=1, keep_dims=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

def _leaky_routing(logits, output_dim):
    """Adds extra dimmension to routing logits.

  This enables active capsules to be routed to the extra dim if they are not a
  good fit for any of the capsules in layer above.

  Args:
    logits: The original logits. shape is
      [input_capsule_num, output_capsule_num] if fully connected. Otherwise, it
      has two more dimmensions.
    output_dim: The number of units in the second dimmension of logits.

  Returns:
    Routing probabilities for each pair of capsules. Same shape as logits.
  """

    # leak is a zero matrix with same shape as logits except dim(2) = 1 because
    # of the reduce_sum.
    leak = tf.zeros_like(logits, optimize=True)
    leak = tf.reduce_sum(leak, axis=1, keep_dims=True)
    leaky_logits = tf.concat([leak, logits], axis=1)
    leaky_routing = tf.nn.softmax(leaky_logits, dim=1)
    return tf.split(leaky_routing, [1, output_dim], 1)[1]

def _slice_softmax(bij, index_list, cluster_num):
    print('cluster num is %d' % cluster_num)
    c = []
    for i in range(len(index_list)):
        for j in range(cluster_num):
            start = sum(index_list[0:i])
            left = tf.constant([-1e15]*start, dtype=tf.float32)
            mid = tf.constant([0.]*index_list[i], dtype=tf.float32)
            right = tf.constant([-1e15]*sum(index_list[i+1:]), dtype=tf.float32)
            mask = tf.concat([left, mid, right], -1)
            b = tf.slice(bij, [cluster_num*i + j, 0], [1, -1]) + mask
            c.append(tf.nn.softmax(b, axis=1))
    return tf.concat(c, 0)

def _update_routing(votes, target_cat_emb, biases, logit_shape, num_dims, input_dim, output_dim,
                    num_routing, leaky, cluster_num, tree_idx_list=None):
    """Sums over scaled votes and applies squash to compute the activations.

  Iteratively updates routing logits (scales) based on the similarity between
  the activation of this layer and the votes of the layer below.

  Args:
    votes: tensor, The transformed outputs of the layer below.
               votes shape [input_dim, output_dim, output atoms]
    biases: tensor, Bias variable.
    logit_shape: tensor, shape of the logit to be initialized.
    num_dims: scalar, number of dimmensions in votes. For fully connected
      capsule it is 4, for convolutional 6.
    input_dim: scalar, number of capsules in the input layer.
    output_dim: scalar, number of capsules in the output layer.
    num_routing: scalar, Number of routing iterations.
    leaky: boolean, if set use leaky routing.

  Returns:
    The activation tensor of the output layer after num_routing iterations.
  """
    votes_t_shape = [2, 0, 1]
    r_t_shape = [1, 2, 0]
    votes_trans = tf.transpose(votes, votes_t_shape)  # votes_trans axis is [2,0,1]

    def _body(i, logits, activations, routes):
        act_3d = tf.cond(i < 1, lambda: tf.expand_dims(target_cat_emb, 0), lambda: tf.expand_dims(activations.read(i-1), 0))
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()  # [1, 1, 1]                   
        tile_shape[0] = input_dim  # [input_dim, 1, 1]                                         
        act_replicated = tf.tile(act_3d, tile_shape)  # [input_dim, output_dim, output_atoms]  
        distances = tf.reduce_sum(votes * act_replicated, axis=-1)  # [input_dim, output_dim]  
        logits += distances                                                                         
        route = _slice_softmax(logits, index_list=tree_idx_list, cluster_num=cluster_num)                                                              
        routes = routes.write(i, route)                                                             
        preactivate_unrolled = route * votes_trans  # route is cij  [input_dim, output_dim, ...][output_atoms, input_dim, output_dim]
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)  # [input_dim, output_dim, output_atoms]
        preactivate = tf.reduce_sum(preact_trans, axis=0) + biases  # [output_dim, output_atoms]
        activation = _squash(preactivate)  # [output_dim, output_atoms]                        
        activations = activations.write(i, activation)                                              
        return (i + 1, logits, activations, routes)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    routes = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations, routes = tf.while_loop(
        lambda i, logits, activations, routes: i < num_routing,
        _body,
        loop_vars=[i, logits, activations, routes],
        swap_memory=True)

    return activations.read(num_routing - 1), routes.read(num_routing - 1)

def _keywords_dynamic_routingto_label(var_keywords_embed, keywords_total_nums, var_label_embed, label_nums, embed_size,
                                      cluster_num, num_routing=3, scope=None, tree_idx_list=None):
    with tf.variable_scope(scope or 'keywords_dynamic_routing_tolabel'):
        # votes shape [input_dim, output_dim, output_atoms] i.e. [c1keywords_nums, c2_nums, word_embed_dim]
        c1keyword_3d = tf.expand_dims(var_keywords_embed, axis=1)  # [c1keywords_nums, 1, word_embed_dim]
        c1keyword_votes = tf.tile(c1keyword_3d, [1, label_nums, 1])  # [c1keywords_nums, c2_nums, word_embed_dim]
        c1keytoc2_biases = tf.get_variable(name='keytolabel_dynamic_route_biases',
                                           shape=[label_nums, embed_size],
                                           initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        c1keytoc2_logit_shape = [keywords_total_nums, label_nums]

        # shape [output_dim, output_atoms] i.e. [c2_nums, word_embed_dim]
        c2emb_routedby_c1keywords, rout = _update_routing(votes=c1keyword_votes,
                                                          target_cat_emb=var_label_embed,
                                                          biases=c1keytoc2_biases,
                                                          logit_shape=c1keytoc2_logit_shape,
                                                          num_dims=3, input_dim=keywords_total_nums,
                                                          output_dim=label_nums, num_routing=num_routing,
                                                          leaky=False,
                                                          tree_idx_list=tree_idx_list,
                                                          cluster_num=cluster_num)
        return c2emb_routedby_c1keywords, rout


def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  print ('use gelu')
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def get_shape_list(tensor):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

class CLED(object):

    def __init__(self, embedding_size, vocab_size, sentence_len, labels_manager_dict,
                 pretrain_embedding=None, filter_num=128, max_model_to_save=5,
                 summaries_dir=None, use_posi_embedding=True, init_lr=1e-3,
                 lr_decay_step=1000, lr_decay_rate=0.95, is_train=False,
                 label_embedding_dict=None, cell_dim=150, c1keywords_embed=None, c2keywords_embed=None,
                 last_dim_theta=None, cluster_num=None):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.sentence_len = sentence_len

        self.c1_nums = labels_manager_dict['c1'].class_num
        self.c2_nums = labels_manager_dict['c2'].class_num
        self.c3_nums = labels_manager_dict['c3'].class_num

        self.pretrain_embedding = pretrain_embedding
        self.train_embedding = False
        self.filter_num = filter_num
        self.filter_sizes = [2, 3]
        self.max_model_to_save = max_model_to_save
        self.summaries_dir = summaries_dir
        self.norm_initializer = tf.random_normal_initializer(stddev=0.1)
        self.xavier_init = tf.contrib.layers.xavier_initializer()
        self.xavier_conv2d_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.init_lr = init_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.is_train = is_train
        self.use_bn = False

        self.label_embedding_c1 = label_embedding_dict['kb_c1']
        self.label_embedding_c2 = label_embedding_dict['kb_c2']
        self.label_embedding_c3 = label_embedding_dict['kb_c3']
        self.cats_nums_dict = {'kb_c1': self.c1_nums, 'kb_c2': self.c2_nums, 'kb_c3': self.c3_nums}

        self.l2_reg = tf.contrib.layers.l2_regularizer(0.001)
        self.cell_dim = cell_dim

        self.c1keywords_embed = c1keywords_embed
        self.c1keywords_total_nums = np.shape(c1keywords_embed)[0]

        self.c2keywords_embed = c2keywords_embed
        self.c2keywords_total_nums = np.shape(c2keywords_embed)[0]

        self.c1toc2_tree_list = [30, 1, 6, 16, 5, 2, 1, 1, 8]  # length9 item70
        self.c2toc3_tree_list = [2, 5, 27, 1, 1, 3, 4, 1, 1, 8,
                                 3, 2, 1, 1, 1, 1, 5, 1, 19, 7,
                                 1, 2, 2, 5, 1, 8, 1, 5, 1, 3,
                                 1, 2, 1, 2, 5, 4, 4, 1, 1, 8,
                                 2, 1, 2, 6, 1, 4, 1, 2, 3, 1,
                                 2, 1, 1, 8, 1, 1, 1, 6, 1, 3,
                                 1, 1, 2, 2, 1, 5, 3, 1, 1, 2]  # length70 item219

        self.pred_vec_dim_theta = last_dim_theta
        self.cluster_num = cluster_num

    def build(self, batch_x, batch_y1, batch_y2, batch_y3, batch_x_mask, sentence_lens):
        self.input_x = batch_x
        self.input_y1 = batch_y1
        self.input_y2 = batch_y2
        self.input_y3 = batch_y3
        self.input_x_mask = batch_x_mask
        self.sent_lengths = sentence_lens

        self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')

        self.add_embedding_op()
        self.add_label_embedding_op()
        self.add_keywords_embedding_op()  # keywords for dynamic routing

        self.add_cnn_layers_op()
        self.add_word_rnn_layers_op()

        self.add_c1label_word_att_op()
        self.add_c2label_word_att_op()
        self.add_c3label_word_att_op()

        # C1 PRED
        self.add_c1_prediction_op()  # add_word_prediction_op
        self.add_pred_c1_att_op()  # add_sentence_layers_op

        # C2 PRED
        # dynamic routing
        self.add_dynamic_routing_c1keywordstoc2_op()  # dynamic rout
        self.add_c1keywordsrouted_c2label_word_att_op()  # dynamic rout

        self.add_c2_prediction_op()  # add_prediction_op
        self.add_pred_c2_att_op()

        self.add_c2_label_embed_loss_op()

        # C3 PRED
        # dynamic routing
        self.add_dynamic_routing_c2keywordstoc3_op()
        self.add_c2keywordsrouted_c3label_word_att_op()
        self.add_c3_prediction_op()
        self.add_c3_label_embed_loss_op()

        # LOSS
        self.add_loss_op()
        self.add_train_op()

    def add_embedding_op(self):
        with tf.name_scope('embedding'):
            if self.pretrain_embedding is None:
                _word_embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size,
                                       self.embedding_size],
                                      -1.0, 1.0),
                    name='embeddings_table',
                    dtype=tf.float32,
                    trainable=True)
            else:
                _word_embeddings = tf.Variable(
                    self.pretrain_embedding,
                    name='embeddings_table',
                    dtype=tf.float32,
                    trainable=self.train_embedding)
                print ('add pretrained word embedding operator trainable is %s' % str(self.train_embedding))

            joint_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                      self.input_x)
            # [B, seq, dim]
            self.docs_words_embeddings = joint_embeddings

    def add_label_embedding_op(self):
        print ('now adding label embedding operator')
        assert (self.label_embedding_c1 is not None), 'c1 label embedding must not be none!!!'
        assert (self.label_embedding_c2 is not None), 'c2 label embedding must not be none!!!'

        print ('now adding label embedding operator')
        self.label_embed_var_dict = {}
        with tf.name_scope('label_embedding'):
            # [label_num, word_dims]
            # cat_1
            self.var_label_embeddings_c1 = tf.Variable(
                self.label_embedding_c1, name='label_embeddings_table_kb_c1', dtype=tf.float32, trainable=True)
            self.label_embed_var_dict['kb_c1'] = self.var_label_embeddings_c1

            # cat_2
            self.var_label_embeddings_c2 = tf.Variable(
                self.label_embedding_c2, name='label_embeddings_table_kb_c2', dtype=tf.float32, trainable=True)
            self.label_embed_var_dict['kb_c2'] = self.var_label_embeddings_c2

            # cat_3
            self.var_label_embeddings_c3 = tf.Variable(
                self.label_embedding_c3, name='label_embeddings_table_kb_c3', dtype=tf.float32, trainable=True)
            self.label_embed_var_dict['kb_c3'] = self.var_label_embeddings_c3

    def add_keywords_embedding_op(self):
        print('now adding keywords embedding operator')
        assert (self.c1keywords_embed is not None), 'c1 keywords embedding must not be none!!!'
        with tf.name_scope('cat_keywords_embedding'):
            # cat_1_keywords  shape: [num_keywords_percat * num_cat, word_dim]
            self.var_keywords_embeddings_c1 = tf.Variable(
                self.c1keywords_embed, name='keywords_embed_table_c1', dtype=tf.float32, trainable=True)
            # cat_2_keywords  shape: [num_keywords_percat * num_cat, word_dim]
            self.var_keywords_embeddings_c2 = tf.Variable(
                self.c2keywords_embed, name='keywords_embed_table_c2', dtype=tf.float32, trainable=True)

    def add_cnn_layers_op(self):
        # [B, seq, dim]
        word_inputs = self.docs_words_embeddings
        pooled_outputs = []
        for filter_size in self.filter_sizes:
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, self.filter_num]
                weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                bias = tf.Variable(tf.constant(0.1, shape=[self.filter_num]), name='b')
                conv = tf.nn.conv1d(word_inputs,
                                    weight,
                                    stride=1,
                                    padding='SAME',
                                    name='conv')

                # Apply nonlinearity
                if self.use_bn:
                    before_h = tf.layers.batch_normalization(tf.nn.bias_add(conv, bias),
                                                             training=self.bn_training,
                                                             name='bn-%s' % filter_size)
                else:
                    before_h = tf.nn.bias_add(conv, bias)
                relu_h = tf.nn.relu(before_h, name='relu')

                pooled_outputs.append(relu_h)

        self.num_filters_total = self.filter_num * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=-1)
        # [B, seq, filter_num * len_filter_size]
        with tf.name_scope('dropout'):
            self.h_dropout = tf.nn.dropout(h_pool, self.dropout_keep)

    def add_word_rnn_layers_op(self):
        with tf.variable_scope('word-encoder') as scope:
            self.sent_lengths = tf.reshape(self.sent_lengths, [-1])

            # word encoder
            cell_fw = rnn.GRUCell(self.cell_dim, name='cell_fw')
            cell_bw = rnn.GRUCell(self.cell_dim, name='cell_bw')
            init_state_fw = tf.tile(tf.get_variable('init_state_fw',
                                                    shape=[1, self.cell_dim],
                                                    initializer=tf.constant_initializer(0)),
                                                    multiples=[get_shape(self.h_dropout)[0], 1])
            init_state_bw = tf.tile(tf.get_variable('init_state_bw',
                                                    shape=[1, self.cell_dim],
                                                    initializer=tf.constant_initializer(0)),
                                                    multiples=[get_shape(self.h_dropout)[0], 1])
            # [B, Seq, cell_dim*2]
            self.rnn_outputs, _ = bidirectional_rnn(cell_fw=cell_fw,
                                               cell_bw=cell_bw,
                                               inputs=self.h_dropout,
                                               input_lengths=self.sent_lengths,
                                               initial_state_fw=init_state_fw,
                                               initial_state_bw=init_state_bw,
                                               scope=scope)

    def add_c1label_word_att_op(self):
        input_x_mask = tf.reshape(self.input_x_mask, [-1, self.sentence_len])
        # [B, e] e=cell_dim*2
        word_outputs, self.word_att_weights = attention_class(inputs=self.rnn_outputs,
                                                   class_embedding=self.label_embed_var_dict['kb_c1'],
                                                   input_mask=input_x_mask,
                                                   sequence_lengths=self.sent_lengths,
                                                   class_num=self.c1_nums, class_embedding_dim=self.embedding_size,
                                                              scope='kb_c1_wordatt')
        with tf.name_scope('c1_dropout'):
            print('ADDING c1 layer normalization')
            word_outputs_ln = layer_norm(word_outputs)
            self.word_outputs = tf.nn.dropout(word_outputs_ln, self.dropout_keep)

    def add_c2label_word_att_op(self):
        print('ADD c2label_word_att_op')
        input_x_mask = tf.reshape(self.input_x_mask, [-1, self.sentence_len])
        # [B, e] e=cell_dim*2
        c2_word_outputs, self.c2_word_att_weights = attention_class(inputs=self.rnn_outputs,
                                                   class_embedding=self.label_embed_var_dict['kb_c2'],
                                                   input_mask=input_x_mask,
                                                   sequence_lengths=self.sent_lengths,
                                                   class_num=self.c2_nums, class_embedding_dim=self.embedding_size,
                                                                    scope='kb_c2_wordatt')
        self.c2_word_outputs = c2_word_outputs

    def add_c3label_word_att_op(self):
        print('ADD_c3label_word_att_op')
        input_x_mask = tf.reshape(self.input_x_mask, [-1, self.sentence_len])
        # [B, e] e=cell_dim*2
        c3_word_outputs, self.c3_word_att_weights = attention_class(inputs=self.rnn_outputs,
                                                                    class_embedding=self.label_embed_var_dict['kb_c3'],
                                                                    input_mask=input_x_mask,
                                                                    sequence_lengths=self.sent_lengths,
                                                                    class_num=self.c3_nums,
                                                                    class_embedding_dim=self.embedding_size,
                                                                    scope='kb_c3_wordatt')
        self.c3_word_outputs = c3_word_outputs

    def add_c1_prediction_op(self):
        with tf.name_scope('output_c1'):
            # [B, e]
            word_outputs = self.word_outputs
            weight = tf.get_variable('W_projection_c1',
                                     shape=[2*self.cell_dim, self.c1_nums],
                                     initializer=self.norm_initializer)
            bias = tf.Variable(tf.constant(0.1, shape=[self.c1_nums]), name='b_c1')
            self.l2_loss_word = tf.add_n([tf.nn.l2_loss(weight), tf.nn.l2_loss(bias)])
            #   [B, class_num] = [B, e] [e, class_num]
            self.scores_word = tf.nn.xw_plus_b(word_outputs, weight, bias, name='scores_c1')

            self.predictions_word = tf.nn.softmax(self.scores_word)
            # [B, class_num, 1]
            scores_word = tf.expand_dims(self.predictions_word, -1)
            # [B, class_num, e] = [class_num, e] [B, class_num, 1]
            pre_att = tf.multiply(self.label_embed_var_dict['kb_c1'], scores_word)
            # [B, e]
            self.pre_att = tf.reduce_sum(pre_att, 1)

    def add_pred_c1_att_op(self):
        with tf.variable_scope('pred_c1_att_op') as scope:
            # [B,e]  e=cell_dim*2
            self.sent_outputs, sent_att_weights = attention_pre(inputs=self.rnn_outputs,
                                                      sequence_lengths=self.sent_lengths,
                                                      pre_att=self.pre_att,
                                                                scope='pred_c1_att')

    # dynamic routing
    def add_dynamic_routing_c1keywordstoc2_op(self):
        print('ADDING c1 keywords routed to c2 operator')
        # shape [output_dim, output_atoms] i.e. [c2_nums, word_embed_dim]
        self.c2emb_routedby_c1keywords, self.rout_c12 = _keywords_dynamic_routingto_label(
                                                        var_keywords_embed=self.var_keywords_embeddings_c1,
                                                        keywords_total_nums=self.c1keywords_total_nums,
                                                        var_label_embed=self.var_label_embeddings_c2,
                                                        label_nums=self.c2_nums, embed_size=self.embedding_size,
                                                        num_routing=3, scope='c1keywords_dynamic_routing_toc2',
                                                        tree_idx_list=self.c1toc2_tree_list, cluster_num=self.cluster_num)

    # keywords dynamic routed c2 att word
    def add_c1keywordsrouted_c2label_word_att_op(self):
        print('ADDING c1keywords routed c2label attention word op')
        input_x_mask = tf.reshape(self.input_x_mask, [-1, self.sentence_len])
        # [B, e] e=cell_dim*2
        c2routed_word_outputs, self.c2routed_word_att_weights = attention_class(inputs=self.rnn_outputs,
                                                            class_embedding=self.c2emb_routedby_c1keywords,
                                                            input_mask=input_x_mask, sequence_lengths=self.sent_lengths,
                                                            class_num=self.c2_nums,
                                                            class_embedding_dim=self.embedding_size,
                                                            scope='routed_c2_wordatt')
        self.c2routed_word_outputs = c2routed_word_outputs

    # with routed_c2_word_att
    def add_c2_prediction_op(self):
        with tf.name_scope('output_c2'):
            # [B,e*3] = [B,e] [B,e] [B,e] e=cell_dim*2
            c1c2word_concat = tf.concat([self.sent_outputs, self.c2_word_outputs, self.c2routed_word_outputs], axis=-1)
            print('ADDING layer normalization before dropout')
            c1c2word_concat_ln = layer_norm(c1c2word_concat)
            c1c2word_concat_ln_drop = tf.nn.dropout(c1c2word_concat_ln, self.dropout_keep)
            print('the input Embeddings of c2 softmax dim is %d' % (self.pred_vec_dim_theta*self.cell_dim))
            c2_hidden_inputs = tf.layers.dense(inputs=c1c2word_concat_ln_drop,
                                              units=self.pred_vec_dim_theta*self.cell_dim,
                                              activation=gelu, use_bias=True, kernel_initializer=self.xavier_init,
                                              kernel_regularizer=self.l2_reg, bias_regularizer=self.l2_reg,
                                              name='c2_hidden_inputs'
                                              )
            weight = tf.get_variable('W_projection_c2',
                                     shape=[self.pred_vec_dim_theta*self.cell_dim, self.c2_nums],
                                     initializer=self.norm_initializer)
            bias = tf.Variable(tf.constant(0.1, shape=[self.c2_nums]), name='b_c2')
            self.l2_loss_sen = tf.add_n([tf.nn.l2_loss(weight), tf.nn.l2_loss(bias)])
            c2_hidden_inputs_drop = tf.nn.dropout(c2_hidden_inputs, self.dropout_keep)
            self.scores_sen = tf.nn.xw_plus_b(c2_hidden_inputs_drop, weight, bias, name='scores_c2')
            self.predictions_sen = tf.nn.softmax(self.scores_sen)
            # [B, class_num, 1]
            c2_pred_scores = tf.expand_dims(self.predictions_sen, axis=-1)
            # [B, class_num, e] = [class_num, e] [B, class_num, 1]
            c2_pred_scored_embed = tf.multiply(self.label_embed_var_dict['kb_c2'], c2_pred_scores)
            self.c2_pred_scored_embed = tf.reduce_sum(c2_pred_scored_embed, axis=1)

    def add_pred_c2_att_op(self):
        with tf.variable_scope('pred_c2_att_op') as scope:
            # [B, e] e=cell_dim*2
            self.pred_c2_att_outputs, pred_c2_att_weights = attention_pre(inputs=self.rnn_outputs,
                                                                          sequence_lengths=self.sent_lengths,
                                                                          pre_att=self.c2_pred_scored_embed,
                                                                          scope='pred_c2_att')

    # @ label embed loss
    def add_c2_label_embed_loss_op(self):
        with tf.name_scope('c2_label_loss'):
            # [B, embedding_size (e)]
            print('ADDING_c2_label_embed_loss_op')
            batch_c2_embeds = tf.matmul(tf.cast(self.input_y2, dtype=tf.float32), self.label_embed_var_dict['kb_c2'])
            weight = tf.get_variable('W_c2_embed_classify',
                                     shape=[self.embedding_size, self.c2_nums],
                                     initializer=self.norm_initializer
                                     )
            bias = tf.Variable(tf.constant(0.1, shape=[self.c2_nums]), name='b_c2_embed_classify')
            c2_embed_pred_scores = tf.nn.xw_plus_b(batch_c2_embeds, weight, bias, name='c2_embed_pred_scores')
            self.c2_embed_preds = tf.nn.softmax(c2_embed_pred_scores)
            self.c2_embed_pred_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=c2_embed_pred_scores, labels=self.input_y2))

    def add_dynamic_routing_c2keywordstoc3_op(self):
        print('ADDING c2 keywords routed to c3 operator')
        # shape [output_dim, output_atoms] i.e. [c3_nums, word_embed_dim]
        self.c3emb_routedby_c2keywords, self.rout_c23 = _keywords_dynamic_routingto_label(
                                                        var_keywords_embed=self.var_keywords_embeddings_c2,
                                                        keywords_total_nums=self.c2keywords_total_nums,
                                                        var_label_embed=self.var_label_embeddings_c3,
                                                        label_nums=self.c3_nums, embed_size=self.embedding_size,
                                                        num_routing=3, scope='c2keywords_dynamic_routing_toc3',
                                                        tree_idx_list=self.c2toc3_tree_list, cluster_num=self.cluster_num)

    # @ keywords dynamic routed c3 att word
    def add_c2keywordsrouted_c3label_word_att_op(self):
        print('ADDING c2keywords routed c3label attention word op')
        input_x_mask = tf.reshape(self.input_x_mask, [-1, self.sentence_len])
        # [B, e] e=cell_dim*2
        c3routed_word_outputs, self.c3routed_word_att_weights = attention_class(inputs=self.rnn_outputs,
                                                            class_embedding=self.c3emb_routedby_c2keywords,
                                                            input_mask=input_x_mask, sequence_lengths=self.sent_lengths,
                                                            class_num=self.c3_nums,
                                                            class_embedding_dim=self.embedding_size,
                                                            scope='routed_c3_wordatt')
        self.c3routed_word_outputs = c3routed_word_outputs

    def add_c3_prediction_op(self):
        with tf.name_scope('output_c3'):
            # [B,e*3] = [B,e] [B,e] [B,e] e=cell_dim*2
            c2c3word_concat = tf.concat([self.pred_c2_att_outputs, self.c3_word_outputs, self.c3routed_word_outputs], axis=-1)
            print('ADDING layer normalization before dropout')
            c2c3word_concat_ln = layer_norm(c2c3word_concat)
            c2c3word_concat_ln_drop = tf.nn.dropout(c2c3word_concat_ln, self.dropout_keep)
            print('the input Embeddings of c3 softmax dim is %d' % (self.pred_vec_dim_theta*self.cell_dim))
            c3_hidden_inputs = tf.layers.dense(inputs=c2c3word_concat_ln_drop,
                                               units=self.pred_vec_dim_theta*self.cell_dim,
                                               activation=gelu, use_bias=True, kernel_initializer=self.xavier_init,
                                               kernel_regularizer=self.l2_reg, bias_regularizer=self.l2_reg,
                                               name='c3_hidden_inputs'
                                               )
            weight = tf.get_variable('W_projection_c3',
                                     shape=[self.pred_vec_dim_theta*self.cell_dim, self.c3_nums],
                                     initializer=self.norm_initializer)
            bias = tf.Variable(tf.constant(0.1, shape=[self.c3_nums]), name='b_c3')
            self.l2_loss_c3 = tf.add_n([tf.nn.l2_loss(weight), tf.nn.l2_loss(bias)])
            c3_hidden_inputs_drop = tf.nn.dropout(c3_hidden_inputs, self.dropout_keep)
            self.scores_c3 = tf.nn.xw_plus_b(c3_hidden_inputs_drop, weight, bias, name='scores_c3')
            self.predictions_c3 = tf.nn.softmax(self.scores_c3)

    # @ label embed loss
    def add_c3_label_embed_loss_op(self):
        with tf.name_scope('c3_label_loss'):
            # [B, embedding_size (e)]
            print('ADDING c3 label embed loss op')
            batch_c3_embeds = tf.matmul(tf.cast(self.input_y3, dtype=tf.float32), self.label_embed_var_dict['kb_c3'])
            weight = tf.get_variable('W_c3_embed_classify',
                                     shape=[self.embedding_size, self.c3_nums],
                                     initializer=self.norm_initializer
                                     )
            bias = tf.Variable(tf.constant(0.1, shape=[self.c3_nums]), name='b_c3_embed_classify')
            c3_embed_pred_scores = tf.nn.xw_plus_b(batch_c3_embeds, weight, bias, name='c3_embed_pred_scores')
            self.c3_embed_preds = tf.nn.softmax(c3_embed_pred_scores)
            self.c3_embed_pred_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=c3_embed_pred_scores, labels=self.input_y3))

    def add_loss_op(self):
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores_word, labels=self.input_y1)
            self.loss_c1 = tf.reduce_mean(losses) + 0.001 * self.l2_loss_word
            tf.summary.scalar('loss_c1', self.loss_c1)

            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores_sen, labels=self.input_y2)
            self.loss_c2 = tf.reduce_mean(losses) + 0.001 * self.l2_loss_sen
            tf.summary.scalar('loss_c2', self.loss_c2)

            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores_c3, labels=self.input_y3)
            self.loss_c3 = tf.reduce_mean(losses) + 0.001 * self.l2_loss_c3
            tf.summary.scalar('loss_c3', self.loss_c3)

            self.loss = self.loss_c1 + self.loss_c2 + self.c2_embed_pred_loss \
                                     + self.loss_c3 + self.c3_embed_pred_loss

        with tf.name_scope('accuracy'):
            truth_c1 = tf.argmax(self.input_y1, 1)
            pred_c1 = tf.argmax(self.predictions_word, 1)
            correct_preds_c1 = tf.cast(tf.equal(pred_c1, truth_c1), 'float')
            self.accuracy_c1 = tf.reduce_mean(correct_preds_c1, name='accuracy_c1')
            tf.summary.scalar('accuracy_c1', self.accuracy_c1)

            truth_c2 = tf.argmax(self.input_y2, 1)
            pred_c2 = tf.argmax(self.predictions_sen, 1)
            correct_preds_c2 = tf.cast(tf.equal(pred_c2, truth_c2), 'float')
            self.accuracy_c2 = tf.reduce_mean(correct_preds_c2, name='accuracy_c2')
            tf.summary.scalar('accuracy_c2', self.accuracy_c2)

            # c2 embedding predicts
            c2_emb_pred = tf.argmax(self.c2_embed_preds, 1)
            c2_emb_corrects = tf.cast(tf.equal(c2_emb_pred, truth_c2), 'float')
            self.accuracy_c2emb = tf.reduce_mean(c2_emb_corrects, name='accuracy_c2emb')
            tf.summary.scalar('accuracy_c2emb', self.accuracy_c2emb)

            truth_c3 = tf.argmax(self.input_y3, 1)
            pred_c3 = tf.argmax(self.predictions_c3, 1)
            correct_preds_c3 = tf.cast(tf.equal(pred_c3, truth_c3), 'float')
            self.accuracy_c3 = tf.reduce_mean(correct_preds_c3, name='accuracy_c3')
            tf.summary.scalar('accuracy_c3', self.accuracy_c3)

            # c3 embedding predicts
            c3_emb_pred = tf.argmax(self.c3_embed_preds, 1)
            c3_emb_corrects = tf.cast(tf.equal(c3_emb_pred, truth_c3), 'float')
            self.accuracy_c3emb = tf.reduce_mean(c3_emb_corrects, name='accuracy_c3emb')
            tf.summary.scalar('accuracy_c3emb', self.accuracy_c3emb)

            # c1c2c3 overall acc
            correct_preds_overall = correct_preds_c1 * correct_preds_c2 * correct_preds_c3
            self.accuracy_overall = tf.reduce_mean(correct_preds_overall, name='accuracy_overall')
            tf.summary.scalar('accuracy_c12overall', self.accuracy_overall)

    def add_train_op(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.use_bn:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.learning_rate = tf.train.exponential_decay(self.init_lr,
                                                        self.global_step,
                                                        self.lr_decay_step,
                                                        self.lr_decay_rate,
                                                        staircase=True)
        tf.summary.scalar('lr', self.learning_rate)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        if self.use_bn:
            with tf.control_dependencies(update_ops):
                grads_and_vars = optimizer.compute_gradients(self.loss)
                self.train_op = optimizer.apply_gradients(
                    grads_and_vars, global_step=self.global_step)
        else:
            grads_and_vars = optimizer.compute_gradients(self.loss)                  
            self.train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=self.global_step)

    def add_summaries(self):
        self.merged_summaries = tf.summary.merge_all()
        self.train_summ_writer = None
        self.dev_summ_writer = None
        if self.summaries_dir:
            train_summ_dir = os.path.join(self.summaries_dir, 'train')
            dev_summ_dir = os.path.join(self.summaries_dir, 'dev')
            self.train_summ_writer = tf.summary.FileWriter(
                train_summ_dir, self.sess.graph)
            self.dev_summ_writer = tf.summary.FileWriter(
                dev_summ_dir, self.sess.graph)

    def initialize_session(self, model_path=None):
        np.random.seed(666)
        tf.set_random_seed(233)
        random.seed(888)
        self.saver = tf.train.Saver(max_to_keep=self.max_model_to_save)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config)
        if model_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, model_path)
        self.add_summaries()
        return self.sess

    def save_model(self, save_path, global_step=None):
        if global_step is None:
            self.saver.save(self.sess, save_path)
        else:   
            self.saver.save(self.sess, save_path, global_step=global_step)

    def run_train_step(self, feed_dict):
        _, global_step, loss, accuracy_c1, accuracy_c2, accuracy_c3, \
        learning_rate, summary, accuracy_c2emb, accuracy_c3emb, \
        dyr_scores_c12, dyr_scores_c23, accuracy_overall = self.sess.run(
            [self.train_op, self.global_step, self.loss, self.accuracy_c1, self.accuracy_c2, self.accuracy_c3,
             self.learning_rate, self.merged_summaries, self.accuracy_c2emb, self.accuracy_c3emb,
             self.rout_c12, self.rout_c23, self.accuracy_overall], feed_dict=feed_dict)
        if self.train_summ_writer is not None:
            self.train_summ_writer.add_summary(summary, global_step)
        return global_step, loss, accuracy_c1, accuracy_c2, accuracy_c3, \
               learning_rate, accuracy_c2emb, accuracy_c3emb, accuracy_overall

    def run_eval_step(self, feed_dict, is_dev=False):
        loss, preds_c1, preds_c2, preds_c3,\
        truths_c1, truths_c2, truths_c3,\
        accuracy_c1, accuracy_c2, accuracy_c3, \
        summary, preds_c2emb, preds_c3emb = self.sess.run(
            [self.loss, self.predictions_word, self.predictions_sen, self.predictions_c3,
             self.input_y1, self.input_y2, self.input_y3,
             self.accuracy_c1, self.accuracy_c2, self.accuracy_c3,
             self.merged_summaries, self.c2_embed_preds, self.c3_embed_preds], feed_dict)
        if is_dev and self.dev_summ_writer is not None:
            self.dev_summ_writer.add_summary(summary)
        return loss, preds_c1, preds_c2, preds_c3, \
               truths_c1, truths_c2, truths_c3, \
               accuracy_c1, accuracy_c2, accuracy_c3, \
               preds_c2emb, preds_c3emb
