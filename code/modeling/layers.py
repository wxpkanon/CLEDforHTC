import os
import tensorflow as tf
import numpy as np
from .utils import get_shape

try:
  from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
  LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple



def bidirectional_rnn(cell_fw, cell_bw, inputs, input_lengths,
                      initial_state_fw=None, initial_state_bw=None,
                      scope=None):
  with tf.variable_scope(scope or 'bi_rnn') as scope:
    (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw,
      cell_bw=cell_bw,
      inputs=inputs,
      sequence_length=input_lengths,
      initial_state_fw=initial_state_fw,
      initial_state_bw=initial_state_bw,
      dtype=tf.float32,
      scope=scope
    )
    outputs = tf.concat((fw_outputs, bw_outputs), axis=2)

    def concatenate_state(fw_state, bw_state):
      if isinstance(fw_state, LSTMStateTuple):
        state_c = tf.concat(
          (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
        state_h = tf.concat(
          (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
        state = LSTMStateTuple(c=state_c, h=state_h)
        return state
      elif isinstance(fw_state, tf.Tensor):
        state = tf.concat((fw_state, bw_state), 1,
                          name='bidirectional_concat')
        return state
      elif (isinstance(fw_state, tuple) and
            isinstance(bw_state, tuple) and
            len(fw_state) == len(bw_state)):
        # multilayer
        state = tuple(concatenate_state(fw, bw)
                      for fw, bw in zip(fw_state, bw_state))
        return state

      else:
        raise ValueError(
          'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)
    return outputs, state

def masking(scores, sequence_lengths, score_mask_value=tf.constant(-np.inf)):
  score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
  score_mask_values = score_mask_value * tf.ones_like(scores)
  return tf.where(score_mask, scores, score_mask_values)

def partial_softmax(logits, weights, dim, name,):
  with tf.variable_scope('partial_softmax'):
    exp_logits = tf.exp(logits)
    if len(exp_logits.get_shape()) == len(weights.get_shape()):
      exp_logits_weighted = tf.multiply(exp_logits, weights)
    else:
      exp_logits_weighted = tf.multiply(exp_logits, tf.expand_dims(weights, -1))
    exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keep_dims=True)
    partial_softmax_score = tf.div(exp_logits_weighted, exp_logits_sum, name=name)
    return partial_softmax_score

def attention(inputs, att_dim, sequence_lengths, scope=None):
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention'):
    word_att_W = tf.get_variable(name='att_W', shape=[att_dim, 1])

    projection = tf.layers.dense(inputs, att_dim, tf.nn.tanh, name='projection')

    alpha = tf.matmul(tf.reshape(projection, shape=[-1, att_dim]), word_att_W)
    alpha = tf.reshape(alpha, shape=[-1, get_shape(inputs)[1]])
    alpha = masking(alpha, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))
    alpha = tf.nn.softmax(alpha)

    outputs = tf.reduce_sum(inputs * tf.expand_dims(alpha, 2), axis=1)
    return outputs, alpha


def attention_pre(inputs, sequence_lengths, pre_att, scope=None):
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention'):
    # [B*s, e] = [B,e]
    word_att_W = tf.tile(pre_att, [1, get_shape(inputs)[1]])
    # [B,s,e] = [B*s, e]
    word_att_W = tf.reshape(word_att_W, [-1, get_shape(inputs)[1], get_shape(pre_att)[1]])
    # [B,s,e]=[B,s,e][B,s,e]
    alpha = tf.multiply(inputs, word_att_W)
    # [B,s] = [B,s,e]
    alpha = tf.reduce_sum(alpha,-1)
    alpha = masking(alpha, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))
    # [B,s]
    alpha = tf.nn.softmax(alpha)
    # [B,e] = [B,s, e][B,s,1]
    outputs = tf.reduce_sum(inputs * tf.expand_dims(alpha, 2), axis=1)
    return outputs, alpha

def attention_class(inputs, class_embedding, input_mask, sequence_lengths, class_num, class_embedding_dim, scope=None):
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention_class'):
    x_mask = tf.expand_dims(input_mask, axis=-1)  # b*s*1
    x_emb_0 = inputs # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask)  # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2)  # b * s * e
    W_class_tran = tf.transpose(class_embedding, [1, 0])  # e * c
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim=0)  # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c
    filter_shape = [3, class_num, class_num]
    W = tf.get_variable(name='W', shape=filter_shape,
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        dtype=tf.float32, trainable=True)
    b = tf.get_variable(name='b', shape=[class_num],
                        initializer=tf.constant_initializer(0.1),
                        dtype=tf.float32, trainable=True)
    print('Scope %s with LEAM module' % scope)
    # [B, seq, class_num]
    conv = tf.nn.conv1d(G, W, stride=1, padding='SAME', name='conv')
    Att_v_1 = tf.nn.bias_add(conv, b)
    # [B, seq]
    Att_v = tf.reduce_max(Att_v_1, axis=-1)
    # [B, seq]
    Att_v_max = masking(Att_v, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))  # b * s
    # [B, seq]
    Att_v_max = tf.nn.softmax(Att_v_max)
    # [B, seq, 1] [b, s ,1]
    Att_v_max = tf.multiply(tf.expand_dims(Att_v_max, -1), x_mask)
    #  [b * s * e] [B, seq, 1]
    x_att = tf.multiply(x_emb_0, Att_v_max)
    # [B, e]
    outputs = tf.reduce_sum(x_att, axis=1)
    return outputs, Att_v_max
