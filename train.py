#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Create: 2020/12/01
#
import argparse
from datetime import datetime
import logging
import os
import time
import glob
import sys
import random

import numpy as np
import tensorflow as tf

import multiprocessing as mp
from data_helper import ClassLabels
from data_helper import DataHelper
from data_helper import smartcat_raw_loader
from modeling import CLED
from data_helper import LabelsTrans
np.random.seed(666)
tf.set_random_seed(233)
random.seed(888)


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] [%(asctime)s]: %(message)s',
                    datefmt='%Y%m%d %H:%M:%S')

ARGS = None
LABELS_MANAGER_DICT = None
DATA_HELPER_OBJ = None
SUMMARIES_DIR = None
MODEL_DIR = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output',
                        help='the dir to save models and summaries.')
    parser.add_argument('--save_summary', default=False, action='store_true',
                        help='save summaries or not.')
    parser.add_argument('--rebuild_tfrecord', default=False, action='store_true',
                        help='rebuild tfrecord file of train or dev data.')
    parser.add_argument('--rebuild_test_tfrecord', default=False, action='store_true',
                        help='rebuild tfrecord file of test data.')

    parser.add_argument('--epoch_num', type=int, default=1,
                        help='number of epoches to run training.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size.')
    parser.add_argument('--filter_num', type=int, default=128,
                        help='number of filters.')
    parser.add_argument('--sentence_len', type=int, default=512,
                        help='max length of sentence to process.')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='dimension of word embedding.')
    parser.add_argument('--max_save_model', type=int, default=3,
                        help='max number of model to save in a train.')
    parser.add_argument('--eval_every_steps', type=int, default=1000,
                        help='evaluate on dev set after this many steps.')
    parser.add_argument('--early_stop_times', type=int, default=10,
                        help=('number of times with no promotion before'
                              'early stopping.'))
    parser.add_argument('--dropout_keep', type=float, default=0.5,
                        help='keep ratio of dropout.')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='initial learning rate.')
    parser.add_argument('--lr_decay_step', type=int, default=1000,
                        help='number of steps to decay learning rate.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95,
                        help='decay rate of learning rate.')
    parser.add_argument('--embedding_file', type=str,
                        help='pretrained embedding file.')
    parser.add_argument('--vocab_file', type=str, required=True,
                        help='vocabulary file')
    parser.add_argument('--c1_labels_file', type=str,
                        help='c1_labels file, is needed when hierarchy attention')
    parser.add_argument('--c2_labels_file', type=str,
                        help='c2_labels file, is needed when hierarchy attention')
    parser.add_argument('--c3_labels_file', type=str,
                        help='c3_labels file, is needed when hierarchy attention')
    parser.add_argument('--train_file', type=str, required=True,
                        help='train data file')
    parser.add_argument('--dev_file', type=str, help='dev data file')
    parser.add_argument('--test_file', type=str, help='test data file')
    parser.add_argument('--c1_kb_label_embeddings_file', type=str, default='',
                        help='C1 prebuilt kb_label embeddings file')

    parser.add_argument('--c2_kb_label_embeddings_file', type=str, default='',
                        help='C2 prebuilt kb_label embeddings file')

    parser.add_argument('--c3_kb_label_embeddings_file', type=str, default='',
                        help='C3 prebuilt kb_label embeddings file')

    parser.add_argument('--cell_dim', type=int, default=150, help='dimension of word embedding.')

    parser.add_argument('--numof_keywords_percat1', type=int, default=20,
                        help='number of keywords per c1 used in model')
    parser.add_argument('--c1_keywords_embeddings_file', type=str, default='',
                        help='C1 prebuilt keywords embeddings file')
    parser.add_argument('--c2_keywords_embeddings_file', type=str, default='',
                        help='C2 prebuilt keywords embeddings file')
    parser.add_argument('--last_dim_theta', type=int, default=2, help='last dim befor softmax')
    parser.add_argument('--cluster_num', type=int, default=20, help='last dim befor softmax')

    return parser.parse_args()


def data_process():
    train_len = 0
    dev_len = 0
    test_len = 0
    p_list = []

    # ################################  train  ######################################
    train_file_seg = ARGS.train_file + '??'
    train_names_seg = []
    train_record_names = []
    for fname in sorted(glob.glob(train_file_seg)):
        train_names_seg.append(fname)
        if ARGS.rebuild_tfrecord:
            logging.info('------build' + fname)
        else:
            logging.info('------reuse no need to build' + fname)

    for index in range(len(train_names_seg)):
        if os.path.exists(train_names_seg[index]):
            train_tfrecord = train_names_seg[index] + '.tfrecord'
            train_record_names.append(train_tfrecord)

            p = mp.Process(target=DATA_HELPER_OBJ.load_data,
                           args=(smartcat_raw_loader, train_names_seg[index],  train_tfrecord, ARGS.rebuild_tfrecord))
            p_list.append(p)

    print('%s tfrecord is DONE' % 'TRAIN')
    sys.stdout.flush()

    # ############################################################################

    # ################################  dev  ######################################
    dev_dataset = None
    dev_record_names = []
    if ARGS.dev_file:
        dev_file_seg = ARGS.dev_file + '??'
        dev_names_seg = []

        for fname in sorted(glob.glob(dev_file_seg)):
            dev_names_seg.append(fname)
            if ARGS.rebuild_tfrecord:
               print('------build' + fname)
            else:
               print('------reuse no need to build' + fname)

        for index in range(len(dev_names_seg)):
            if os.path.exists(dev_names_seg[index]):
                dev_tfrecord = dev_names_seg[index] + '.tfrecord'
                dev_record_names.append(dev_tfrecord)

                p = mp.Process(target=DATA_HELPER_OBJ.load_data,
                               args=(smartcat_raw_loader, dev_names_seg[index],  dev_tfrecord, ARGS.rebuild_tfrecord))
                p_list.append(p)

        print('%s tfrecord is DONE' % 'DEV')
        sys.stdout.flush()
    # ############################################################################

    # ################################  test  ######################################
    test_dataset = None
    test_record_names = []
    if ARGS.test_file:
        test_file_seg = ARGS.test_file + '??'
        test_names_seg = []

        for fname in sorted(glob.glob(test_file_seg)):
            test_names_seg.append(fname)
            if ARGS.rebuild_test_tfrecord:
                print('------build' + fname)
            else:
                print('------reuse no need to build' + fname)
        sys.stdout.flush()

        for index in range(len(test_names_seg)):
            if os.path.exists(test_names_seg[index]):
                test_tfrecord = test_names_seg[index] + '.tfrecord'
                test_record_names.append(test_tfrecord)

                p = mp.Process(target=DATA_HELPER_OBJ.load_data,
                          args=(smartcat_raw_loader, test_names_seg[index], test_tfrecord, ARGS.rebuild_test_tfrecord))
                p_list.append(p)

        print('%s tfrecord is DONE' % 'TEST')
        sys.stdout.flush()

    # ############################################################################

    for pi in p_list:
        pi.start()
    for pi in p_list:
        pi.join()

    print('BATCH SIZE IS %d' % ARGS.batch_size)
    train_dataset = DATA_HELPER_OBJ.create_dataset(
        train_record_names, ARGS.batch_size, shuf_buffer_size=10000)
    for fname in train_record_names:
        train_len += sum(1 for _ in tf.python_io.tf_record_iterator(fname))
    print('%s dataset is DONE example: %d' % ('TRAIN', train_len))
    sys.stdout.flush()

    if len(dev_record_names) != 0:
        dev_dataset = DATA_HELPER_OBJ.create_dataset(dev_record_names, ARGS.batch_size)
        for fname in dev_record_names:
            dev_len += sum(1 for _ in tf.python_io.tf_record_iterator(fname))
        print('%s dataset is DONE example: %d' % ('DEV', dev_len))
        sys.stdout.flush()

    if len(test_record_names) != 0:
        test_dataset = DATA_HELPER_OBJ.create_dataset(test_record_names, ARGS.batch_size)
        for fname in test_record_names:
            test_len += sum(1 for _ in tf.python_io.tf_record_iterator(fname))
        print('%s dataset is DONE example: %d' % ('TEST', test_len))
        sys.stdout.flush()

    return train_dataset, dev_dataset, test_dataset

def do_train(train_dataset, dev_dataset, test_dataset):
    # Create a feedable and real batch iterators {{{.
    data_handle = tf.placeholder(tf.string, shape=[])
    data_iterator = tf.data.Iterator.from_string_handle(
        data_handle, train_dataset.output_types, train_dataset.output_shapes)
    doc_id, x_batch, y1_batch, y2_batch, y3_batch, batch_sentence_len, batch_word_mask = data_iterator.get_next()

    train_iter = train_dataset.make_initializable_iterator()
    if dev_dataset is not None:
        dev_iter = dev_dataset.make_initializable_iterator()
    if test_dataset is not None:
        test_iter = test_dataset.make_initializable_iterator()
    # }}}.
    # Create and build model.
    print('Creating model instance and initializing session...')
    model = CLED(DATA_HELPER_OBJ.embedding_dim,
                    DATA_HELPER_OBJ.vocab_size,
                    ARGS.sentence_len,
                    LABELS_MANAGER_DICT,
                    DATA_HELPER_OBJ.pretrained_embedding,
                    ARGS.filter_num,
                    ARGS.max_save_model,
                    SUMMARIES_DIR,
                    init_lr=ARGS.init_lr,
                    lr_decay_step=ARGS.lr_decay_step,
                    label_embedding_dict=DATA_HELPER_OBJ.prebuilt_label_embeds_dict,
                    cell_dim=ARGS.cell_dim,
                    c1keywords_embed=DATA_HELPER_OBJ.c1_keywords_embedding,
                    c2keywords_embed=DATA_HELPER_OBJ.c2_keywords_embedding,
                    last_dim_theta=ARGS.last_dim_theta,
                    cluster_num=ARGS.cluster_num)
    model.build(x_batch, y1_batch, y2_batch, y3_batch, batch_word_mask, batch_sentence_len)
    sess = model.initialize_session()

    train_handle = sess.run(train_iter.string_handle())
    if dev_dataset is not None:
        dev_handle = sess.run(dev_iter.string_handle())
    if test_dataset is not None:
        test_handle = sess.run(test_iter.string_handle())

    print('dropout keep prob is %s' % str(ARGS.dropout_keep))
    # Epoches loop.
    early_stop = False
    early_stop_info = [1e10, 0]
    for epoch_num in range(ARGS.epoch_num):
        print('Start loop of epoch %d.' % epoch_num)
        sess.run(train_iter.initializer)

        # Batches loop.
        while True:
            try:
                start = time.time()
                global_step, batch_loss, c1_batch_acc, c2_batch_acc, c3_batch_acc, \
                learning_rate, c2emb_batch_acc, c3emb_batch_acc, acc_overall = model.run_train_step(
                    {model.dropout_keep:ARGS.dropout_keep, data_handle:train_handle})
                step_time = time.time() - start
                print('epoch %d:, gstep:%d, b-loss:%.4f, cat1_b-acc:%.4f, cat2_b-acc:%.4f, cat3_b-acc:%.4f,'
                             'lr:%.6f, step_time:%.2fs. '
                      'cat2emb_b-acc:%.4f, cat3emb_b-acc:%.4f, overall_b-acc:%.4f' %
                             (epoch_num, global_step, batch_loss, c1_batch_acc, c2_batch_acc, c3_batch_acc,
                              learning_rate, step_time,
                              c2emb_batch_acc, c3emb_batch_acc, acc_overall))
                if global_step % 500 == 0:
                    sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                print('Out of data in epoch %d.' % epoch_num)
                break

            # Periodical evaluation {{{.
            if (ARGS.eval_every_steps > 0 and global_step % ARGS.eval_every_steps == 0):
                # Evaluate on test set.
                if test_dataset is not None:
                    sess.run(test_iter.initializer)
                    do_eval(model, {model.dropout_keep: 1.0, data_handle: test_handle}, 'test')

                # Evaluate on dev set and judge early-stopping.
                if dev_dataset is not None:
                    sess.run(dev_iter.initializer)
                    early_stop = eval_and_earlystop(
                        model, epoch_num, global_step,
                        {model.dropout_keep:1.0, data_handle:dev_handle},
                        early_stop_info)
                    if early_stop:
                        break
            # }}}.

        if early_stop:
            break

        # Evaluate on test set.
        if test_dataset is not None:
            sess.run(test_iter.initializer)
            do_eval(model, {model.dropout_keep:1.0, data_handle:test_handle}, 'test')

        # Evaluate on dev set and judge early-stopping.
        if dev_dataset is not None:
            sess.run(dev_iter.initializer)
            early_stop = eval_and_earlystop(
                model, epoch_num, global_step,
                {model.dropout_keep:1.0, data_handle:dev_handle},
                early_stop_info)
            if early_stop:
                break
        # Save model every epoch if no dev set.
        else:
            save_model(model, epoch_num, global_step, 0)

def eval_and_earlystop(model, epoch_num, global_step, feed_dict, es_info, c31trans=None, c32trans=None):
    min_dev_loss, not_impv_times = es_info
    dev_acc_c1, dev_acc_c2, dev_loss = do_eval(model, feed_dict, 'dev')
    if dev_loss < min_dev_loss:
        min_dev_loss = dev_loss
        not_impv_times = 0
        save_model(model, epoch_num, global_step, dev_acc_c2)
    else:
        not_impv_times += 1
        print('Dev loss not decresed from %.4f for %d times.' %
                     (min_dev_loss, not_impv_times))
        if not_impv_times >= ARGS.early_stop_times:
            print('Early stopped!')
            es_info[:] = [min_dev_loss, not_impv_times]
            return True

    es_info[:] = [min_dev_loss, not_impv_times]
    return False


def do_eval(model, feed_dict, data_name='dev', c31trans=None, c32trans=None):
    is_dev = (data_name == 'dev')
    eval_loss = 0.0
    all_preds_c1 = []
    all_truths_c1 = []

    all_preds_c2 = []
    all_truths_c2 = []
    all_preds_c2emb = []

    all_preds_c3 = []
    all_truths_c3 = []
    all_preds_c3emb = []

    batch_num = 0
    while True:
        batch_num += 1
        try:
            loss, preds_c1, preds_c2, preds_c3, \
            truths_c1, truths_c2, truths_c3, _, _, _, preds_c2emb, preds_c3emb = model.run_eval_step(feed_dict, is_dev)
            all_preds_c1.extend(np.argmax(preds_c1, axis=1).tolist())
            all_truths_c1.extend(np.argmax(truths_c1, axis=1).tolist())

            all_preds_c2.extend(np.argmax(preds_c2, axis=1).tolist())
            all_truths_c2.extend(np.argmax(truths_c2, axis=1).tolist())

            all_preds_c2emb.extend(np.argmax(preds_c2emb, axis=1).tolist())

            all_preds_c3.extend(np.argmax(preds_c3, axis=1).tolist())
            all_truths_c3.extend(np.argmax(truths_c3, axis=1).tolist())

            all_preds_c3emb.extend(np.argmax(preds_c3emb, axis=1).tolist())

            eval_loss += loss
        except tf.errors.OutOfRangeError:
            break

    eval_loss /= batch_num

    eval_acc_c1 = cal_acc(all_preds_c1, all_truths_c1)

    eval_acc_c2 = cal_acc(all_preds_c2, all_truths_c2)
    eval_acc_c2emb = cal_acc(all_preds_c2emb, all_truths_c2)

    eval_acc_c3 = cal_acc(all_preds_c3, all_truths_c3)
    eval_acc_c3emb = cal_acc(all_preds_c3emb, all_truths_c3)

    eval_acc_overall = cal_overall_acc(all_preds_c1, all_truths_c1, all_preds_c2, all_truths_c2,
                                       all_preds_c3, all_truths_c3)
    print('Evaluate on "%s" data, loss: %.4f, cat1_acc: %.4f, cat2_acc: %.4f, cat3_acc: %.4f, '
          'cat2emb_acc: %.4f, cat3emb_acc: %.4f, overall_acc: %.4f' %
                 (data_name, eval_loss, eval_acc_c1, eval_acc_c2, eval_acc_c3,
                  eval_acc_c2emb, eval_acc_c3emb, eval_acc_overall))
    sys.stdout.flush()

    return eval_acc_c1, eval_acc_c2, eval_loss

def cal_acc(all_preds, all_truths):
    pred_y = np.array(all_preds)
    truth_y = np.array(all_truths)
    corr_num = np.sum(np.equal(truth_y, pred_y).astype(int))
    eval_acc = corr_num / float(len(truth_y))
    return eval_acc

def cal_overall_acc(all_preds_c1, all_truths_c1, all_preds_c2, all_truths_c2, all_preds_c3, all_truths_c3):
    corr_c1 = np.equal(np.array(all_preds_c1), np.array(all_truths_c1)).astype(int)
    corr_c2 = np.equal(np.array(all_preds_c2), np.array(all_truths_c2)).astype(int)
    corr_c3 = np.equal(np.array(all_preds_c3), np.array(all_truths_c3)).astype(int)
    overall_corr = corr_c1 * corr_c2 * corr_c3
    acc_overall = np.sum(overall_corr) / float(len(overall_corr))
    return acc_overall

def save_model(model, epoch_num, global_step, dev_acc):
    model_name = ('%s/model_ed-%d_sl-%d_fn-%d_ep-%d_step-%d_devacc-'
                  '%.4f.ckpt' % (MODEL_DIR,
                                 DATA_HELPER_OBJ.embedding_dim,
                                 ARGS.sentence_len,
                                 ARGS.filter_num,
                                 epoch_num,
                                 global_step,
                                 dev_acc))
    model.save_model(model_name)

def main():
    np.random.seed(666)
    tf.set_random_seed(233)
    random.seed(888)

    global ARGS
    ARGS = parse_args()

    # Create models dir.
    train_file_name = ARGS.train_file.split('/')[-1]
    global MODEL_DIR
    MODEL_DIR = os.path.join(ARGS.output_dir, 'models', train_file_name)
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Create summaries dir.
    if ARGS.save_summary:
        timestamp = datetime.now().strftime('run_%Y%m%d%H%M') 
        global SUMMARIES_DIR
        SUMMARIES_DIR = os.path.join(ARGS.output_dir, 'summaries', timestamp)
        if not os.path.isdir(SUMMARIES_DIR):
            os.makedirs(SUMMARIES_DIR)

    # Create instance of ClassLabels.
    print('Creating ClassLabels instance...')

    c1labels_manager = ClassLabels(ARGS.c1_labels_file)
    c2labels_manager = ClassLabels(ARGS.c2_labels_file)
    c3labels_manager = ClassLabels(ARGS.c3_labels_file)
    global LABELS_MANAGER_DICT
    LABELS_MANAGER_DICT = {'c1': c1labels_manager, 'c2': c2labels_manager, 'c3': c3labels_manager}

    label_embed_files_path = {}
    label_embed_files_path['kb_c1'] = ARGS.c1_kb_label_embeddings_file
    label_embed_files_path['kb_c2'] = ARGS.c2_kb_label_embeddings_file
    label_embed_files_path['kb_c3'] = ARGS.c3_kb_label_embeddings_file
    # Create instance of DataHelper.
    print('Creating DataHelper instance...')
    global DATA_HELPER_OBJ
    DATA_HELPER_OBJ = DataHelper(LABELS_MANAGER_DICT,
                                 vocab_file=ARGS.vocab_file,
                                 sentence_len=ARGS.sentence_len,
                                 embed_dim=ARGS.embedding_dim,
                                 embed_file=ARGS.embedding_file,
                                 label_embed_files_dict=label_embed_files_path,
                                 numof_keywords_percat1=ARGS.numof_keywords_percat1,
                                 c1keywords_file=ARGS.c1_keywords_embeddings_file,
                                 c2keywords_file=ARGS.c2_keywords_embeddings_file
                                 )

    # Load corpus and create dataset.
    print('Loading corpus files...')
    train_dataset, dev_dataset, test_dataset = data_process()
    # Do training.
    print('Start training.')
    do_train(train_dataset, dev_dataset, test_dataset)
    print('Finished!')

if __name__ == '__main__':
    main()
