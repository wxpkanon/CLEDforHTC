#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import collections
import logging
import os
import sys


import numpy as np
import tensorflow as tf


class Vocabulary(object):
    """Defines vocabulary.
    """
    def __init__(self, vocab_file):
        def add_word(word, word2idx, idx2word):
            """Adds word to list.
            """
            if word not in word2idx:
                index = len(word2idx)
                word2idx[word] = index
                idx2word[index] = word

        word_list = []
        try:
            word_list = [l.strip('\n') for l in
                         codecs.open(vocab_file, encoding='utf-8').readlines()]
        except:
            raise Exception(
                'Failed to load vocabulary from file(%s)!' % vocab_file)

        self.word_index = {}
        self.index_word = {}
        for word in word_list:
            add_word(word, self.word_index, self.index_word)

        print('Finished loading dictionary, totally %d words' %
                     len(self.word_index))

    def __len__(self):
        return len(self.word_index)

    def text_to_ids(self, text, trunc_len=None):
        """Transform words to token ids.
        """
        word_list = None
        if isinstance(text, str):
            word_list = text.split()
        elif isinstance(text, list):
            word_list = text
        else:
            raise Exception('Invalid value!')

        id_list = []
        for word in word_list:
            id_list.append(self.word_index.get(word, 0))

        # Truncate and pad.
        if trunc_len is not None:
            if len(id_list) < trunc_len:
                id_list.extend([0] * (trunc_len - len(id_list)))
            elif len(id_list) > trunc_len:
                id_list = id_list[0:trunc_len]

        return id_list

    def dump(self, save_file):
        """Dumps vocabulary to file.
        """
        with codecs.open(save_file, 'w', encoding='utf-8') as fout:
            for word in self.word_index:
                fout.write('%s\n' % word)


class ClassLabels(object):
    """Defines labels of classification.
    """
    def __init__(self, label_def):
        self.label_list = []
        self.label2id = {}
        self.id2label = {}
        if isinstance(label_def, str):
            items = [l.strip() for l in codecs.open(label_def, encoding='utf-8').readlines()]
        elif isinstance(label_def, list):
            items = label_def
        else:
            raise Exception('Invalid value!', label_def)

        for label in items:
            self.label_list.append(label)
            idx = len(self.label2id)
            self.label2id[label] = len(self.label2id)
            self.id2label[idx] = label

    def label_to_id(self, labels, need_onehot=False):
        """Change label to id.
        """
        ids = []
        for label in labels:
            assert (label in self.label2id), 'Undefined label: %s.' % label
            label_id = self.label2id[label]
            if need_onehot:
                onehot = [0] * self.class_num
                onehot[label_id] = 1
                ids.append(onehot)
            else:
                ids.append(label_id)
        return ids

    def id_to_label(self, ids):
        """Changes id to label.
        """
        labels = []
        for _id in ids:
            assert (_id in self.id2label), 'Undefined label id: %d.' % _id
            labels.append(self.id2label[_id])
        return labels

    @property
    def class_num(self):
        """Total number of classes.
        """
        return len(self.label2id)

    def labels(self):
        """All text-format labels.
        """
        return self.label2id.keys()


def smartcat_raw_loader(fname, valid_labels=None, with_label=True, sep_token='SEP', sentence_len=768):
    """A corpus loader function.

    Loads corpus with columns docid[Tab]truth_label[Tab]title[Tab]content.
    """
    docids = []
    texts = []
    c1lb_list = []
    c2lb_list = []
    c3lb_list = []
    sentence_lens = []
    word_mask = []
    with codecs.open(fname, encoding='utf-8') as fin:
        cnt = 0
        for i, line in enumerate(fin):
            if i % 10000 == 0:
                print('Finished loading %d lines.' % i)

            fields = line.strip('\n').split('\t')
            if with_label:

                # docid, _, _, label, text = fields[0:5]
                docid = fields[0].strip()
                c1lb = fields[1].strip()
                c2lb = fields[2].strip()
                c3lb = fields[3].strip()

                text = fields[9].strip()
                words = text.split()
                if valid_labels is not None and c1lb in valid_labels:
                    docids.append(docid)
                    c1lb_list.append(c1lb)
                    c2lb_list.append(c2lb)
                    c3lb_list.append(c3lb)
                    texts.append(text)

                    sentence_lens.append([min(len(words), sentence_len)])
                    w_mask = np.zeros(sentence_len).astype('float32')
                    w_mask[0:min(len(words), sentence_len)] = 1
                    word_mask.append(w_mask)
            else:
                pass

            cnt = i
        print('RAW_DATA_Count: %d, %s' % (cnt, fname))
    return docids, texts, c1lb_list, c2lb_list, c3lb_list, sentence_lens, word_mask


class DataHelper(object):
    """Data processing helper class.
    """
    def __init__(self, labels_manager_dict, vocab_file, sentence_len, embed_dim=None,
                 embed_file=None, label_embed_files_dict=None, numof_keywords_percat1=20, c1keywords_file=None,
                 c2keywords_file=None):
        if embed_dim is None and embed_file is None:
            raise Exception('Neither embed_dim nor embed_file is given, but'
                            'at least one is needed!')

        self.labels_manager_c1 = labels_manager_dict['c1']
        self.labels_manager_c2 = labels_manager_dict['c2']
        self.labels_manager_c3 = labels_manager_dict['c3']

        self.cat1_total = self.labels_manager_c1.class_num
        self.cat2_total = self.labels_manager_c2.class_num
        self.cat3_total = self.labels_manager_c3.class_num

        self.sentence_len = sentence_len
        if embed_dim is not None:
            self.embedding_dim = embed_dim

        # number of keywords per category used in model
        self.numof_keywords_percat1 = numof_keywords_percat1

        # Init vocabulary from vocab file.
        self.vocab = Vocabulary(vocab_file)

        # Loading pretrained embedding.
        if embed_file is not None:
            self.pretrained_embedding, self.word_embedding_map = self._load_embedding(embed_file)
            print('Finished loading pretrained embedding with given'
                         'vocabulary file, shape of embedding matrix is %s.' %
                         str(self.pretrained_embedding.shape))
        else:
            self.pretrained_embedding = None

        self.prebuilt_label_embeds_dict = {}

        for label_type in label_embed_files_dict.keys():
            cat_i = label_type.split('_')[-1]
            self.prebuilt_label_embeds_dict[label_type] \
                = self._prebuilt_label_embed_loader(files_dict=label_embed_files_dict,
                                                    labeltype=label_type, cat_i_manager=labels_manager_dict[cat_i])
        # load keywords of cat embedding
        if c1keywords_file is not None:
            self.c1_keywords_embedding = self._load_keywords_embedding(fname=c1keywords_file,
                            cat_i_manager=self.labels_manager_c1, numof_keywords_touse_percat=numof_keywords_percat1)
        if c2keywords_file is not None:
            self.c2_keywords_embedding = self._load_keywords_embedding(fname=c2keywords_file,
                            cat_i_manager=self.labels_manager_c2, numof_keywords_touse_percat=numof_keywords_percat1)


    def _prebuilt_label_embed_loader(self, files_dict, labeltype, cat_i_manager):
        # loading pre built label embedding
        if files_dict[labeltype] is not None:
            prebuilt_label_embed = self._load_label_embedding(files_dict[labeltype], cat_i_manager)
            print('Finished loading prebuilt label embedding with given'
                  '%s LABEL FILE, shape of embedding matrix is %s.  file path %s' %
                  (labeltype, str(prebuilt_label_embed.shape), files_dict[labeltype]))
        else:
            prebuilt_label_embed = None
        return prebuilt_label_embed

    def _load_embedding(self, fname):
        word_embedding_map = {}
        with codecs.open(fname, encoding='utf-8', errors='ignore') as fin:
            line = fin.readline().strip()
            param_fields = line.split()
            self.embedding_dim = int(param_fields[1])
            lnum = 0
            for line in fin:
                lnum += 1
                try:
                    values = line.strip().split()
                    word = values[0]
                except Exception:
                    print('exception line num: %d' % lnum)
                    print('exception line in embedding file:'+line)

                if word in self.vocab.word_index and len(values[1:])==self.embedding_dim:
                    try:
                        vec = np.asarray(values[1:], dtype='float32')
                        word_embedding_map[word] = vec
                    except ValueError:
                        pass

        default_vec = [0] * self.embedding_dim 
        embedding_vecs = []
        for idx in range(len(self.vocab)):
            word = self.vocab.index_word[idx]
            if word in word_embedding_map:
                embedding_vecs.append(word_embedding_map[word])
            else:
                embedding_vecs.append(default_vec)
        return np.array(embedding_vecs), word_embedding_map

    def _words_list_to_onevec(self, words_list):
        assert (len(words_list) != 0), 'words_list must not be 0 length'
        embed_vecs = []
        for w in words_list:
            if w in self.word_embedding_map:
                embed_vecs.append(self.word_embedding_map[w])
        assert (len(embed_vecs) != 0), 'all the words not found in vocab: ' % ' '.join(words_list)
        np_embed_vecs = np.asarray(embed_vecs, dtype='float32')
        one_vec = np.average(np_embed_vecs, axis=0)
        return one_vec

    def _load_label_embedding(self, fname, cat_i_manager):
        label_embedding_map = {}
        with codecs.open(fname, encoding='utf-8', errors='ignore') as fin:
            for line in fin:
                try:
                    sub_line = line.strip().split('\t')
                    words = sub_line[-1].split()
                    label = sub_line[-2]
                except Exception:
                    print ('exception line in embedding file:'+line)
                if label in cat_i_manager.labels():
                    try:
                        vec = self._words_list_to_onevec(words)
                        label_embedding_map[label] = vec
                    except Exception:
                        pass
                else:
                    print ('LABEL EMBEDDING: not found label in label manager '
                           'OR embedding dim wrong: %s' % label.encode('utf-8'))
                    sys.stdout.flush()

        print ('total label is %d' % cat_i_manager.class_num)
        sys.stdout.flush()
        default_vec = [0] * self.embedding_dim
        embedding_vecs = []
        for idx in range(cat_i_manager.class_num):
            label_name = cat_i_manager.id_to_label([idx])[0]
            if label_name in label_embedding_map:
                embedding_vecs.append(label_embedding_map[label_name])
            else:
                print ('%s NOT FOUND in label_embedding_map, so use default vector' % label_name.encode('utf-8'))
                sys.stdout.flush()
                embedding_vecs.append(default_vec)
        return np.array(embedding_vecs)

    def _load_keywords_embedding(self, fname, cat_i_manager, numof_keywords_touse_percat):
        cat_keyword_embedding_map = dict([(c, []) for c in cat_i_manager.labels()])
        num_keyword_notin_wordvecmap = 0
        with codecs.open(fname, encoding='utf-8', errors='ignore') as fin:
            for line in fin:
                try:
                    sub_line = line.strip().split('\t')
                    cat = sub_line[2]
                    keyword = sub_line[0]
                    if len(cat_keyword_embedding_map[cat]) < numof_keywords_touse_percat:
                        if keyword in self.word_embedding_map:
                            keyword_vec = self.word_embedding_map[keyword]
                            cat_keyword_embedding_map.get(cat).append(keyword_vec)
                        else:
                            num_keyword_notin_wordvecmap += 1

                except Exception:
                    print ('exception line in embedding file:'+line)
                    sys.stdout.flush()
        embedding_vecs = []  # shape [keywords, embed_dim]
        for idx in range(cat_i_manager.class_num):
            label_name = cat_i_manager.id_to_label([idx])[0]
            assert (label_name in cat_keyword_embedding_map), 'NOT FOUND IN cat keywords embed map: %s !' % label_name
            embedding_vecs.extend(cat_keyword_embedding_map[label_name])

        np_embedding_vecs = np.asarray(embedding_vecs, dtype='float32')
        print('LOAD DONE! %s' % fname)
        print('keywords embedding SHAPE is %s' % str(np_embedding_vecs.shape))

        return np_embedding_vecs

    @property
    def vocab_size(self):
        return len(self.vocab)

    def load_data(self, loader, data_file, tfrecord_file, rebuild):
        if not callable(loader):
            raise Exception('Parameter %s is not callable!' % str(loader))

        if os.path.exists(tfrecord_file):
            if rebuild:
                os.remove(tfrecord_file)
            else:
                print('Use existing tfrecord file: %s.' % tfrecord_file)
                return

        print('Start to rebuild tfrecord file: %s.' % tfrecord_file)
        tf_writer = tf.python_io.TFRecordWriter(tfrecord_file)

        docids, texts, c1lb_list, c2lb_list, c3lb_list, sentence_lens, word_masks \
            = loader(data_file, valid_labels=self.labels_manager_c1.labels(), sentence_len=self.sentence_len)

        c1id_list = self.labels_manager_c1.label_to_id(c1lb_list)
        c2id_list = self.labels_manager_c2.label_to_id(c2lb_list)
        c3id_list = self.labels_manager_c3.label_to_id(c3lb_list)

        total_examples = len(docids)
        for i, (docid, words, c1_id, c2_id, c3_id, sentence_len, word_mask) in enumerate(
                zip(docids, texts, c1id_list, c2id_list, c3id_list, sentence_lens, word_masks)):
            if i % 10000 == 0:
                print('Transforming %d example of %d.' % (i, total_examples))

            token_ids = self.vocab.text_to_ids(words, self.sentence_len)
            c1_id_onehot = [0] * self.cat1_total
            c1_id_onehot[int(c1_id)] = 1

            c2_id_onehot = [0] * self.cat2_total
            c2_id_onehot[int(c2_id)] = 1

            c3_id_onehot = [0] * self.cat3_total
            c3_id_onehot[int(c3_id)] = 1

            features = collections.OrderedDict()
            features['docid'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[docid.encode()]))
            features['input_ids'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=token_ids))
            features['label_c1_id_onehot'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=c1_id_onehot))
            features['label_c2_id_onehot'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=c2_id_onehot))
            features['label_c3_id_onehot'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=c3_id_onehot))

            features['input_sentence_len'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=sentence_len))
            features['word_mask'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=word_mask))

            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features))
            tf_writer.write(tf_example.SerializeToString())

        print('Finished transforming all %d examples.' % total_examples)
        return docids

    def create_dataset(self, tfrecord, batch_size, shuf_buffer_size=5000):
        def _parse_example(serial_example):
            name_to_features = {
                "docid": tf.FixedLenFeature([], tf.string),
                "input_ids": tf.FixedLenFeature([self.sentence_len], tf.int64),
                "label_c1_id_onehot": tf.FixedLenFeature([self.cat1_total], tf.int64),
                "label_c2_id_onehot": tf.FixedLenFeature([self.cat2_total], tf.int64),
                "label_c3_id_onehot": tf.FixedLenFeature([self.cat3_total], tf.int64),
                "input_sentence_len": tf.FixedLenFeature([1], tf.int64),
                "word_mask": tf.FixedLenFeature([self.sentence_len], tf.float32)
            }
            example = tf.parse_single_example(serial_example, name_to_features)
            return example['docid'], example['input_ids'], example['label_c1_id_onehot'], example['label_c2_id_onehot'],\
                   example['label_c3_id_onehot'], example['input_sentence_len'], example['word_mask']

        dataset = tf.data.TFRecordDataset(tfrecord)
        dataset = dataset.map(_parse_example)
        if shuf_buffer_size > 0:
            print('SHUFFLE DATASET buffer_size is %d' % shuf_buffer_size)
            dataset = dataset.shuffle(shuf_buffer_size)
        dataset = dataset.batch(batch_size)
        return dataset

class LabelsTrans(object):

    def __init__(self, c3labels=None, c1labels=None, c2flag=None):
        # if use c3 to c2, please set c2flag True
        if c3labels is None:
            raise Exception('INVALID c3labels: you should initialize a data_helper.ClassLabels instance firstly.')
        elif c1labels is None:
            raise Exception('INVALID c1labels: you should initialize a data_helper.ClassLabels instance firstly.')

        self.c3label2c1label = {}
        self.c3id2c1id = {}

        for _c3 in c3labels.labels():
            _c1 = _c3.split('_')[0]
            if c2flag:
                _c1 = '_'.join(_c3.split('_')[0:2])  # actually here _c1 is c2 label

            if _c1 in c1labels.labels():
                self.c3label2c1label[_c3] = _c1
                _c3id = c3labels.label_to_id([_c3])[0]
                _c1id = c1labels.label_to_id([_c1])[0]
                self.c3id2c1id[_c3id] = _c1id
            else:
                raise Exception('the c1 of some c3 is not found in c1labelfile: ', _c1)
        print('INITIALIZE LABELSTRANS IS DONE!')

    def c3label_to_c1label(self, c3labels):
        c1labels = []
        for l in c3labels:
            assert (l in self.c3label2c1label), 'Undefined: %s !' % l
            c1labels.append(self.c3label2c1label[l])
        return c1labels

    def c3id_to_c1id(self, c3ids):
        c1ids = []
        for _id in c3ids:
            assert (_id in self.c3id2c1id), 'Unfound c3 id: %s ' % str(_id)
            c1ids.append(self.c3id2c1id[_id])
        return c1ids
