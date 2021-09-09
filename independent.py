from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import json
import copy
import threading
import numpy as np
import tensorflow as tf

import util
import coref_ops
import metrics
import optimization
from bert import modeling
from bert import tokenization
from pytorch_to_tf import load_from_pytorch_checkpoint


class CorefModel(object):
  def __init__(self, config):
    self.config = config
    self.max_segment_len = config['max_segment_len']
    self.max_span_width = config["max_span_width"]
    self.gold = {}
    self.eval_data = None # Load eval data lazily.
    self.tokenizer = tokenization.FullTokenizer(
                  vocab_file=config['vocab_file'], do_lower_case=False)
    self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])

    # topic 
    self.use_topic = self.config["use_lda_topics"]
    if self.use_topic:
      self.topic_labels = np.load(self.config["lda_label_path"], allow_pickle=True)['labels'][()]

    # pool
    examples = [json.loads(line) for line in open(self.config["nn_pool_path"])]
    sentences = [e["sentences"][0] for e in examples]
    text_len = np.array([len(s) for s in sentences])
    max_sentence_length = max(text_len)

    input_ids, input_mask = [], []
    for i, sentence in enumerate(sentences):
      sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
      sent_len = len(sent_input_ids)
      sent_input_mask = [1] * sent_len
      sent_input_ids.extend([0] * (max_sentence_length - sent_len))
      sent_input_mask.extend([0] * (max_sentence_length - sent_len))
      input_ids.append(sent_input_ids)
      input_mask.append(sent_input_mask)
    input_ids = np.array(input_ids)
    input_mask = np.array(input_mask)
    self.input_ids_pool = input_ids
    self.input_mask_pool = input_mask
    self.text_len_pool = text_len
    self.pool_size = len(examples)

    input_props = []
    input_props.append((tf.int32, [None, None])) # input_ids.
    input_props.append((tf.int32, [None, None])) # input_mask
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.int32, [None, None])) # Speaker IDs.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None])) # Cluster ids.
    # pool
    input_props.append((tf.int32, [self.pool_size, None])) # input_ids pool
    input_props.append((tf.int32, [self.pool_size, None])) # input_mask pool
    input_props.append((tf.int32, [self.pool_size])) # Text lengths pool
    input_props.append((tf.int32, [None])) # Cluster ids pool
    input_props.append((tf.bool, [None, self.pool_size])) # Include indicator
    input_props.append((tf.bool, [None])) # Coref in pool
    input_props.append((tf.float32, [self.config["num_lda_topics"]])) # Lda topic labels.


    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    # bert stuff
    tvars = tf.trainable_variables()
    # If you're using TF weights only, tf_checkpoint and init_checkpoint can be the same
    # Get the assignment map from the tensorflow checkpoint. Depending on the extension, use TF/Pytorch to load weights.
    assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config['tf_checkpoint'])
    init_from_checkpoint = tf.train.init_from_checkpoint if config['init_checkpoint'].endswith('ckpt') else load_from_pytorch_checkpoint
    init_from_checkpoint(config['init_checkpoint'], assignment_map)
    print("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      # init_string)
      print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

    num_train_steps = int(
                    self.config['num_docs'] * self.config['num_epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    self.global_step = tf.train.get_or_create_global_step()
    self.max_eval_f1 = tf.Variable(0.0, name="max_eval_f1", trainable=False)
    self.train_op = optimization.create_custom_optimizer(tvars,
                      self.loss, self.config['bert_learning_rate'], self.config['task_learning_rate'],
                      num_train_steps, num_warmup_steps, False, self.global_step, freeze=-1,
                      task_opt=self.config['task_optimizer'], eps=config['adam_eps'])

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    def _enqueue_loop():
      while True:
        global_step = session.run(self.global_step)
        random.seed(self.config["random_seed"] + global_step)
        random.shuffle(train_examples)
        if self.config['single_example']:
          for example in train_examples:
            tensorized_example = self.tensorize_example(example, is_training=True)
            feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
            session.run(self.enqueue_op, feed_dict=feed_dict)
        else:
          examples = []
          for example in train_examples:
            tensorized = self.tensorize_example(example, is_training=True)
            if type(tensorized) is not list:
              tensorized = [tensorized]
            examples += tensorized
          random.shuffle(examples)
          print('num examples', len(examples))
          for example in examples:
            feed_dict = dict(zip(self.queue_input_tensors, example))
            session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session, step='max'):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() ]
    saver = tf.train.Saver(vars_to_restore)
    if step in ['max', 'pretrained']:
      path = "model." + step + ".ckpt"
    else:
      path = "model-" + step
    if self.config["restore_path"] != "":
      checkpoint_path = os.path.join(os.path.split(self.config["log_dir"])[0], self.config["restore_path"], path)
    else:
      checkpoint_path = os.path.join(self.config["log_dir"], path)
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_span_labels(self, tuples, label_dict):
    if len(tuples) > 0:
      starts, ends, labels = zip(*tuples)
    else:
      starts, ends, labels = [], [], []
    return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

  def get_speaker_dict(self, speakers):
    speaker_dict = {'UNK': 0, '[SPL]': 1}
    for s in speakers:
      if s not in speaker_dict and len(speaker_dict) < self.config['max_num_speakers']:
        speaker_dict[s] = len(speaker_dict)
    return speaker_dict


  def tensorize_example(self, example, is_training):
    clusters = copy.deepcopy((example["clusters"]))
    doc_key = example["doc_key"]

    # return: gold_mentions sorted, cluster_ids, cluster_ids_pool, coref_in_pool, include_indicator
    clusters_mention_only = []
    clusters_coref_in_pool = []
    clusters_exclude_nns = []
    cluster_ids_pool = np.zeros(self.pool_size)
    for cluster_id, cluster in enumerate(clusters):
      cluster_mention_only = []
      cluster_coref_in_pool = False
      cluster_exclude_nns = []
      for mention in cluster:
        if isinstance(mention, list):
          cluster_mention_only.append(mention)
        elif isinstance(mention, dict):
          cluster_coref_in_pool = True
          nn_ant = [mention['nn']]
          for key in ['synonym', 'hypernym']:
            nn_ant.extend(mention[key])
            cluster_exclude_nns.extend(mention[key])
          cluster_exclude_nns.extend(mention['hyponym'])
          for nn in nn_ant:
            cluster_ids_pool[nn] = cluster_id + 1
      clusters_coref_in_pool.append(cluster_coref_in_pool)
      clusters_mention_only.append(cluster_mention_only)
      clusters_exclude_nns.append(cluster_exclude_nns)
    # mentions from cluster
    gold_mentions = [tuple(m) for m in util.flatten(clusters_mention_only)]
    # other mentions from candidate NPs and non-referential pronouns
    for prp in example["pronoun_info"]:
      if tuple(prp["current_pronoun"]) not in gold_mentions:
        gold_mentions.append(tuple(prp["current_pronoun"]))
    for mention in example["pronoun_info"][-1]["candidate_NPs"]:
      if tuple(mention) not in gold_mentions:
        gold_mentions.append(tuple(mention))
    gold_mentions = sorted(gold_mentions)
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    coref_in_pool = np.zeros(len(gold_mentions), dtype=np.bool)
    include_indicator = np.ones([len(gold_mentions), self.pool_size], dtype=np.bool)
    for cluster_id, cluster in enumerate(clusters_mention_only):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
        if clusters_coref_in_pool[cluster_id]:
          coref_in_pool[gold_mention_map[tuple(mention)]] = True
          nns_to_exclude = np.unique(np.array(clusters_exclude_nns[cluster_id]))
          if len(nns_to_exclude) > 0:
            include_indicator[gold_mention_map[tuple(mention)]][nns_to_exclude] = False

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = example["speakers"]
    speaker_dict = self.get_speaker_dict(util.flatten(speakers) + ['caption'])

    text_len = np.array([len(s) for s in sentences])
    max_sentence_length = max(text_len)

    input_ids, input_mask, speaker_ids = [], [], []
    for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
      sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
      sent_len = len(sent_input_ids)
      sent_input_mask = [1] * sent_len
      sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
      sent_input_ids.extend([0] * (max_sentence_length - sent_len))
      sent_input_mask.extend([0] * (max_sentence_length - sent_len))
      sent_speaker_ids.extend([0] * (max_sentence_length - sent_len))
      input_ids.append(sent_input_ids)
      speaker_ids.append(sent_speaker_ids)
      input_mask.append(sent_input_mask)
    input_ids = np.array(input_ids)
    input_mask = np.array(input_mask)
    speaker_ids = np.array(speaker_ids)
    assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

    if self.use_topic:
      topic_label = self.topic_labels[doc_key]
    else:
      topic_label = np.zeros([self.config["num_lda_topics"]])

    self.gold[doc_key] = example["clusters"]

    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
    example_tensors = (input_ids, input_mask, text_len, speaker_ids, is_training, gold_starts, gold_ends, cluster_ids, self.input_ids_pool, self.input_mask_pool, self.text_len_pool, cluster_ids_pool, include_indicator, coref_in_pool, topic_label)

    return example_tensors

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def coarse_to_fine_pruning(self, top_span_emb, c):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    fast_antecedent_scores = self.get_fast_antecedent_scores(top_span_emb) # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask)) # [k, k]
    if self.config['use_prior']:
      antecedent_distance_buckets = self.bucket_distance(antecedent_offsets) # [k, c]
      with tf.variable_scope("antecedent_distance", reuse=tf.AUTO_REUSE):
        distance_scores = util.projection(tf.nn.dropout(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), self.dropout), 1, initializer=tf.truncated_normal_initializer(stddev=0.02)) #[10, 1]
      antecedent_distance_scores = tf.gather(tf.squeeze(distance_scores, 1), antecedent_distance_buckets) # [k, c]
      fast_antecedent_scores += antecedent_distance_scores

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def coarse_to_fine_pruning_pool(self, top_span_emb, pool_span_emb, antecedent_offsets_pool, c, include_indicator_pool=None):
    p = util.shape(pool_span_emb, 0)
    antecedent_offsets = tf.tile(tf.expand_dims(antecedent_offsets_pool, 1), [1, p]) # [k, p]
    fast_antecedent_scores = self.get_fast_antecedent_scores_pool(top_span_emb, pool_span_emb) # [k, p]
    if include_indicator_pool is not None:
        fast_antecedent_scores += tf.log(tf.to_float(include_indicator_pool)) # [k, c]
    if self.config['use_prior']:
      antecedent_distance_buckets = self.bucket_distance(antecedent_offsets) # [k, p]
      with tf.variable_scope("antecedent_distance", reuse=tf.AUTO_REUSE):
        distance_scores = util.projection(tf.nn.dropout(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), self.dropout), 1, initializer=tf.truncated_normal_initializer(stddev=0.02)) #[10, 1]
      antecedent_distance_scores = tf.gather(tf.squeeze(distance_scores, 1), antecedent_distance_buckets) # [k, p]
      fast_antecedent_scores += antecedent_distance_scores

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_fast_antecedent_scores, top_antecedent_offsets   


  def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, is_training, gold_starts, gold_ends, cluster_ids, input_ids_pool, input_mask_pool, text_len_pool, cluster_ids_pool, include_indicator, coref_in_pool, topic_label):

    model = modeling.BertModel(
      config=self.bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      use_one_hot_embeddings=False,
      scope='bert')
    all_encoder_layers = model.get_all_encoder_layers()
    mention_doc = model.get_sequence_output()

    if self.use_topic:
      doc_feat = tf.reduce_mean(mention_doc[:, 0, :], axis=0, keepdims=True) # [1, emb=768]

    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

    mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask) # [k, emb]

    candidate_starts = gold_starts # [num_candidates]
    candidate_ends = gold_ends # [num_candidates]

    candidate_cluster_ids = cluster_ids # [num_candidates]

    candidate_span_emb = self.get_span_emb(mention_doc, mention_doc, candidate_starts, candidate_ends) # [num_candidates, emb]

    k = util.shape(candidate_starts, 0)    

    # topic prediction module
    if self.use_topic:
      # doc_feat: [1, emb]
      # projection
      with tf.variable_scope('topic_embedding'):
        if self.config["topic_hidden_layers"] > 0:
          topic_emb = util.ffnn(doc_feat, num_hidden_layers=self.config["topic_hidden_layers"], hidden_size=self.config["topic_hidden_size"], output_size=self.config["topic_hidden_size"], dropout=self.dropout)
        else:
          topic_emb = util.projection(doc_feat, self.config["topic_hidden_size"]) # [1, emb]
      if self.config["add_topic_loss"]:
        # prediction
        with tf.variable_scope('topic_prediction'):
          topic_prediction = util.projection(topic_emb, self.config["num_lda_topics"]) # [1, num_topics]
        if self.config["topic_loss_func"] == "l2":
          topic_prediction = tf.nn.softmax(topic_prediction, 1) # [1, num_topics]
          # loss
          topic_loss = self.config["topic_loss_weight"] * tf.nn.l2_loss(topic_prediction - tf.expand_dims(topic_label, 0))
      else:
        topic_loss = tf.zeros(1)
        topic_prediction = tf.zeros([1, self.config["num_lda_topics"]])      

    c = tf.minimum(self.config["max_top_antecedents"], k)

    top_span_starts = candidate_starts # [k]
    top_span_ends = candidate_ends # [k]
    top_span_emb = candidate_span_emb # [k, emb]
    top_span_cluster_ids = candidate_cluster_ids # [k]

    dummy_scores = tf.zeros([k, 1]) # [k, 1]
    top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, c)

    if self.config['use_metadata']:
      speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)
      top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]i
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
    else:
      same_speaker = None

    if self.config['fine_grained']:
      for i in range(self.config["coref_depth"]):
        with tf.variable_scope("coref_layer", reuse=(i > 0)):
          if self.use_topic:
            # topic_emb: [1, emb]
            with tf.variable_scope("project_span_to_topic", reuse=tf.AUTO_REUSE):
              projected_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, self.config["topic_hidden_size"]), self.dropout) # [k, emb]
            pair_emb = tf.concat([projected_top_span_emb, tf.tile(topic_emb, [k, 1]), projected_top_span_emb * topic_emb], 1) # [k, emb]
            # debug
            with tf.variable_scope("topic_score", reuse=tf.AUTO_REUSE):
              span_topic_scores = util.projection(pair_emb, 1) # [k, 1]
            ant_topic_scores = tf.gather(span_topic_scores, top_antecedents) # [k, c, 1]
            ant_topic_scores = tf.squeeze(ant_topic_scores, 2) # [k, c]

            topic_scores = span_topic_scores + ant_topic_scores # [k, c]
          else:
            topic_scores = tf.zeros([k, c])

          top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]
          top_antecedent_scores = top_fast_antecedent_scores + topic_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, same_speaker) # [k, c]

          if self.config["coref_depth"] > 1:
            top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
            top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
            attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1) # [k, emb]
            with tf.variable_scope("f"):
              f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1), util.shape(top_span_emb, -1))) # [k, emb]
              top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb # [k, emb]
    else:
        top_antecedent_scores = top_fast_antecedent_scores

    top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]

    top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]
    top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask))) # [k, c]
    same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
    non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
    pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
    dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) # [k, 1]
    top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]
    loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]
    loss = tf.reduce_sum(loss) # []

    def get_predicion_and_loss_from_pool():
      # calculate similarity between mentions and nn in pools
      # pool
      model = modeling.BertModel(
        config=self.bert_config,
        is_training=is_training,
        input_ids=input_ids_pool,
        input_mask=input_mask_pool,
        use_one_hot_embeddings=False,
        scope='bert')
      all_encoder_layers = model.get_all_encoder_layers()
      mention_doc_pool = model.get_sequence_output() # [k, sent_len, emb]
      mention_doc_pool = self.flatten_emb_by_sentence(mention_doc_pool, input_mask_pool) # [k, emb]
      pool_ends = tf.cumsum(text_len_pool) - 2
      pool_starts = pool_ends - text_len_pool + 3
      pool_span_emb = self.get_span_emb(mention_doc_pool, mention_doc_pool, pool_starts, pool_ends) # [pool_size, emb]

      antecedent_offsets_pool = tf.range(k)
      antecedent_offsets_pool = tf.boolean_mask(antecedent_offsets_pool, coref_in_pool) # [k]
      top_span_starts_pool = tf.boolean_mask(candidate_starts, coref_in_pool) # [k]
      top_span_ends_pool = tf.boolean_mask(candidate_ends, coref_in_pool) # [k]
      top_span_emb_pool = tf.boolean_mask(candidate_span_emb, coref_in_pool) # [k, emb]
      top_span_cluster_ids_pool = tf.boolean_mask(candidate_cluster_ids, coref_in_pool) # [k]
      include_indicator_pool = tf.boolean_mask(include_indicator, coref_in_pool) #[k, pool_size]

      k_pool = util.shape(top_span_starts_pool, 0)
      c_pool = self.config["max_top_antecedents_pool"]

      dummy_scores_pool = tf.zeros([k_pool, 1]) # [k, 1]
      if self.config["exclude_synonyms"]:
        top_antecedents_pool, top_fast_antecedent_scores_pool, top_antecedent_offsets_pool = self.coarse_to_fine_pruning_pool(top_span_emb_pool, pool_span_emb, antecedent_offsets_pool, c_pool, include_indicator_pool) # [k, c]
      else:
        top_antecedents_pool, top_fast_antecedent_scores_pool, top_antecedent_offsets_pool = self.coarse_to_fine_pruning_pool(top_span_emb_pool, pool_span_emb, antecedent_offsets_pool, c_pool) # [k, c]

      if self.config['use_metadata']:
        same_speaker_pool = tf.zeros([k_pool, c_pool], dtype=tf.bool) # [k, c]
      else:
        same_speaker_pool = None

      if self.config['use_fine_grained_in_pool']:
        # use top_span_embeddings that include fine grained information of mentions in the same clusters
        top_span_emb_pool = tf.boolean_mask(top_span_emb, coref_in_pool) # [k, emb]

      if self.config['fine_grained']:
        with tf.variable_scope("coref_layer", reuse=tf.AUTO_REUSE):
          if self.use_topic:
            # topic_emb: [1, emb]
            with tf.variable_scope("project_span_to_topic", reuse=tf.AUTO_REUSE):
              projected_top_span_emb_pool = tf.nn.dropout(util.projection(pool_span_emb, self.config["topic_hidden_size"]), self.dropout) # [p, emb]
            pair_emb_pool = tf.concat([projected_top_span_emb_pool, tf.tile(topic_emb, [self.pool_size, 1]), projected_top_span_emb_pool * topic_emb], 1) # [p, emb]
            # debug
            with tf.variable_scope("topic_score", reuse=tf.AUTO_REUSE):
              pool_topic_scores = util.projection(pair_emb_pool, 1) # [p, 1]
            ant_topic_scores_pool = tf.gather(pool_topic_scores, top_antecedents_pool) # [k, c, 1]
            ant_topic_scores_pool = tf.squeeze(ant_topic_scores_pool, 2) # [k, c]

            topic_scores_pool = tf.boolean_mask(span_topic_scores, coref_in_pool) + ant_topic_scores_pool # [k, c]
          else:
            topic_scores_pool = tf.zeros([k_pool, c_pool])

          top_antecedent_emb_pool = tf.gather(pool_span_emb, top_antecedents_pool) # [k, c, emb]
          top_antecedent_scores_pool = top_fast_antecedent_scores_pool + topic_scores_pool + self.get_slow_antecedent_scores(top_span_emb_pool, top_antecedents_pool, top_antecedent_emb_pool, top_antecedent_offsets_pool, same_speaker_pool) # [k, c]
      else:
          top_antecedent_scores_pool = top_fast_antecedent_scores_pool

      top_antecedent_scores_pool = tf.concat([dummy_scores_pool, top_antecedent_scores_pool], 1) # [k, c + 1]

      top_antecedent_cluster_ids_pool = tf.gather(cluster_ids_pool, top_antecedents_pool) # [k, c]
      same_cluster_indicator_pool = tf.equal(top_antecedent_cluster_ids_pool, tf.expand_dims(top_span_cluster_ids_pool, 1)) # [k, c]
      non_dummy_indicator_pool = tf.expand_dims(top_span_cluster_ids_pool > 0, 1) # [k, 1]
      pairwise_labels_pool = tf.logical_and(same_cluster_indicator_pool, non_dummy_indicator_pool) # [k, c]
      dummy_labels_pool = tf.logical_not(tf.reduce_any(pairwise_labels_pool, 1, keepdims=True)) # [k, 1]
      top_antecedent_labels_pool = tf.concat([dummy_labels_pool, pairwise_labels_pool], 1) # [k, c + 1]
      loss_pool = self.softmax_loss(top_antecedent_scores_pool, top_antecedent_labels_pool) # [k]
      loss_pool = tf.reduce_sum(loss_pool) # []

      return top_antecedents_pool, top_antecedent_scores_pool, loss_pool

    def zero_prediction_and_loss_from_pool():
      return tf.zeros(1, tf.int32), tf.zeros(1, tf.float32), 0.0

    top_antecedents_pool, top_antecedent_scores_pool, loss_pool = tf.cond(tf.reduce_any(coref_in_pool), lambda: get_predicion_and_loss_from_pool(), lambda: zero_prediction_and_loss_from_pool())

    loss += loss_pool
    if self.use_topic:
      loss += topic_loss    

    outputs = [top_antecedents, top_antecedent_scores, top_antecedents_pool, top_antecedent_scores_pool]
    if self.use_topic:
      outputs.extend([topic_prediction, topic_loss])
    else:
      outputs.extend([tf.zeros([1, 1]), tf.zeros(1)])

    return outputs, loss


  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = 1 + span_ends - span_starts # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      with tf.variable_scope("use_features", reuse=tf.AUTO_REUSE):
        span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
      head_attn_reps = tf.matmul(mention_word_scores, context_outputs) # [K, T]
      span_emb_list.append(head_attn_reps)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb # [k, emb]

  def get_mention_scores(self, span_emb, span_starts, span_ends):
      with tf.variable_scope("mention_scores"):
        span_scores = util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]
      if self.config['use_prior']:
        span_width_emb = tf.get_variable("span_width_prior_embeddings", [self.config["max_span_width"], self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)) # [W, emb]
        span_width_index = span_ends - span_starts # [NC]
        with tf.variable_scope("width_scores"):
          width_scores =  util.ffnn(span_width_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [W, 1]
        width_scores = tf.gather(width_scores, span_width_index)
        span_scores += width_scores
      return span_scores


  def get_width_scores(self, doc, starts, ends):
    distance = ends - starts
    span_start_emb = tf.gather(doc, starts)
    hidden = util.shape(doc, 1)
    with tf.variable_scope('span_width'):
      span_width_emb = tf.gather(tf.get_variable("start_width_embeddings", [self.config["max_span_width"], hidden], initializer=tf.truncated_normal_initializer(stddev=0.02)), distance) # [W, emb]
    scores = tf.reduce_sum(span_start_emb * span_width_emb, axis=1)
    return scores


  def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
      num_words = util.shape(encoded_doc, 0) # T
      num_c = util.shape(span_starts, 0) # NC
      doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1]) # [K, T]
      mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1), doc_range <= tf.expand_dims(span_ends, 1)) #[K, T]
      with tf.variable_scope("mention_word_attn", reuse=tf.AUTO_REUSE):
        word_attn = tf.squeeze(util.projection(encoded_doc, 1, initializer=tf.truncated_normal_initializer(stddev=0.02)), 1)
      mention_word_attn = tf.nn.softmax(tf.log(tf.to_float(mention_mask)) + tf.expand_dims(word_attn, 0))
      return mention_word_attn


  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, same_speaker):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), tf.to_int32(same_speaker)) # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb=7012]

    with tf.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]

    return slow_antecedent_scores # [k, c]

  def get_fast_antecedent_scores(self, top_span_emb):
    with tf.variable_scope("src_projection", reuse=tf.AUTO_REUSE):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

  def get_fast_antecedent_scores_pool(self, top_span_emb, pool_span_emb):
    with tf.variable_scope("src_projection", reuse=tf.AUTO_REUSE):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(pool_span_emb, self.dropout) # [p, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k] or [k, p]    

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))


  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index, (i, predicted_index)
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      print("Loaded {} eval examples.".format(len(self.eval_data)))

  def evaluate(self, session, global_step=None, official_stdout=False, keys=None, eval_mode=False):
    self.load_eval_data()

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    pr_coref_evaluator_seen = metrics.PrCorefEvaluatorSeen()
    pr_coref_evaluator_unseen = metrics.PrCorefEvaluatorUnseen()
    losses = []
    if self.use_topic and self.config["add_topic_loss"]:
      topic_loss_total = 0

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      top_span_starts = tensorized_example[5]
      top_span_ends = tensorized_example[6]
      coref_in_pool = tensorized_example[-2]
      topic_label = tensorized_example[-1]
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      loss, outputs = session.run([self.loss, self.predictions], feed_dict=feed_dict)
      top_antecedents, top_antecedent_scores, top_antecedents_pool, top_antecedent_scores_pool, topic_prediction, topic_loss = outputs
      losses.append(loss)
      predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
      gold_clusters = []
      for gc in example["clusters"]:
        gold_cluster = []
        for m in gc:
          if isinstance(m, list):
            gold_cluster.append(tuple(m))
        if len(gold_cluster) > 1:
          gold_clusters.append(tuple(gold_cluster))
      coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, coref_evaluator)
      pr_coref_evaluator_seen.update(coref_predictions[example["doc_key"]], example["pronoun_info"], example["sentences"])
      if np.any(coref_in_pool):
        pr_coref_evaluator_unseen.update(example["pronoun_info"], top_antecedents_pool, top_antecedent_scores_pool, top_span_starts[coref_in_pool], top_span_ends[coref_in_pool])
      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

      if self.use_topic:
        topic_loss_total += topic_loss

    summary_dict = {}

    losses = np.mean(losses)
    summary_dict["Total loss"] = losses
    print("Total loss: {:.6f}".format(float(losses)))
    if self.use_topic:
      topic_loss_total /= len(self.eval_data)
      summary_dict["Topic Prediction " + self.config['topic_loss_func'] + " loss"] = topic_loss_total
      print("Topic Prediction " + self.config['topic_loss_func'] + " loss: {:.6f}".format(float(topic_loss_total)))

    p,r,f = coref_evaluator.get_prf()
    summary_dict["Coref average F1 (py)"] = f
    print("Coref average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["Coref average precision (py)"] = p
    print("Coref average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Coref average recall (py)"] = r
    print("Coref average recall (py): {:.2f}%".format(r * 100))

    p,r,f = pr_coref_evaluator_seen.get_prf()

    summary_dict["Pronoun Coref average F1 (py)"] = f
    print("Pronoun Coref average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["Pronoun Coref average precision (py)"] = p
    print("Pronoun Coref average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Pronoun Coref average recall (py)"] = r
    print("Pronoun Coref average recall (py): {:.2f}%".format(r * 100))

    recall = pr_coref_evaluator_unseen.get_recall()
    summary_dict["Recall@1 for seen nn"] = recall['seen']['r@1']['nn']
    print("Recall@1 for seen nn: {:.2f}%".format(recall['seen']['r@1']['nn'] * 100))
    summary_dict["Recall@10 for seen nn, syn, hyper"] = recall['seen']['r@10']['nn_syn_hyper']
    print("Recall@10 for seen nn, syn, hyper: {:.2f}%".format(recall['seen']['r@10']['nn_syn_hyper'] * 100))
    print(recall['seen'])
    summary_dict["Recall@1 for unseen nn"] = recall['unseen']['r@1']['nn']
    print("Recall@1 for unseen nn: {:.2f}%".format(recall['unseen']['r@1']['nn'] * 100))
    summary_dict["Recall@10 for unseen nn, syn, hyper"] = recall['unseen']['r@10']['nn_syn_hyper']
    print("Recall@10 for unseen nn, syn, hyper: {:.2f}%".format(recall['unseen']['r@10']['nn_syn_hyper'] * 100))
    print(recall['unseen'])
    
    average_f1 = recall['unseen']['r@1']['nn']

    max_eval_f1 = tf.maximum(self.max_eval_f1, average_f1)
    self.update_max_f1 = tf.assign(self.max_eval_f1, max_eval_f1)

    return util.make_summary(summary_dict), average_f1
