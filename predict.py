from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os
import os.path as osp
import argparse
import numpy as np
import time

import tensorflow as tf
import util

parser = argparse.ArgumentParser(description='pronoun resolution prediction')
parser.add_argument('model', type=str,
                    help='model name to evaluate')
parser.add_argument('--step', type=str, default='max',
                    help='global steps to restore from')
parser.add_argument('--split', type=str, default='test',
                    help='split to evaluate, test or val')
parser.add_argument('--input_dir', type=str, default='data',
                    help='input dir')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='log dir')
parser.add_argument('--output_dir', type=str, default='output',
                    help='output dir')


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get_prp_ant_in_pool(pronoun_info, top_antecedents, top_antecedent_scores, top_span_starts, top_span_ends):
  for pronoun_example in pronoun_info:
    if pronoun_example['reference_type'] != 0:
      continue
    tmp_pronoun_index = pronoun_example['current_pronoun'][0]
    for span_id, (start, end) in enumerate(zip(top_span_starts, top_span_ends)):
      if tmp_pronoun_index == start and tmp_pronoun_index == end:
        # if match, must coref in pool
        # top 10 highest score
        predicted_nn = np.argsort(top_antecedent_scores[span_id])[::-1][:10]
        # only keep nn with score > 0:
        if 0 in predicted_nn:
          index_0 = np.where(predicted_nn == 0)[0][0]
          predicted_nn = predicted_nn[:index_0]
        else:
          index_0 = None
        # top 10 predicted nn index
        if len(predicted_nn) > 0:
          predicted_nn = list(top_antecedents[span_id][predicted_nn - 1])
        else:
          predicted_nn = []
        if index_0 is not None:
          predicted_nn.append(-1)

        pronoun_example['predicted_nn'] = predicted_nn
        break
  return pronoun_info


if __name__ == "__main__":
  args = parser.parse_args()
  if len(sys.argv) == 1:
    sys.argv.append(args.model)
  else:
    sys.argv[1] = args.model
  config = util.initialize_from_env()
  config["log_dir"] = os.path.join(args.log_dir, args.model)

  model = util.get_model(config)

  input_filename = args.split + f'.vispro.pool.1.1.bert.{config["max_segment_len"]}.jsonlines'
  output_filename = args.split + '.vispro.pool.1.1.prediction.jsonlines'
  input_filename = osp.join(args.input_dir, input_filename)
  output_filename = osp.join(args.output_dir, args.model, output_filename)

  # Create output dir
  output_dir = osp.split(output_filename)[0]
  if not osp.exists(output_dir):
    os.makedirs(output_dir)

  configtf = tf.ConfigProto()
  configtf.gpu_options.allow_growth = True
  with tf.Session(config=configtf) as session:
    model.restore(session, args.step)    

    time_start = time.time()
    with open(output_filename, "w") as output_file:
      with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
          example = json.loads(line)
          tensorized_example = model.tensorize_example(example, is_training=False)
          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
          outputs = session.run(model.predictions, feed_dict=feed_dict)
          top_antecedents, top_antecedent_scores, top_antecedents_pool, top_antecedent_scores_pool, topic_prediction, topic_loss = outputs

          top_span_starts = tensorized_example[5]
          top_span_ends = tensorized_example[6]
          # clusters in dialog
          predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
          example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
          # save prediction of pool
          coref_in_pool = tensorized_example[-2]
          top_span_starts_pool = top_span_starts[coref_in_pool]
          top_span_ends_pool = top_span_ends[coref_in_pool]
          example["pronoun_info"] = get_prp_ant_in_pool(example["pronoun_info"], top_antecedents_pool, top_antecedent_scores_pool, top_span_starts_pool, top_span_ends_pool)

          output_file.write(json.dumps(example, cls=MyEncoder))
          output_file.write("\n")
          if example_num % 100 == 0:
            print("Decoded {} examples.".format(example_num + 1))
    time_end = time.time()
    print(f'Average running time: {(time_end-time_start)/(example_num + 1):.4f}s')
  print("Output saved to " + output_filename)
