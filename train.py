#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import time
import sys

import tensorflow as tf
import util

import argparse
parser = argparse.ArgumentParser(description='train model')
parser.add_argument('model', type=str,
                    help='model name to train')
parser.add_argument('--log_dir', type=str, default='logs', 
                    help='dir of training log')


def set_log_file(fname):
  # set log file
  # simple tricks for duplicating logging destination in the logging module such as:
  # logging.getLogger().addHandler(logging.FileHandler(filename))
  # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
  # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
  # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything

  # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
  tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
  os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
  os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
  args = parser.parse_args()
  if len(sys.argv) == 1:
    sys.argv.append(args.model)
  else:
    sys.argv[1] = args.model  

  log_dir = os.path.join(args.log_dir, args.model)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)
  log_file = os.path.join(log_dir, 'train.log')
  set_log_file(log_file)
  
  config = util.initialize_from_env()
  config["log_dir"] = os.path.join(args.log_dir, args.model)

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]

  model = util.get_model(config)
  saver = tf.train.Saver(max_to_keep=1)

  max_steps = config['num_epochs'] * config['num_docs']
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0

  config_tf = tf.ConfigProto()
  config_tf.gpu_options.allow_growth = True

  with tf.Session(config=config_tf) as session:
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(session, ckpt.model_checkpoint_path)
      max_step, max_f1 = session.run([model.global_step, model.max_eval_f1])
      print('Restoring from max f1 of %.2f' % max_f1)

    initial_time = time.time()
    while True:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, eval_f1 = model.evaluate(session, tf_global_step)
        _ = session.run(model.update_max_f1)

        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        print("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))

        if tf_global_step >= max_steps:
          break
