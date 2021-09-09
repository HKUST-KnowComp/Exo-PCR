#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os.path as osp
import metrics
from collections import OrderedDict

parser = argparse.ArgumentParser(description='evaluate pronoun resolution')
parser.add_argument('model', type=str,
                    help='model name to evaluate')
parser.add_argument('--split', type=str, default='test',
                    help='split to evaluate, test or val')
parser.add_argument('--output_dir', type=str, default='output',
                    help='output dir')


def main(args):
    evaluate_file = args.split + f'.vispro.pool.1.1.prediction.jsonlines'
    evaluate_file = osp.join(args.output_dir, args.model, evaluate_file)
    test_data = [json.loads(line) for line in open(evaluate_file)]
    print('Evaluate prediction of ' + evaluate_file)

    pr_coref_evaluator_seen = metrics.PrCorefEvaluatorSeen()
    eval_nn_type = ['nn']
    eval_recall_num = [1, 5, 10]
    recall = dict()
    for pronoun_type in ['NotDiscussed', 'Discussed']:
        recall[pronoun_type] = pd.DataFrame(index=eval_nn_type, columns=['r@' + str(i) for i in eval_recall_num])
        recall[pronoun_type] = recall[pronoun_type].fillna(0)
    all_count = {'Discussed':0, 'NotDiscussed': 0}

    for i, tmp_example in enumerate(test_data):
        # clusters in dialog
        pr_coref_evaluator_seen.update(tmp_example["predicted_clusters"], tmp_example["pronoun_info"], tmp_example["sentences"])
        # prediction of pool
        doc_key = tmp_example['doc_key']
        for prp_count, pronoun_example in enumerate(tmp_example["pronoun_info"]):
            if pronoun_example['reference_type'] != 0:
                continue
            tmp_correct_candidate_NPs = pronoun_example['correct_NPs']
            coref_in_pool = False
            # extract nns
            correct_ant = {key: [] for key in ['nn', 'nn_syn', 'nn_hyper', 'nn_hypo']}
            pronoun_type = 'NotDiscussed'
            for NP in tmp_correct_candidate_NPs:
                if isinstance(NP, dict):
                    correct_ant['nn'].append(NP['nn'])
                    correct_ant['nn_syn'].extend(NP['synonym'])
                    correct_ant['nn_hyper'].extend(NP['hypernym'])
                    correct_ant['nn_hypo'].extend(NP['hyponym'])
                    coref_in_pool = True
                else:
                    pronoun_type = 'Discussed'
            if not coref_in_pool:
                continue
            correct_nn = set(correct_ant['nn'])
            correct_syn_hyper_hypo = set(correct_ant['nn_hyper'] + correct_ant['nn_syn'] + correct_ant['nn_hypo'])
            predicted_nn = pronoun_example["predicted_nn"]
            for hyper in correct_syn_hyper_hypo:
                if hyper in predicted_nn:
                    predicted_nn.remove(hyper)
            # calculate recall
            for recall_n in eval_recall_num:
                for recall_type in eval_nn_type:
                    if len(correct_nn & set(predicted_nn[:recall_n])) > 0:
                        recall[pronoun_type]['r@' + str(recall_n)][recall_type] += 1
            all_count[pronoun_type] += 1

    results = []
    for pronoun_type in ['NotDiscussed', 'Discussed']:
        print('Pronoun resolution in pool: ' + pronoun_type)
        recall[pronoun_type] /= all_count[pronoun_type]
        print(recall[pronoun_type])
        results.extend(recall[pronoun_type].loc['nn'].to_list())

    print('Pronoun resolution in dialog')
    p,r,f = pr_coref_evaluator_seen.get_prf()
    print('P: %.2f, R: %.2f, F1: %.2f' % (p * 100, r * 100, f * 100))
    results.extend([p, r, f])

    return results
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)    
