from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def verify_correct_NP_match(predicted_NP, gold_NPs, model, matched_gold_ids):
    if model == 'exact':
        for gold_id, tmp_gold_NP in enumerate(gold_NPs):
            if gold_id in matched_gold_ids:
                continue
            if tmp_gold_NP[0] == predicted_NP[0] and tmp_gold_NP[1] == predicted_NP[1]:
                return gold_id
    elif model == 'cover':
        for gold_id, tmp_gold_NP in enumerate(gold_NPs):
            if gold_id in matched_gold_ids:
                continue
            if tmp_gold_NP[0] <= predicted_NP[0] and tmp_gold_NP[1] >= predicted_NP[1]:
                return gold_id
            if tmp_gold_NP[0] >= predicted_NP[0] and tmp_gold_NP[1] <= predicted_NP[1]:
                return gold_id
    return None
    

class PrCorefEvaluator(object):
    def __init__(self):
        self.all_coreference = 0
        self.predict_coreference = 0
        self.correct_predict_coreference = 0
        # for not cap only
        self.all_coref_not_cap_only = 0
        self.predict_coref_not_cap_only = 0
        self.correct_predict_coref_not_cap_only = 0
        # for cap only
        self.all_coref_cap_only = 0
        self.predict_coref_cap_only = 0
        self.correct_predict_coref_cap_only = 0

        self.pronoun_list = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them', 'They', 'it', 'It', 'his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']

    def get_prf(self):
        results = {}
        results['p'] = 0 if self.predict_coreference == 0 else self.correct_predict_coreference / self.predict_coreference
        results['r'] = 0 if self.all_coreference == 0 else self.correct_predict_coreference / self.all_coreference
        results['f'] = 0 if results['p'] + results['r'] == 0 else 2 * results['p'] * results['r'] / (results['p'] + results['r'])
        results['p_not_cap_only'] = 0 if self.predict_coref_not_cap_only == 0 else self.correct_predict_coref_not_cap_only / self.predict_coref_not_cap_only
        results['r_not_cap_only'] = 0 if self.all_coref_not_cap_only == 0 else self.correct_predict_coref_not_cap_only / self.all_coref_not_cap_only
        results['f_not_cap_only'] = 0 if results['p_not_cap_only'] + results['r_not_cap_only'] == 0 else 2 * results['p_not_cap_only'] * results['r_not_cap_only'] / (results['p_not_cap_only'] + results['r_not_cap_only'])
        results['p_cap_only'] = 0 if self.predict_coref_cap_only == 0 else self.correct_predict_coref_cap_only / self.predict_coref_cap_only
        results['r_cap_only'] = 0 if self.all_coref_cap_only == 0 else self.correct_predict_coref_cap_only / self.all_coref_cap_only
        results['f_cap_only'] = 0 if results['p_cap_only'] + results['r_cap_only'] == 0 else 2 * results['p_cap_only'] * results['r_cap_only'] / (results['p_cap_only'] + results['r_cap_only'])

        return results


    def update(self, predicted_clusters, pronoun_info, sentences, tokens_np=None):
        all_sentence = list()
        predicted_clusters = [tuple(pc) for pc in predicted_clusters]

        for s in sentences:
            all_sentence += s

        if tokens_np is not None:
            tokens_np_len = np.sum(tokens_np != '', axis=1)
            tokens_np_end = np.cumsum(tokens_np_len) + len(all_sentence) - 2
            neg_nps = []
            for neg_end, neg_len in zip(tokens_np_end, tokens_np_len):
                neg_nps.append([neg_end - neg_len + 3, neg_end])

        for pronoun_example in pronoun_info:
            tmp_pronoun_index = pronoun_example['current_pronoun'][0]

            tmp_candidate_NPs = pronoun_example['candidate_NPs']
            if tokens_np is not None:
                tmp_candidate_NPs += neg_nps
            tmp_correct_candidate_NPs = pronoun_example['correct_NPs']

            if pronoun_example['coreference_in_cap_only']:
                self.all_coref_cap_only += len(tmp_correct_candidate_NPs)
            else:
                self.all_coref_not_cap_only += len(tmp_correct_candidate_NPs)
           
            find_pronoun = False
            for coref_cluster in predicted_clusters:
                for mention in coref_cluster:
                    mention_start_index = mention[0]
                    if mention_start_index == tmp_pronoun_index:
                        find_pronoun = True
                if find_pronoun and pronoun_example['reference_type'] == 0:
                    matched_cdd_np_ids = []
                    matched_crr_np_ids = []
                    for mention in coref_cluster:
                        mention_start_index = mention[0]
                        tmp_mention_span = (
                            mention_start_index,
                            mention[1])
                        matched_np_id = verify_correct_NP_match(tmp_mention_span, tmp_candidate_NPs, 'cover', matched_cdd_np_ids)
                        if matched_np_id is not None:
                            # exclude such scenario: predict 'its' and overlap with candidate 'its eyes'
                            # predict +1 but correct +0
                            if tmp_mention_span[0] < len(all_sentence) and\
                                tmp_mention_span[0] == tmp_mention_span[1] and\
                                all_sentence[tmp_mention_span[0]] in self.pronoun_list and\
                                len(tmp_candidate_NPs[matched_np_id]) > 1:
                                continue
                            matched_cdd_np_ids.append(matched_np_id)
                            self.predict_coreference += 1
                            if pronoun_example['coreference_in_cap_only']:
                                self.predict_coref_cap_only += 1
                            else:
                                self.predict_coref_not_cap_only += 1
                            matched_np_id = verify_correct_NP_match(tmp_mention_span, tmp_correct_candidate_NPs, 'cover', matched_crr_np_ids)
                            if matched_np_id is not None:
                                matched_crr_np_ids.append(matched_np_id)
                                self.correct_predict_coreference += 1
                                if pronoun_example['coreference_in_cap_only']:
                                    self.correct_predict_coref_cap_only += 1
                                else:
                                    self.correct_predict_coref_not_cap_only += 1
                    break

            self.all_coreference += len(tmp_correct_candidate_NPs)


class PrCorefEvaluatorConll(object):
    def __init__(self):
        self.all_coreference = 0
        self.predict_coreference = 0
        self.correct_predict_coreference = 0

        self.pronoun_list = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them', 'They', 'it', 'It', 'his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']

    def get_prf(self):
        p = 0 if self.predict_coreference == 0 else self.correct_predict_coreference / self.predict_coreference
        r = 0 if self.all_coreference == 0 else self.correct_predict_coreference / self.all_coreference
        f = 0 if p + r == 0 else 2 * p * r / (p + r)

        return p, r, f


    def update(self, predicted_clusters, pronoun_info, sentences):
        all_sentence = list()
        predicted_clusters = [tuple(pc) for pc in predicted_clusters]

        for s in sentences:
            all_sentence += s

        for pronoun_example in pronoun_info:
            tmp_pronoun_index = pronoun_example['current_pronoun'][0]

            tmp_candidate_NPs = []
            for mention in pronoun_example['candidate_NPs']:
                tmp_candidate_NPs.append((mention[0], mention[1]))
            tmp_correct_candidate_NPs = []
            for mention in pronoun_example['correct_NPs']:
                tmp_correct_candidate_NPs.append((mention[0], mention[1]))
           
            find_pronoun = False
            for coref_cluster in predicted_clusters:
                for mention in coref_cluster:
                    mention_start_index = mention[0]
                    if mention_start_index == tmp_pronoun_index:
                        find_pronoun = True
                if find_pronoun:
                    matched_cdd_np_ids = []
                    matched_crr_np_ids = []
                    for mention in coref_cluster:
                        matched_np_id = verify_correct_NP_match(mention, tmp_candidate_NPs, 'cover', matched_cdd_np_ids)
                        if matched_np_id is not None:
                            # exclude such scenario: predict 'its' and overlap with candidate 'its eyes'
                            # predict +1 but correct +0
                            if mention[0] < len(all_sentence) and\
                                mention[0] == mention[1] and\
                                all_sentence[mention[0]] in self.pronoun_list and\
                                len(tmp_candidate_NPs[matched_np_id]) > 1:
                                continue
                            matched_cdd_np_ids.append(matched_np_id)
                            self.predict_coreference += 1
                            matched_np_id = verify_correct_NP_match(mention, tmp_correct_candidate_NPs, 'cover', matched_crr_np_ids)
                            if matched_np_id is not None:
                                matched_crr_np_ids.append(matched_np_id)
                                self.correct_predict_coreference += 1
                    break

            self.all_coreference += len(tmp_correct_candidate_NPs)


class PrCorefEvaluatorUnseen(object):
    def __init__(self):
        self.eval_nn_type = ['nn', 'nn_syn', 'nn_syn_hyper']
        self.eval_recall_num = [1, 5, 10]
        self.recall = dict()
        for pronoun_type in ['unseen', 'seen']:
            self.recall[pronoun_type] = pd.DataFrame(index=self.eval_nn_type, columns=['r@' + str(i) for i in self.eval_recall_num])
            self.recall[pronoun_type] = self.recall[pronoun_type].fillna(0)
        self.all = {'seen':0, 'unseen': 0}

    def get_recall(self):
        for pronoun_type in ['unseen', 'seen']:
            self.recall[pronoun_type] /= self.all[pronoun_type]

        return self.recall

    def update(self, pronoun_info, top_antecedents, top_antecedent_scores, top_span_starts, top_span_ends):
        for pronoun_example in pronoun_info:
            if pronoun_example['reference_type'] != 0:
                continue
            tmp_correct_candidate_NPs = pronoun_example['correct_NPs']

            tmp_pronoun_index = pronoun_example['current_pronoun'][0]
            for span_id, (start, end) in enumerate(zip(top_span_starts, top_span_ends)):
                if tmp_pronoun_index == start and tmp_pronoun_index == end:
                    # if match, must coref in pool
                    # extract nns
                    correct_ant = {key: [] for key in self.eval_nn_type}
                    pronoun_type = 'unseen'
                    for NP in tmp_correct_candidate_NPs:
                        if isinstance(NP, dict):
                            correct_ant['nn'].append(NP['nn'])
                            correct_ant['nn_syn'].extend(NP['synonym'])
                            correct_ant['nn_syn_hyper'].extend(NP['hypernym'])
                        else:
                            pronoun_type = 'seen'
                    for key in self.eval_nn_type:
                        correct_ant[key] = set(correct_ant[key])
                    correct_ant['nn_syn'] = correct_ant['nn_syn'] | correct_ant['nn']
                    correct_ant['nn_syn_hyper'] = correct_ant['nn_syn_hyper'] | correct_ant['nn_syn']
                    # top 10 highest score
                    predicted_nn = np.argsort(top_antecedent_scores[span_id])[::-1][:10]
                    # note top_antecedent_scores has dim 0 for dummy
                    if 0 in predicted_nn:
                        index_0 = np.where(predicted_nn == 0)[0][0]
                        predicted_nn = np.delete(predicted_nn, index_0)
                    else:
                        index_0 = None
                    # top 10 predicted nn index
                    predicted_nn = top_antecedents[span_id][predicted_nn - 1]
                    if index_0 is not None:
                        predicted_nn = np.insert(predicted_nn, index_0, -1)
                    # calculate recall
                    for recall_n in self.eval_recall_num:
                        for recall_type in self.eval_nn_type:
                            if len(correct_ant[recall_type] & set(predicted_nn[:recall_n])) > 0:
                                self.recall[pronoun_type]['r@' + str(recall_n)][recall_type] += 1
                    self.all[pronoun_type] += 1
                    break


class PrCorefEvaluatorSeen(object):
    def __init__(self):
        self.all_coreference = 0
        self.predict_coreference = 0
        self.correct_predict_coreference = 0

        self.pronoun_list = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them', 'They', 'it', 'It', 'his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']

    def get_prf(self):
        p = 0 if self.predict_coreference == 0 else self.correct_predict_coreference / self.predict_coreference
        r = 0 if self.all_coreference == 0 else self.correct_predict_coreference / self.all_coreference
        f = 0 if p + r == 0 else 2 * p * r / (p + r)

        return p, r, f


    def update(self, predicted_clusters, pronoun_info, sentences):        
        all_sentence = list()
        predicted_clusters = [tuple(pc) for pc in predicted_clusters]

        for s in sentences:
            all_sentence += s

        for pronoun_example in pronoun_info:
            if pronoun_example['reference_type'] != 0:
                continue
            tmp_pronoun_index = pronoun_example['current_pronoun'][0]
            tmp_candidate_NPs = pronoun_example['candidate_NPs']
            tmp_correct_candidate_NPs = [p for p in pronoun_example['correct_NPs'] if isinstance(p, list)]
            find_pronoun = False
            for coref_cluster in predicted_clusters:
                for mention in coref_cluster:
                    mention_start_index = mention[0]
                    if mention_start_index == tmp_pronoun_index:
                        find_pronoun = True
                if find_pronoun:
                    matched_cdd_np_ids = []
                    matched_crr_np_ids = []
                    for mention in coref_cluster:
                        mention_start_index = mention[0]
                        tmp_mention_span = (
                            mention_start_index,
                            mention[1])
                        matched_np_id = verify_correct_NP_match(tmp_mention_span, tmp_candidate_NPs, 'cover', matched_cdd_np_ids)
                        if matched_np_id is not None:
                            # exclude such scenario: predict 'its' and overlap with candidate 'its eyes'
                            # predict +1 but correct +0
                            if tmp_mention_span[0] < len(all_sentence) and\
                                tmp_mention_span[0] == tmp_mention_span[1] and\
                                all_sentence[tmp_mention_span[0]] in self.pronoun_list and\
                                len(tmp_candidate_NPs[matched_np_id]) > 1:
                                continue
                            matched_cdd_np_ids.append(matched_np_id)
                            self.predict_coreference += 1
                            matched_np_id = verify_correct_NP_match(tmp_mention_span, tmp_correct_candidate_NPs, 'cover', matched_crr_np_ids)
                            if matched_np_id is not None:
                                matched_crr_np_ids.append(matched_np_id)
                                self.correct_predict_coreference += 1
                    break

            self.all_coreference += len(tmp_correct_candidate_NPs)        