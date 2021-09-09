# Exophoric Pronoun Resolution in Dialogues with Topic Regularization

## Introduction
This is the data and the source code for the EMNLP 2021 paper "Exophoric Pronoun Resolution in Dialogues with Topic Regularization."

## Usage of Model

### Getting Started
* Install python 3.6+ and the following requirements: `pip install -r requirements.txt`.
* Build custom kernels by running `setup_all.sh`.
    * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* Download pretrained coreference BERT models with `./download_pretrained.sh <model_name>` (e.g,: bert_base, spanbert_base). This downloads BERT/SpanBERT models finetuned on OntoNotes as the baseline models to `models` folder.

### Preprocessing data

We use [VisPro](https://github.com/HKUST-KnowComp/Visual_PCR) as the experiment dataset. The training, validation, and test set contains 4,000, 500, and 500 dialogs, respectively. 

The data preprocessing is described in Sec. 5.4 of the submitted paper. To enable straight reproduction, we upload the preprocessed data files in the `data` folder.


### Training Instructions

* Run `python get_lda.py` to generate LDA topic labels.
* Experiment configurations are found in `experiments.conf`.
* Choose an experiment that you would like to run, e.g. `bert_base`.
* For training and prediction, set the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* Training: `python train.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* Prediction: `python predict.py <experiment>`
* Evaluation: `python evaluate.py <experiment>`. Evaluation metrics are described in Sec. 5.2 of the submitted paper.
* The hyperparameters are described in Sec. 5.4 of the submitted paper. We follow the hyperparameters set in the baseline model and did not perform hyperparameter search.

## Model Performance

The performance is evaluated on a 11GB GeForce RTX 2080 Ti with CUDA 9.2 and CUDNN 7.2.

|                                  | Test Set          |               |          | Validation Set           |               |           |
|----------------------------------|-------------------|---------------|----------|-------------------|---------------|-----------|
|                                  | Out\-of\-text     |               | In\-text | Out\-of\-text     |               | In\-text  |
|                                  | Not Discussed R@1 | Discussed R@1 | F1       | Not Discussed R@1 | Discussed R@1 | F1        |
| BERT\-base                       | 87\.45            | 89\.74        | 83\.47   | 91\.94            | 91\.72        | 85\.56    |
| BERT\-base \+ topic \(ours\)     | 90\.49            | 92\.46        | 84\.72   | 94\.31            | 91\.33        | 86\.12    |
| SpanBERT\-base                   | 87\.65            | 91\.38        | 83\.94   | 93\.36            | 92\.33        | 86\.77    |
| SpanBERT\-base \+ topic \(ours\) | 90\.28            | 93\.63        | 84\.87   | 95\.73            | 92\.64        | 86\.80    |


The number of parameters and inference time are 

|                                  | \#params | Inference Time per Dialog/s |
|----------------------------------|----------|-----------------------------|
| BERT\-base                       | 436M     | 0\.0886                     |
| BERT\-base \+ topic \(ours\)     | 447M     | 0\.0898                     |
| SpanBERT\-base                   | 436M     | 0\.0897                     |
| SpanBERT\-base \+ topic \(ours\) | 448M     | 0\.0908                     |


## Acknowledgment
We built the training framework based on the original [BERT and SpanBERT for Coreference Resolution](https://github.com/mandarjoshi90/coref).

