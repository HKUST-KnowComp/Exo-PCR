#!/bin/bash
echo Downloading $1
wget -P models http://nlp.cs.washington.edu/pair2vec/$1.tar.gz
tar xvzf models/$1.tar.gz -C models
rm models/$1.tar.gz


download_bert(){
  model=$1
  wget -P models https://storage.googleapis.com/bert_models/2018_10_18/$model.zip
  unzip models/$model.zip
  rm models/$model.zip
  mv $model models/
}

download_spanbert(){
  model=$1
  wget -P models https://dl.fbaipublicfiles.com/fairseq/models/$model.tar.gz
  mkdir models/$model
  tar xvfz models/$model.tar.gz -C models/$model
  rm models/$model.tar.gz
}


if [ $1==bert_base ]; then
    download_bert cased_L-12_H-768_A-12
elif [ $1==bert_large ]; then
    download_bert cased_L-24_H-1024_A-16
elif [ $1==spanbert_base ]; then
    download_spanbert spanbert_hf_base
elif [ $1==spanbert_large ]; then
    download_spanbert spanbert_hf
fi