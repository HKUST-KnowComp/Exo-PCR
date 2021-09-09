#!/bin/bash
echo Downloading $1
wget -P models http://nlp.cs.washington.edu/pair2vec/$1.tar.gz
tar xvzf models/$1.tar.gz -C models
rm models/$1.tar.gz
