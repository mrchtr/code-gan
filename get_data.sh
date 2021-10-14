#!/bin/bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tjz92ShC3gwklJwNMr7nGJEXF6TvCS5X' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tjz92ShC3gwklJwNMr7nGJEXF6TvCS5X" -O out_eval.txt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13x-PzszXLCuf3-9132RjLVa5HGKA4eWB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13x-PzszXLCuf3-9132RjLVa5HGKA4eWB" -O out_train.txt && rm -rf /tmp/cookies.txt

mv out_eval.txt ./data/dataset/out_eval.txt
mv out_train.txt ./data/dataset/out_train.txt

mkdir data/dataset/tokenizer
cp ./data/dataset/out_eval.txt ./data/dataset/tokenizer/out_eval.txt
mkdir ./data/dataset/tokenizer/splitted
split -b 1M ./data/dataset/tokenizer/out_test.txt ./data/dataset/tokenizer/splitted/file
mv ./data/dataset/tokenizer/splitted/fileaa ./data/dataset/tokenizer/file.txt
rm -rf  ./data/dataset/tokenizer/splitted


split -b 1M -d  ./data/dataset/tokenizer/out_eval.txt part

