#!/bin/bash
pip3 install nltk

mkdir -p data
wget -P data/ 'https://raw.githubusercontent.com/zaratsian/Spark/master/data/airlines.csv'

hdfs dfs -mkdir -p data/airline-reviews  && hdfs dfs -copyFromLocal data/airlines.csv data/airline-reviews

exit 0
