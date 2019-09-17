#!/bin/bash
#
set -eux

RELUTILDIR='releaseutil'
# Run from the main release directory

# Sanity check
./conllulex2json.py streusle.conllulex > /dev/null

mkdir -p {train,dev,test}

$RELUTILDIR/stats.sh streusle.conllulex

python $RELUTILDIR/split.py streusle.conllulex ud_train_sent_ids.txt > train/streusle.ud_train.conllulex

python $RELUTILDIR/split.py streusle.conllulex ud_dev_sent_ids.txt > dev/streusle.ud_dev.conllulex

python $RELUTILDIR/split.py streusle.conllulex ud_test_sent_ids.txt > test/streusle.ud_test.conllulex

cd train
../$RELUTILDIR/stats.sh streusle.ud_train.conllulex

../conllulex2json.py streusle.ud_train.conllulex > streusle.ud_train.json

../govobj.py streusle.ud_train.json > streusle.ud_train.govobj.json
cd -

cd dev
../$RELUTILDIR/stats.sh streusle.ud_dev.conllulex

../conllulex2json.py streusle.ud_dev.conllulex > streusle.ud_dev.json

../govobj.py streusle.ud_dev.json > streusle.ud_dev.govobj.json
cd -

cd test
../$RELUTILDIR/stats.sh streusle.ud_test.conllulex

../conllulex2json.py streusle.ud_test.conllulex > streusle.ud_test.json

../govobj.py streusle.ud_test.json > streusle.ud_test.govobj.json
cd -

echo "* [train](train/STATS.md), [dev](dev/STATS.md), [test](test/STATS.md)" >> STATS.md

echo "Done with release.sh"
