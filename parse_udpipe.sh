#!/usr/bin/env bash

if [[ ! -d udpipe-1.2.0-bin ]]; then
    curl -L --remote-name-all https://github.com/ufal/udpipe/releases/download/v1.2.0/udpipe-1.2.0-bin.zip
    unzip udpipe-1.2.0-bin.zip
fi
if [[ ! -f english-ewt-ud-2.4-190531.udpipe ]]; then
    curl -L --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/english-ewt-ud-2.4-190531.udpipe
fi

for div in train dev test; do
    cut -d"	" -f-10 ${div}/streusle.ud_${div}.conllulex > ${div}/streusle.ud_${div}.conllu
    cut -d"	" -f11- ${div}/streusle.ud_${div}.conllulex | sed 's/^#.*//' > ${div}/streusle.ud_${div}.lex
    udpipe-1.2.0-bin/bin-linux64/udpipe english-ewt-ud-2.4-190531.udpipe ${div}/streusle.ud_${div}.conllu --tag --parse \
        > ${div}/streusle.ud_${div}.udpipe-2.4-190531.conllu
    paste ${div}/streusle.ud_${div}.udpipe-2.4-190531.conllu ${div}/streusle.ud_${div}.lex \
        > ${div}/streusle.ud_${div}.udpipe-2.4-190531.conllulex
done
