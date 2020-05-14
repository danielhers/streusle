#!/usr/bin/env bash

cut -d"	" -f11- $1 | sed 's/^#.*//' > ${1%conllulex}lex  # grep -v '^#'