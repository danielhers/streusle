#!/usr/bin/env bash

cut -d"	" -f-10 $1 | sed 's/^#.*//' > ${1%lex}
