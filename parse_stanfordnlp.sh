#!/usr/bin/env bash

pip install stanfordnlp

yes ""|python -c "import stanfordnlp; stanfordnlp.download('en', confirm_if_exists=True)"

for div in train dev test; do
    cut -d"	" -f-10 ${div}/streusle.ud_${div}.conllulex > ${div}/streusle.ud_${div}.conllu
    cut -d"	" -f11- ${div}/streusle.ud_${div}.conllulex | sed 's/^#.*//' > ${div}/streusle.ud_${div}.lex
    python -c "import stanfordnlp
with open('${div}/streusle.ud_${div}.conllu', encoding='utf-8') as f, open('${div}/streusle.ud_${div}.stanfordnlp.conllu', 'w', encoding='utf-8') as g:
    print(stanfordnlp.Pipeline(tokenize_pretokenized=True)(f.read()).conll_file.conll_as_string(), file=g)"
    paste ${div}/streusle.ud_${div}.stanfordnlp.conllu ${div}/streusle.ud_${div}.lex \
        > ${div}/streusle.ud_${div}.stanfordnlp.conllulex
done
