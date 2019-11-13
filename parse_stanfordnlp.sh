#!/usr/bin/env bash

pip install stanfordnlp udapi

yes ""|python -c "import stanfordnlp; stanfordnlp.download('en', '.', confirm_if_exists=True)"

for div in dev; do  # train dev test
    cut -d"	" -f-10 ${div}/streusle.ud_${div}.conllulex > ${div}/streusle.ud_${div}.conllu
    cut -d"	" -f11- ${div}/streusle.ud_${div}.conllulex | sed 's/^#.*//' > ${div}/streusle.ud_${div}.lex
    sed 's/SpaceAfter=No/_/;/^# text = /d' ${div}/streusle.ud_${div}.conllu | udapy read.Conllu write.Conllu |
      while read line; do
        if [[ ${line} == "# text = "* ]]; then
          echo "${line#*= }"
        fi
      done > ${div}/streusle.ud_${div}.txt
#    python -c "import stanfordnlp
#with open('${div}/streusle.ud_${div}.txt', encoding='utf-8') as f, open('${div}/streusle.ud_${div}.stanfordnlp.conllu', 'w', encoding='utf-8') as g:
#    print(stanfordnlp.Pipeline(models_dir='.', tokenize_pretokenized=True)(f.read()).conll_file.conll_as_string(), file=g, end='')"
    # , processors='tokenize,pos'
    python -m stanfordnlp.models.parser --include_comments --save_dir=en_ewt_models --eval_file ${div}/streusle.ud_${div}.conllu --output_file ${div}/streusle.ud_${div}.stanfordnlp.conllu --shorthand en_ewt --mode predict --batch_size 5000

    paste ${div}/streusle.ud_${div}.stanfordnlp.conllu ${div}/streusle.ud_${div}.lex \
        > ${div}/streusle.ud_${div}.stanfordnlp.conllulex
done
