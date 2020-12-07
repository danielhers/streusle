STREUSLE Dataset and Code for Rule-based STREUSLE-to-UCCA Converter
===================================================================

This branch accompanies the [COLING 2020](https://coling2020.org/) paper [Comparison by Conversion: Reverse-Engineering UCCA from Syntax and Lexical Semantics](https://www.aclweb.org/anthology/2020.coling-main.264.pdf):

```
@inproceedings{hershcovich-etal-2020-comparison,
    title = "Comparison by Conversion: Reverse-Engineering {UCCA} from Syntax and Lexical Semantics",
    author = "Hershcovich, Daniel  and
      Schneider, Nathan  and
      Dvir, Dotan  and
      Prange, Jakob  and
      de Lhoneux, Miryam  and
      Abend, Omri",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.264",
    pages = "2947--2966",
    abstract = "Building robust natural language understanding systems will require a clear characterization of whether and how various linguistic meaning representations complement each other. To perform a systematic comparative analysis, we evaluate the mapping between meaning representations from different frameworks using two complementary methods: (i) a rule-based converter, and (ii) a supervised delexicalized parser that parses to one framework using only information from the other as features. We apply these methods to convert the STREUSLE corpus (with syntactic and lexical semantic annotations) to UCCA (a graph-structured full-sentence meaning representation). Both methods yield surprisingly accurate target representations, close to fully supervised UCCA parser quality{---}indicating that UCCA annotations are partially redundant with STREUSLE annotations. Despite this substantial convergence between frameworks, we find several important areas of divergence.",
}
```

Changes in this repository, with respect to the code and data in [official STREUSLE repository](https://github.com/nert-nlp/streusle), are:
* [Converter from STREUSLE + UD to UCCA](conllulex2ucca.py), described in the paper and [documented in detail](README_conllulex2ucca.md).
* Scripts to convert [from CoNLL-U-Lex to CoNLL-U](conllulex2conllu.sh) (by keeping only the first 10 columns), and [from CoNLL-U-Lex to Lex](conllulex2lex.sh) (by dropping the first 10 columns).
* Scripts to parse to CoNLL-U, using [UDPipe](parse_udpipe.sh) and [StanfordNLP](parse_stanfordnlp.sh), and concatenate the Lex columns from the original file. These allow experimenting with predicted (rather than gold) syntax.

See also the [streusle2ucca-2019](https://github.com/danielhers/streusle/tree/streusle2ucca-2019) branch, containing the code for the alternative converter, described in Appendix B of the paper.
