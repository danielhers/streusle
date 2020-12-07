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

The code and data is based on the [official STREUSLE repository](https://github.com/nert-nlp/streusle).
