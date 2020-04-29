#!/usr/bin/env python3
"""
Given a .conllulex file, convert each sentence to a UCCA Passage.
If the script is called directly, outputs the data as XML, Pickle or JSON files.

@since: 2020-04-27
@author: Daniel Hershcovich, Nathan Schneider
@requires: ucca==1.3.0 semstr==1.2.2 tqdm
"""

import argparse
import csv
import os
import re
import sys
from itertools import groupby
from operator import attrgetter, itemgetter
from typing import List, Iterable, Optional

from depedit.depedit import DepEdit, ParsedToken
from semstr.convert import iter_files, write_passage
from tqdm import tqdm
from ucca import core, layer0, layer1, evaluation
from ucca.evaluation import UNLABELED, LABELED
from ucca.ioutil import get_passages
from ucca.normalization import normalize

from conllulex2json import load_sents
from supersenses import coarsen_pss

SENT_ID = "sent_id"
MWE_TYPES = ("swes", "smwes", "wmwes")

# Rules to assimilate UD to UCCA, based on https://www.aclweb.org/anthology/N19-1047/
DEPEDIT_TRANSFORMATIONS = ["\t".join(transformation) for transformation in [
    ("func=/.*/;func=/cc/;func=/conj|root/", "#1>#3;#3>#2", "#1>#2"),         # raise cc        over conj, root
    ("func=/.*/;func=/mark/;func=/advcl/", "#1>#3;#3>#2", "#1>#2"),           # raise mark      over advcl
    ("func=/.*/;func=/advcl/;func=/appos|root/", "#1>#3;#3>#2", "#1>#2"),     # raise advcl     over appos, root
    ("func=/.*/;func=/appos/;func=/root/", "#1>#3;#3>#2", "#1>#2"),           # raise appos     over root
    ("func=/.*/;func=/conj/;func=/parataxis|root/", "#1>#3;#3>#2", "#1>#2"),  # raise conj      over parataxis, root
    ("func=/.*/;func=/parataxis/;func=/root/", "#1>#3;#3>#2", "#1>#2"),       # raise parataxis over root
]]
DEPEDIT_TRANSFORMED_DEPRELS = set(re.search(r"(?<=func=/)\w+", t).group(0) for t in DEPEDIT_TRANSFORMATIONS)


class ConllulexToUccaConverter:
    def __init__(self):
        self.depedit = DepEdit(DEPEDIT_TRANSFORMATIONS)

    def convert(self, sent: dict) -> core.Passage:
        """
        Create one UCCA passage from a STREUSLE sentence dict.
        :param sent: conllulex2json sentence dict, containing "sent_id" and "toks"
        :return: UCCA Passage where each Terminal corresponds to a token from the original sentence
        """
        # Assimilate UD tree a UCCA conventions a little bit
        self.depedit_transform(sent)

        # Create UCCA passage
        passage = core.Passage(ID=sent[SENT_ID].replace("reviews-", ""))

        # Create terminals and add to layer 0
        l0 = layer0.Layer0(passage)
        tokens = [Token(tok, l0) for tok in sent["toks"]]

        # Auxiliary data structure for UD tree: dependency nodes
        nodes = [Node(None)] + tokens  # Prepend root
        for node in nodes:  # Link heads to dependents
            node.link(nodes)

        # Link tokens to their single/multi-word expressions and vice versa
        exprs = {expr_type: sent[expr_type].values() for expr_type in MWE_TYPES}
        for expr_type, expr_values in exprs.items():
            for expr in expr_values:
                for tok_num in expr["toknums"]:
                    node = nodes[tok_num]
                    node.exprs[expr_type] = expr
                    expr.setdefault("nodes", []).append(node)

        # TODO intermediate structure - annotate units with attributes such as "scene-evoking"

        # TODO Create primary UCCA tree
        l1 = layer1.Layer1(passage)
        # add a child to unit (use unit=None for root), returns the new child: l1.add_fnode(unit, tag)
        # add a child with multiple categories (e.g., P+S): l1.add_fnode_multiple(unit, [(tag,) for tag in tags])
        # add a remote child (units already exist): l1.add_remote(parent, tag, child)
        # link preterminal to terminal: unit.add(Categories.Terminal, terminal)

        return passage

    def depedit_transform(self, sent):
        """
        Apply pre-conversion transformations to dependency tree
        NOTE: DepEdit does not modify enhanced dependencies
        """
        parsed_tokens = [tok2depedit(tok) for tok in sent["toks"]]
        self.depedit.process_sentence(parsed_tokens)
        for tok, parsed_token in zip(sent["toks"], parsed_tokens):  # Update tokens according to transformed properties
            tok.update(depedit2tok(parsed_token))

    def evaluate(self, converted_passage: core.Passage, sent: dict, reference_passage: core.Passage,
                 report: Optional[csv.writer] = None):
        if report:
            toks = {tok["#"]: tok for tok in sent["toks"]}
            exprs = {frozenset(expr["toknums"]): (expr_id, expr_type, expr) for expr_type in ("swes", "smwes", "wmwes")
                     for expr_id, expr in sent[expr_type].items()}
            converted_units, reference_units = [{positions: list(units) for positions, units in
                                                 groupby(sorted(passage.layer(layer1.LAYER_ID).all,
                                                                key=evaluation.get_yield),
                                                         key=evaluation.get_yield)}
                                                for passage in (converted_passage, reference_passage)]
            for positions in sorted(set(exprs).union(converted_units).union(reference_units)):
                if positions:
                    expr_id, expr_type, expr = exprs.get(positions, ("", "", {}))
                    ref_units = reference_units.get(positions, [])
                    pred_units = converted_units.get(positions, [])
                    tokens = [toks[i] for i in sorted(positions)]
                    if report:
                        def _join(k):
                            return " ".join(tok[k] for tok in tokens)

                        def _yes(x):
                            return "Yes" if x else ""

                        def _unit_attrs(xs):
                            return [", ".join(x.ID for x in xs),  # unit id in xml
                                    ", ".join(map(str, filter(None, (x.extra.get("tree_id")
                                                                     for x in xs)))),  # tree-id in ucca-app
                                    "|".join(sorted(t for x in xs for t in getattr(x, "ftags", [x.ftag])
                                                    or ())),  # category
                                    "|".join(sorted(t for x in xs for e in x.incoming_basic if e.attrib.get("remote")
                                                    for t in e.tags)),  # remote
                                    _yes(len([t for x in xs for t in x.terminals]) > 1),  # unanalyzable
                                    ", ".join(map(str, xs))]  # annotation

                        terminals = reference_passage.layer(layer0.LAYER_ID).all
                        fields = [
                            reference_passage.ID,  # sent_id
                            _join("word") or " ".join(terminals.by_position(p).text for p in sorted(positions)),  # text
                            _join("deprel"), _join("upos"), _join("edeps"),
                            expr_id, expr_type, expr.get("lexcat"), expr.get("ss"), expr.get("ss2"),
                            _yes(len({tok["head"] for tok in tokens} - positions) <= 1),  # is a dependency subtree?
                        ]
                        fields += _unit_attrs(ref_units) + _unit_attrs(pred_units)
                        report.writerow([f or "" for f in fields])
        return (evaluation.evaluate(converted_passage, reference_passage), reference_passage.ID,
                " ".join(t.text for t in sorted(reference_passage.layer(layer0.LAYER_ID).all, key=attrgetter(
                    "position"))), str(converted_passage), str(reference_passage))


REPORT_HEADERS = [
    "sent_id", "text", "deprel", "upos", "edeps", "expr_id", "expr_type", "lexcat", "ss", "ss2",
    "subtree", "ref_unit_id", "ref_tree_id", "ref_category", "ref_remote", "ref_unanalyzable",
    "ref_annotation", "pred_unit_id", "pred_tree_id", "pred_category", "pred_remote", "pred_unanalyzable",
    "pred_annotation"]

DEPEDIT_FIELDS = dict(  # Map UD/STREUSLE word properties to DepEdit token properties
    tok_id="#", text="word", lemma="lemma", pos="upos", cpos="xpos", morph="feats", head="head", func="deprel",
    head2=None, func2=None, num=None, child_funcs=None, position="#"
)  # tok fields: #, word, lemma, upos, xpos, feats, head, deprel, edeps, misc, smwe, wmwe, lextag


def tok2depedit(tok: dict) -> ParsedToken:
    """
    Translate JSON property names and values to DepEdit aliases
    :param tok: dict of token properties
    :return: DepEdit ParsedToken with mapped properties (attributes)
    """
    return ParsedToken(**{k: None if v is None else tok[v] for k, v in DEPEDIT_FIELDS.items()})


def depedit2tok(token: ParsedToken) -> dict:
    """
    Translate a DepEdit ParsedToken to a dict with JSON property names and values
    :param token: ParsedToken
    :return: dict of mapped token properties
    """
    return {v: getattr(token, k) for k, v in DEPEDIT_FIELDS.items() if v is not None and hasattr(token, k)}


class Node:
    """
    Dependency node.
    """

    def __init__(self, tok: Optional[dict], exprs: Optional[dict] = None):
        """
        :param tok: conllulex2json token dict (from "toks"), or None for the root
        :param exprs: dict with entries for any of "swes", "smwes" and "wmwes", each a dict of strings to values
        """
        self.tok: dict = tok  # None for root; dict created by conllulex2json for tokens
        self.exprs: dict = exprs or {}
        self.position: int = 0  # Position in the sentence (root is zero)
        self.incoming_basic: List[Edge] = []  # List of Edges from heads (basic UD)
        self.outgoing_basic: List[Edge] = []  # List of Edges to dependents (basic UD)
        # NOTE: DepEdit does not modify enhanced dependencies
        self.incoming_enhanced: List[Edge] = []  # List of Edges from heads (enhanced UD)
        self.outgoing_enhanced: List[Edge] = []  # List of Edges to dependents (enhanced UD)

    def link(self, nodes: List["Node"], enhanced: bool = False) -> None:
        """
        Set incoming and outgoing edges after all Nodes have been created.
        :param nodes: List of all Nodes in the sentences, ordered by position (starting with the root in position 0)
        :param enhanced: whether to use enhanced dependencies rather than basic dependencies
        """
        if self.tok:
            self.incoming_basic = [Edge(nodes[self.tok["head"]], self, self.tok["deprel"])]
            # NOTE: DepEdit does not modify enhanced dependencies
            self.incoming_enhanced = [Edge(nodes[int(head)], self, deprel, enhanced=True) for head, _, deprel in
                                      [edep.partition(":") for edep in self.tok["edeps"].split("|")]
                                      if "." not in head]  # Omit null nodes - TODO convert to implicit?
        else:
            self.incoming_basic = []
            self.incoming_enhanced = []
        for edge in self.incoming_basic:
            edge.head.outgoing_basic.append(edge)
        for edge in self.incoming_enhanced:
            edge.head.outgoing_enhanced.append(edge)

    @property
    def head(self) -> Optional["Node"]:
        """
        Shortcut for getting the Node's primary head if it exists and None otherwise.
        :return: head Node
        """
        return self.incoming_basic[0].head if self.incoming_basic else None

    @property
    def deprel(self) -> Optional[str]:
        """
        Shortcut for getting the Node's primary dependency relation if it exists and None otherwise.
        :return: dependency relation str
        """
        return self.incoming_basic[0].deprel if self.incoming_basic else None

    @property
    def basic_deprel(self) -> Optional[str]:
        """
        Remove any relation subtypes (":" and anything following it) from dependency relation.
        :return: basic dependency relation str
        """
        return self.incoming_basic[0].baserel if self.incoming_basic else None

    @property
    def swe(self) -> Optional[dict]:
        return self.exprs.get("swes")

    @property
    def smwe(self) -> Optional[dict]:
        return self.exprs.get("smwes")

    @property
    def wmwe(self) -> Optional[dict]:
        return self.exprs.get("wmwes")

    @property
    def ss(self) -> Optional[str]:
        for expr in self.exprs.values():
            ss = expr.get("ss")
            if ss:
                return ss
        return ""

    @property
    def lexcat(self) -> Optional[str]:
        for expr in self.exprs.values():
            lexcat = expr.get("lexcat")
            if lexcat:
                return lexcat
        return ""

    @property
    def lexlemma(self) -> Optional[str]:
        for expr in self.exprs.values():
            lexlemma = expr.get("lexlemma")
            if lexlemma:
                return lexlemma
        return ""

    def __str__(self):
        return "ROOT"  # Overridden in Token

    def __repr__(self):
        return f"{self.__class__.__name__}({self})"


class Token(Node):
    """
    Dependency node that is not the root, wrapper for conllulex2json token dict (from "toks").
    """

    def __init__(self, tok: dict, l0: layer0.Layer0):
        """
        :param tok: conllulex2json token dict (from "toks")
        :param l0: UCCA Layer0 to add a Terminal to
        """
        super().__init__(tok)
        self.position: int = self.tok["#"]
        self.is_punct: bool = self.tok["upos"] == "PUNCT"
        self.terminal: layer0.Terminal = l0.add_terminal(text=tok["word"], punct=self.is_punct)
        self.terminal.extra.update(dep=tok["deprel"].partition(":")[0],
                                   head=tok["head"] - self.position if tok["head"] else 0,
                                   lemma=tok["lemma"], orth=tok["word"], pos=tok["upos"], tag=tok["xpos"])

    def __str__(self):
        return self.tok['word']


class Edge:
    """
    Edge connecting two Nodes.
    """

    def __init__(self, head: Node, dep: Node, deprel: str, enhanced: bool = False):
        """
        :param head: Node this edge comes from
        :param dep: dependent (Node this edge goes to)
        :param deprel: dependency relation (edge label)
        :param enhanced: whether this is an enhanced edge
        """
        self.head: Node = head
        self.dep: Node = dep
        self.deprel: str = deprel
        self.enhanced: bool = enhanced

    @property
    def baserel(self) -> Optional[str]:
        """
        Remove any relation subtypes (":" and anything following it) from dependency relation.
        :return: basic dependency relation str
        """
        return self.deprel.partition(":")[0]

    def __str__(self):
        return f"{self.head} -{self.deprel}-> {self.dep}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.head}, {self.dep}, {self.deprel})"


class ConcatenatedFiles:
    """
    Wrapper for multiple files to allow seamless iteration over the concatenation of their lines.
    Reads the whole content into memory first, to allow showing a progress bar with percentage of read lines.
    """

    def __init__(self, filenames: Iterable[str]):
        """
        :param filenames: filenames or glob patterns to concatenate
        """
        self.lines: List[str] = []
        self.name: Optional[str] = None
        for filename in iter_files(filenames):
            with open(filename, encoding="utf-8") as f:
                self.lines += list(f)
            self.name = filename

    def __iter__(self) -> Iterable[str]:
        return iter(self.lines)

    def read(self):
        return "\n".join(self.lines)


class SSMapper:
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, ss):
        return coarsen_pss(ss, self.depth) if ss.startswith('p.') else ss


def main(args: argparse.Namespace) -> None:
    converter = ConllulexToUccaConverter()
    run(args, converter)


def run(args, converter):
    if args.report and not args.evaluate:
        argparser.error("--report requires --evaluate")
    os.makedirs(args.out_dir, exist_ok=True)
    sentences = list(load_sents(ConcatenatedFiles(args.filenames), ss_mapper=SSMapper(args.depth),
                                validate_pos=False, validate_type=False))
    converted = {}
    for sent in tqdm(sentences, unit=" sentences", desc="Converting"):
        passage = converter.convert(sent)
        if args.normalize:
            normalize(passage, extra=args.extra_normalization)
        if args.write:
            write_passage(passage, out_dir=args.out_dir, output_format="json" if args.format == "json" else None,
                          binary=args.format == "pickle", verbose=args.verbose)
        converted[passage.ID] = passage, sent
    if args.evaluate:
        passages = ((converted.get(ref_passage.ID), ref_passage) for ref_passage in get_passages(args.evaluate))
        if args.report:
            report_f = open(args.report, "w", encoding="utf-8", newline="")
            report = csv.writer(report_f, delimiter="\t")
            report.writerow(REPORT_HEADERS)
        else:
            report_f = report = None
        results = [converter.evaluate(converted_passage, sent, reference_passage, report)
                   for (converted_passage, sent), reference_passage in
                   tqdm(filter(itemgetter(0), passages), unit=" passages", desc="Evaluating", total=len(converted))]
        if report_f:
            report_f.close()
        summary = evaluation.Scores.aggregate(s for s, *_ in results)
        summary.print()
        prefix = re.sub(r"(^\./*)|(/$)", "", args.out_dir)
        scores_filename = prefix + ".scores.tsv"
        with open(scores_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(summary.titles() + ["ID", "text", "pred", "ref"])
            for result, *fields in results:
                writer.writerow(result.fields() + fields)
        print(f"Wrote '{scores_filename}'", file=sys.stderr)
        summary_filename = prefix + ".summary.tsv"
        with open(summary_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(summary.titles(LABELED) + summary.titles(UNLABELED))
            writer.writerow(summary.fields(LABELED) + summary.fields(UNLABELED))
        print(f"Wrote '{summary_filename}'", file=sys.stderr)
        print(f"Evaluated {len(results)} out of {len(converted)} sentences "
              f"({100 * len(results) / len(converted):.2f}%).", file=sys.stderr)


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("filenames", nargs="+", help=".conllulex or .json STREUSLE file name(s) to convert")
    parser.add_argument("-o", "--out-dir", default=".", help="output directory")
    parser.add_argument("-f", "--format", choices=("xml", "pickle", "json"), default="xml", help="output format")
    parser.add_argument("-v", "--verbose", action="store_true", help="extra information")
    parser.add_argument("-n", "--no-write", action="store_false", dest="write", help="do not write files")
    parser.add_argument("--normalize", action="store_true", help="normalize UCCA passages after conversion")
    parser.add_argument("--extra-normalization", action="store_true", help="apply extra UCCA normalization")
    parser.add_argument("--evaluate", help="directory/filename pattern of gold UCCA passage(s) for evaluation")
    parser.add_argument("--report", help="output filename for report of units, subtrees and multi-word expressions")
    parser.add_argument('--depth', metavar='D', type=int, choices=range(1, 5), default=4,
                        help='depth of hierarchy at which to cluster SNACS supersense labels '
                             '(default: 4, i.e. no collapsing)')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Convert .conllulex files to UCCA")
    add_arguments(argparser)
    main(argparser.parse_args())
