#!/usr/bin/env python3
"""
Given a .conllulex file, convert each sentence to a UCCA Passage.
If the script is called directly, outputs the data as XML, Pickle or JSON files.

@since: 2018-12-27
@author: Daniel Hershcovich
@requires: ucca=1.1.4 semstr==1.1.8 tqdm git+https://github.com/danielhers/depedit
"""

import argparse
import os
from operator import attrgetter, itemgetter
from typing import List, Iterable, Optional

from depedit.depedit import DepEdit, ParsedToken
from semstr.convert import iter_files, write_passage
from tqdm import tqdm
from ucca import core, layer0, layer1, evaluation
from ucca.ioutil import get_passages
from ucca.layer1 import EdgeTags as Categories
from ucca.normalization import normalize

from conllulex2json import load_sents

SENT_ID = "sent_id"
UD_TO_UCCA = dict(
    acl=Categories.Elaborator, advcl=Categories.ParallelScene, advmod=Categories.Adverbial, amod=Categories.Elaborator,
    appos=Categories.Center, aux=Categories.Function, case=Categories.Relator, cc=Categories.Linker,
    ccomp=Categories.Participant, compound=Categories.Elaborator, conj=Categories.ParallelScene,
    cop=Categories.Function, csubj=Categories.Participant, dep=Categories.Function, det=Categories.Elaborator,
    discourse=Categories.ParallelScene, expl=Categories.Function, fixed=Categories.Center,
    goeswith=Categories.Elaborator, head=Categories.Center, iobj=Categories.Participant, list=Categories.ParallelScene,
    mark=Categories.Function, nmod=Categories.Elaborator, nsubj=Categories.Participant, nummod=Categories.Elaborator,
    obj=Categories.Participant, obl=Categories.Participant, orphan=Categories.Participant,
    parataxis=Categories.ParallelScene, vocative=Categories.Participant, xcomp=Categories.Participant,
    root=Categories.ParallelScene, punct=Categories.Punctuation,
)
DEPEDIT_TRANSFORMATIONS = ["\t".join(transformation) for transformation in [
    ("func=/.*/;func=/cc/;func=/conj|root/",        "#1>#3;#3>#2", "#1>#2"),  # raise cc over conj, root
    ("func=/.*/;func=/mark/;func=/advcl/",          "#1>#3;#3>#2", "#1>#2"),  # raise mark over advcl
    ("func=/.*/;func=/advcl/;func=/appos|root/",    "#1>#3;#3>#2", "#1>#2"),  # raise advcl over appos, root
    ("func=/.*/;func=/appos/;func=/root/",          "#1>#3;#3>#2", "#1>#2"),  # raise appos over root
    ("func=/.*/;func=/conj/;func=/parataxis|root/", "#1>#3;#3>#2", "#1>#2"),  # raise conj over parataxis, root
    ("func=/.*/;func=/parataxis/;func=/root/",      "#1>#3;#3>#2", "#1>#2"),  # raise parataxis over root
]]
UNANALYZABLE_DEPREL = {
    "flat", "fixed", "goeswith",
}
UNANALYZABLE_UPOS = {
    "PROPN",
}
UNANALYZABLE_MWE_LEXCAT_SS = {
    ("V.VPC.full", "v.social"), ("V.VPC.full", "v.cognition"), ("V.VPC.full", "v.stative"),
    ("V.VPC.full", "v.possession"), ("V.VPC.full", "v.communication"), ("V.VPC.full", "v.change"),
    # ("N", "n.GROUP"), ("N", "n.LOCATION"),
    ("CCONJ", None), ("ADV", None), ("DISC", None),
}


class ConllulexToUccaConverter:
    def __init__(self, enhanced: bool = False, map_labels: bool = False, **kwargs):
        """
        :param enhanced: whether to use enhanced dependencies rather than basic dependencies
        :param map_labels: whether to translate UD relations to UCCA categories
        """
        del kwargs
        self.enhanced = enhanced
        self.map_labels = map_labels
        self.depedit = DepEdit(DEPEDIT_TRANSFORMATIONS)

    def convert(self, sent: dict) -> core.Passage:
        """
        Create one UCCA passage from a STREUSLE sentence dict.
        :param sent: conllulex2json sentence dict, containing "sent_id" and "toks"
        :return: UCCA Passage where each Terminal corresponds to a token from the original sentence
        """
        # Apply pre-conversion transformations to dependency tree
        toks = sent["toks"]
        parsed_tokens = [tok2depedit(tok) for tok in toks]
        self.depedit.process_sentence(parsed_tokens)
        for tok, parsed_token in zip(toks, parsed_tokens):  # Take transformed properties and update tokens accordingly
            tok.update(depedit2tok(parsed_token))

        # Create passage
        passage = core.Passage(ID=sent[SENT_ID].replace("reviews-", ""))

        # Create terminals
        l0 = layer0.Layer0(passage)
        tokens = [Token(tok, l0) for tok in toks]
        nodes = [Node(None)] + tokens  # Prepend root
        for node in nodes:  # Link heads to dependents
            node.link(nodes, enhanced=self.enhanced)

        # Create primary UCCA tree
        sorted_nodes = topological_sort(nodes)
        l1 = layer1.Layer1(passage)
        remote_edges = []
        for node in sorted_nodes:
            if node.incoming:
                edge, *remotes = node.incoming
                remote_edges += remotes
                if node.is_analyzable():
                    node.preterminal = node.unit = l1.add_fnode(edge.head.unit,
                                                                self.map_label(node=node, deprel=edge.deprel))
                    if any(edge.dep.is_analyzable() for edge in node.outgoing):  # Intermediate head node for hierarchy
                        node.preterminal = l1.add_fnode(node.preterminal, self.map_label(deprel="head"))
                else:  # Unanalyzable: share preterminal with head
                    node.preterminal = edge.head.preterminal
                    node.unit = edge.head.unit

        # Join strong multi-word expressions to one unanalyzable unit
        for smwe in sent["smwes"].values():
            if (smwe["lexcat"], smwe["ss"]) in UNANALYZABLE_MWE_LEXCAT_SS:
                mwe_nodes = [nodes[tok_num] for tok_num in smwe["toknums"]]
                head = min(mwe_nodes, key=sorted_nodes.index)  # Highest in the tree
                for node in mwe_nodes:
                    node.preterminal = head.preterminal

        # Create remote edges if there are any reentrancies (none if not using enhanced deps)
        for edge in remote_edges:
            parent = edge.head.unit or l1.heads[0]  # Use UCCA root if no unit set for node
            child = edge.dep.unit or l1.heads[0]
            if child not in parent.children and parent not in child.iter():  # Avoid cycles and multi-edges
                l1.add_remote(parent, self.map_label(node=edge.dep, deprel=edge.deprel), child)

        # Link preterminals to terminals
        for node in tokens:
            node.preterminal.add(Categories.Terminal, node.terminal)

        return passage

    def map_label(self, node: Optional["Node"] = None, deprel: Optional[str] = None) -> str:
        """
        Map UD relation label to UCCA category.
        :param node: dependency Node that this label corresponds to, containing a `tok' attribute, which is a dict
        :param deprel: UD relation label, alternatively specifying just the dependency relation when no node exists
        :return: mapped UCCA category
        """
        if deprel is None:
            if node is None:
                raise ValueError("Either node or deprel must be specified")
            deprel = node.deprel
        # TODO use supersenses to find Scene-evoking phrases and select labels accordingly
        return UD_TO_UCCA.get(deprel.partition(":")[0], deprel) if self.map_labels else deprel


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

    def __init__(self, tok: Optional[dict]):
        """
        :param tok: conllulex2json token dict (from "toks"), or None for the root
        """
        self.tok: dict = tok  # None for root; dict created by conllulex2json for tokens
        self.position: int = 0  # Position in the sentence (root is zero)
        self.incoming: List[Edge] = []  # List of Edges from heads
        self.outgoing: List[Edge] = []  # List of Edges to dependents
        self.level = self.heads_visited = None  # For topological sort
        self.unit = self.preterminal = None  # Corresponding UCCA units

    def link(self, nodes: List["Node"], enhanced: bool = False) -> None:
        """
        Set incoming and outgoing edges after all Nodes have been created.
        :param nodes: List of all Nodes in the sentences, ordered by position (starting with the root in position 0)
        :param enhanced: whether to use enhanced dependencies rather than basic dependencies
        """
        if self.tok:
            if enhanced:
                self.incoming = [Edge(nodes[int(head)], self, deprel) for head, _, deprel in
                                 [edep.partition(":") for edep in self.tok["edeps"].split("|")]]
            else:
                self.incoming = [Edge(nodes[self.tok["head"]], self, self.tok["deprel"])]
        else:
            self.incoming = []
        for edge in self.incoming:
            edge.head.outgoing.append(edge)

    @property
    def head(self) -> Optional["Node"]:
        """
        Shortcut for getting the Node's primary head if it exists and None otherwise.
        :return: head Node
        """
        return self.incoming[0].head if self.incoming else None

    @property
    def deprel(self) -> Optional[str]:
        """
        Shortcut for getting the Node's primary dependency relation if it exists and None otherwise.
        :return: dependency relation str
        """
        return self.incoming[0].deprel if self.incoming else None

    @property
    def basic_deprel(self) -> Optional[str]:
        """
        Remove any relation subtypes (":" and anything following it) from dependency relation.
        :return: basic dependency relation str
        """
        return self.deprel.partition(":")[0] if self.deprel else None

    def is_analyzable(self) -> bool:
        """
        Determine if the token requires a preterminal UCCA unit. Otherwise it will be attached to its head's unit.
        """
        return self.basic_deprel not in UNANALYZABLE_DEPREL and not (
                self.head.tok and self.tok["upos"] == self.head.tok["upos"] and self.tok["upos"] in UNANALYZABLE_UPOS)

    def __str__(self):
        return "ROOT"

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

    def __str__(self):
        return self.tok['word']


class Edge:
    """
    Edge connecting two Nodes.
    """

    def __init__(self, head: Node, dep: Node, deprel: str):
        """
        :param head: Node this edge comes from
        :param dep: dependent (Node this edge goes to)
        :param deprel: dependency relation (edge label)
        """
        self.head: Node = head
        self.dep: Node = dep
        self.deprel: str = deprel

    def __str__(self):
        return f"{self.head} -{self.deprel}-> {self.dep}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.head}, {self.dep}, {self.deprel})"


def topological_sort(nodes: List[Node]) -> List[Node]:
    """
    Sort dependency nodes into topological ordering to facilitate creating parents before children in the UCCA graph.
    :param nodes: List of all Nodes in the sentences, ordered by position (starting with the root in position 0)
    :return List containing the same nodes but sorted topologically
    """
    levels = {}
    remaining = []
    for node in nodes:
        node.level = None
        node.heads_visited = set()
        if not node.outgoing:  # Start from the leaves
            remaining.append(node)
    while remaining:
        node = remaining.pop()
        if node.level is None:  # Not done with this node yet
            if node.incoming:  # Got a head
                remaining_heads = [edge.head for edge in node.incoming
                                   if edge.head.level is None and edge.head not in node.heads_visited]
                if remaining_heads:
                    node.heads_visited.update(remaining_heads)  # To avoid cycles
                    remaining += [node] + remaining_heads
                    continue
                node.level = 1 + max(edge.head.level or 0 for edge in node.incoming)  # Done with heads
            else:  # Root
                node.level = 0
            levels.setdefault(node.level, set()).add(node)
    return [node for level, level_nodes in sorted(levels.items())
            for node in sorted(level_nodes, key=attrgetter("position"))]  # Sort by level; within level, by position


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


def evaluate(converted_passage, sent, reference_passage, mwe_report=None):
    if mwe_report:
        toks = {tok["#"]: tok for tok in sent["toks"]}
        sent_mwes = {frozenset(mwe["toknums"]): (mwe_id, mwe_type, mwe) for mwe_type in ("smwes", "wmwes")
                     for mwe_id, mwe in sent[mwe_type].items()}
        reference_units = {evaluation.get_yield(unit): unit for unit in reference_passage.layer(layer1.LAYER_ID).all}
        for positions in sorted(set(sent_mwes) | set(key for key, unit in reference_units.items()
                                                     if len(unit.terminals) > 1)):
            mwe_id, mwe_type, mwe = sent_mwes.get(positions, ("", "", {}))
            unit = reference_units.get(positions)
            tokens = [toks[i] for i in sorted(positions)]
            print(reference_passage.ID, " ".join(tok["word"] for tok in tokens), mwe_id, mwe_type,
                  mwe.get("lexcat") or "", mwe.get("ss") or "", mwe.get("ss2") or "",
                  " ".join(tok["deprel"] for tok in tokens),
                  "Yes" if len({tok["head"] for tok in tokens} - positions) <= 1 else "",
                  unit.ID if unit else "", unit.extra.get("tree_id", "") if unit else "", unit.ftag if unit else "",
                  "Yes" if unit and len(unit.terminals) > 1 else "", str(unit) if unit else "",
                  file=mwe_report, sep="\t")
    return evaluation.evaluate(converted_passage, reference_passage)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    sentences = list(load_sents(ConcatenatedFiles(args.filenames)))
    converter = ConllulexToUccaConverter(**vars(args))
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
        if args.mwe_report:
            mwe_report = open(args.mwe_report, "w", encoding="utf-8")
            print("sent_id", "text", "mwe_id", "mwe_type", "lexcat", "ss", "ss2", "deprels", "subtree",
                  "unit_id", "tree_id", "category", "unanalyzable", "annotation", file=mwe_report, sep="\t")
        else:
            mwe_report = None
        scores = [evaluate(converted_passage, sent, reference_passage, mwe_report)
                  for (converted_passage, sent), reference_passage in
                  tqdm(filter(itemgetter(0), passages), unit=" passages", desc="Evaluating", total=len(converted))]
        if mwe_report:
            mwe_report.close()
        evaluation.Scores.aggregate(scores).print()
        print(f"Evaluated {len(scores)} out of {len(converted)} sentences ({100 * len(scores) / len(converted):.2f}%).")
    elif args.mwe_report:
        argparser.error("--mwe-report requires --evaluate")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Convert .conllulex files to UCCA")
    argparser.add_argument("filenames", nargs="+", help=".conllulex file name(s) to convert")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
    argparser.add_argument("-f", "--format", choices=("xml", "pickle", "json"), default="xml", help="output format")
    argparser.add_argument("-v", "--verbose", action="store_true", help="extra information")
    argparser.add_argument("-n", "--no-write", action="store_false", dest="write", help="do not write files")
    argparser.add_argument("-e", "--enhanced", action="store_true", help="use enhanced dependencies rather than basic")
    argparser.add_argument("-m", "--map-labels", action="store_true", help="predict UCCA categories for edge labels")
    argparser.add_argument("--normalize", action="store_true", help="normalize UCCA passages after conversion")
    argparser.add_argument("--extra-normalization", action="store_true", help="apply extra UCCA normalization")
    argparser.add_argument("--evaluate", help="directory/filename pattern of gold UCCA passage(s) for evaluation")
    argparser.add_argument("--mwe-report", help="output filename for report comparing multi-word expressions")
    main(argparser.parse_args())
