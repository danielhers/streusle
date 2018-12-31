#!/usr/bin/env python3
"""
Given a .conllulex file, convert each sentence to a UCCA Passage.
If the script is called directly, outputs the data as XML, Pickle or JSON files.

@since: 2018-12-27
@author: Daniel Hershcovich
@requires: ucca=1.0.129 semstr==1.1 tqdm
"""

import argparse
import os
from operator import attrgetter
from typing import List, Iterable, Optional

from semstr.convert import iter_files, write_passage
from tqdm import tqdm
from ucca import core, layer0, layer1

from conllulex2json import load_sents

SENT_ID = "sent_id"


def convert(sent: dict) -> core.Passage:
    """
    Create one UCCA passage from a STREUSLE sentence dict.
    :param sent: conllulex2json sentence dict, containing "sent_id" and "toks"
    :return: UCCA Passage where each Terminal corresponds to a token from the original sentence
    """
    passage = core.Passage(ID=sent[SENT_ID])
    l0 = layer0.Layer0(passage)
    tokens = [Token(tok, l0) for tok in sent["toks"]]  # Create terminals
    nodes = [Node(None)] + tokens  # Prepend root
    for node in nodes:  # Link heads to dependents
        node.link(nodes)
    nodes = topological_sort(nodes)
    l1 = layer1.Layer1(passage)
    remote_edges = []
    for node in nodes:  # Create primary UCCA tree
        if node.incoming:
            edge, *remotes = node.incoming
            edge.dep.preterminal = edge.dep.unit = l1.add_fnode(edge.head.unit, edge.deprel)
            remote_edges += remotes
            if node.is_analyzable():
                node.preterminal = l1.add_fnode(node.preterminal, "head")  # Intermediate head for hierarchy
    for edge in remote_edges:  # Create remote edges if there are any reentrancies (none if not using enhanced deps)
        parent = edge.head.unit or l1.heads[0]  # Use UCCA root if no unit set for node
        child = edge.dep.unit or l1.heads[0]
        if child not in parent.children and parent not in child.iter():  # Avoid cycles and multi-edges
            l1.add_remote(parent, edge.deprel, child)
    for node in tokens:
        node.preterminal.add(layer1.EdgeTags.Terminal, node.terminal)
    return passage


class Node:
    """
    Dependency node.
    """
    def __init__(self, tok: Optional[dict]):
        """
        :param tok: conllulex2json token dict (from "toks"), or None for the root
        """
        self.tok = tok  # None for root; dict created by conllulex2json for tokens
        self.position = 0  # Position in the sentence (root is zero)
        self.incoming = []  # List of Edges from heads
        self.outgoing = []  # List of Edges to dependents
        self.level = self.heads_visited = None  # For topological sort
        self.unit = self.preterminal = None  # Corresponding UCCA units

    def link(self, nodes: List["Node"]) -> None:
        """
        Set incoming and outgoing edges after all Nodes have been created.
        :param nodes: List of all Nodes in the sentences, ordered by position (starting with the root in position 0)
        """
        self.incoming = [Edge(nodes[self.tok["head"]], self, self.tok["deprel"])] if self.tok else []
        for edge in self.incoming:
            edge.head.outgoing.append(edge)

    @property
    def head(self) -> Optional["Node"]:
        """
        Shortcut for getting the Node's primary head if it exists and None otherwise
        :return: head Node
        """
        return self.incoming[0].head if self.incoming else None

    @property
    def deprel(self) -> Optional[str]:
        """
        Shortcut for getting the Node's primary dependency relation if it exists and None otherwise
        :return: dependency relation str
        """
        return self.incoming[0].deprel if self.incoming else None

    def is_analyzable(self) -> bool:
        """
        Determine if the token requires a preterminal UCCA unit. Otherwise it will be attached to its head's unit.
        """
        return bool(self.outgoing)


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
        self.position = self.tok["#"]
        self.is_punct = self.tok["upos"] == "PUNCT"
        self.terminal = l0.add_terminal(text=tok["word"], punct=self.is_punct)


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
        self.head = head
        self.dep = dep
        self.deprel = deprel


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
        self.lines = []
        self.name = None
        for filename in iter_files(filenames):
            with open(filename, encoding="utf-8") as f:
                self.lines += list(f)
            self.name = filename

    def __iter__(self) -> Iterable[str]:
        return iter(self.lines)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    sents = list(load_sents(ConcatenatedFiles(args.filenames)))
    t = tqdm(sents, unit=" sentences", desc="Converting")
    for sent in t:
        t.set_postfix({SENT_ID: sent[SENT_ID]})
        passage = convert(sent)
        if args.write:
            write_passage(passage, out_dir=args.out_dir, output_format="json" if args.format == "json" else None,
                          binary=args.format == "pickle", verbose=args.verbose)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Convert .conllulex files to UCCA")
    argparser.add_argument("filenames", nargs="+", help=".conllulex file name(s) to convert")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
    argparser.add_argument("-f", "--format", choices=("xml", "pickle", "json"), default="xml", help="output format")
    argparser.add_argument("-v", "--verbose", action="store_true", help="extra information")
    argparser.add_argument("-n", "--no-write", action="store_false", dest="write", help="do not write files")
    main(argparser.parse_args())
