#!/usr/bin/env python3
"""
Given a .conllulex file, convert each sentence to a UCCA Passage.
If the script is called directly, outputs the data as XML, Pickle or JSON files.

@since: 2018-12-27
@author: Daniel Hershcovich
@requires: ucca==1.1.7 semstr==1.1.10 scikit-learn==0.20.2 tqdm git+https://github.com/danielhers/depedit
"""

import argparse
from typing import List, Optional

from ucca import core, layer1
from ucca.layer1 import EdgeTags as Categories

from conllulex2ucca import Node, Edge, topological_sort, UD_TO_UCCA, UNANALYZABLE_DEPREL, ConllulexToUccaConverter, \
    add_arguments, run


class ConllulexToUccaSimpleConverter(ConllulexToUccaConverter):
    def __init__(self, enhanced: bool = False, map_labels: bool = False, **kwargs):
        """
        :param enhanced: whether to use enhanced dependencies rather than basic dependencies
        :param map_labels: whether to translate UD relations to UCCA categories
        """
        super().__init__(enhanced=enhanced, map_labels=map_labels, **kwargs)

    def convert(self, sent: dict) -> core.Passage:
        """
        Create one UCCA passage from a STREUSLE sentence dict.
        :param sent: conllulex2json sentence dict, containing "sent_id" and "toks"
        :return: UCCA Passage where each Terminal corresponds to a token from the original sentence
        """
        toks = self.depedit_transform(sent)
        passage, nodes, tokens = self.create_passage(sent, toks)

        # Create primary UCCA tree
        sorted_nodes = topological_sort(nodes)
        l1 = layer1.Layer1(passage)
        remote_edges = []
        for node in sorted_nodes:
            if node.incoming:
                edge, *remotes = node.incoming
                remote_edges += remotes
                if node.basic_deprel not in UNANALYZABLE_DEPREL:
                    tags = self.map_label(node, edge)
                    node.unit = node.preterminal = l1.add_fnode_multiple(edge.head.unit, [(tag,) for tag in tags])
                    if any(edge.dep.basic_deprel not in UNANALYZABLE_DEPREL for edge in node.outgoing):
                        # Intermediate head node for hierarchy
                        tags = self.map_label(node)
                        node.preterminal = l1.add_fnode_multiple(node.preterminal, [(tag,) for tag in tags])
                else:  # Unanalyzable: share preterminal with head
                    node.preterminal = edge.head.preterminal
                    node.unit = edge.head.unit

        # Create remote edges if there are any reentrancies (none if not using enhanced deps)
        for edge in remote_edges:
            parent = edge.head.unit or l1.heads[0]  # Use UCCA root if no unit set for node
            child = edge.dep.preterminal
            if child not in parent.children and parent not in child.iter():  # Avoid cycles and multi-edges:
                l1.add_remote_multiple(parent, self.map_label(edge.dep, edge), child)

        # Link preterminals to terminals
        for node in tokens:
            node.preterminal.add(Categories.Terminal, node.terminal)

        return passage

    def map_label(self, node: "Node", edge: Optional["Edge"] = None) -> List[str]:
        """
        Map UD relation label to UCCA categories for a corresponding edge.
        :param node: dependency Node that this label corresponds to, containing a `tok' attribute, which is a dict
        :param edge: optionally, a dependency Edge that the created edge corresponds to
        :return: mapped UCCA categories
        """
        del node
        if edge is None:
            deprel = basic_deprel = "head"
        else:
            deprel = edge.deprel
            basic_deprel = edge.basic_deprel
        return [UD_TO_UCCA.get(basic_deprel, deprel) if self.map_labels else deprel]


def main(args: argparse.Namespace) -> None:
    converter = ConllulexToUccaSimpleConverter(**vars(args))
    run(args, converter)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Convert .conllulex files to UCCA")
    add_arguments(argparser)
    main(argparser.parse_args())
