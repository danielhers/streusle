#!/usr/bin/env python3
"""
Given a .conllulex file, convert each sentence to a UCCA Passage.
If the script is called directly, outputs the data as XML, Pickle or JSON files.

@since: 2018-12-27
@author: Daniel Hershcovich
@requires: ucca==1.1.7 semstr==1.1.10 scikit-learn==0.20.2 tqdm git+https://github.com/danielhers/depedit
"""

import argparse
import os
import sys
import urllib.request
from itertools import zip_longest
from operator import attrgetter, itemgetter
from typing import List, Iterable, Optional

import numpy as np
from depedit.depedit import DepEdit, ParsedToken
from semstr.convert import iter_files, write_passage
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from ucca import core, layer0, layer1, evaluation
from ucca.ioutil import get_passages
from ucca.layer1 import EdgeTags as Categories
from ucca.normalization import normalize

from conllulex2json import load_sents
from lexcatter import ALL_LEXCATS
from relnoun_lists import RELNOUNS
from supersenses import ALL_SS

SENT_ID = "sent_id"
UD_TO_UCCA = dict(  # Majority-based mapping of UD deprel to UCCA category, from confusion matrix on EWT training set
    acl=Categories.Elaborator, advcl=Categories.ParallelScene, advmod=Categories.Adverbial, amod=Categories.Elaborator,
    appos=Categories.Center, aux=Categories.Function, case=Categories.Relator, cc=Categories.Linker,
    ccomp=Categories.Participant, compound=Categories.Elaborator, conj=Categories.ParallelScene,
    cop=Categories.Function, csubj=Categories.Participant, dep=Categories.Function, det=Categories.Function,
    discourse=Categories.ParallelScene, expl=Categories.Function, fixed=Categories.Center,
    goeswith=Categories.Elaborator, head=Categories.Center, iobj=Categories.Participant, list=Categories.ParallelScene,
    mark=Categories.Function, nmod=Categories.Elaborator, nsubj=Categories.Participant, nummod=Categories.Elaborator,
    obj=Categories.Participant, obl=Categories.Participant, orphan=Categories.Participant,
    parataxis=Categories.ParallelScene, vocative=Categories.Participant, xcomp=Categories.Participant,
    root=Categories.ParallelScene, punct=Categories.Punctuation,
)
DEPEDIT_TRANSFORMATIONS = ["\t".join(transformation) for transformation in [  # Rules to assimilate UD to UCCA
    ("func=/.*/;func=/cc/;func=/conj|root/", "#1>#3;#3>#2", "#1>#2"),  # raise cc over conj, root
    ("func=/.*/;func=/mark/;func=/advcl/", "#1>#3;#3>#2", "#1>#2"),  # raise mark over advcl
    ("func=/.*/;func=/advcl/;func=/appos|root/", "#1>#3;#3>#2", "#1>#2"),  # raise advcl over appos, root
    ("func=/.*/;func=/appos/;func=/root/", "#1>#3;#3>#2", "#1>#2"),  # raise appos over root
    ("func=/.*/;func=/conj/;func=/parataxis|root/", "#1>#3;#3>#2", "#1>#2"),  # raise conj over parataxis, root
    ("func=/.*/;func=/parataxis/;func=/root/", "#1>#3;#3>#2", "#1>#2"),  # raise parataxis over root
]]
DEPEDIT_TRANSFORMED_DEPRELS = {"cc", "mark", "advcl", "appos", "conj", "parataxis"}
UNANALYZABLE_DEPREL = {  # UD dependency relations whose dependents are grouped with the heads as unanalyzable units
    "flat", "fixed", "goeswith",
}
ANALYZABLE_DEPREL = {
    "conj", "list", "vocative",
}
UNANALYZABLE_MWE_LEXCAT_SS = {  # Pairs of multi-word expression lexical category and supersense grouped as unanalyzable
    ("V.VPC.full", "v.social"), ("V.VPC.full", "v.cognition"), ("V.VPC.full", "v.stative"),
    ("V.VPC.full", "v.possession"), ("V.VPC.full", "v.communication"), ("V.VPC.full", "v.change"),
    ("CCONJ", None), ("ADV", None), ("DISC", None),
}
ID2CATEGORY = dict(enumerate(v for k, v in vars(Categories).items() if v and not k.startswith("__")))
CATEGORY2ID = {v: i for i, v in ID2CATEGORY.items()}
ALL_DEPREL = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "dep",
    "det", "discourse", "expl", "fixed", "goeswith", "head", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj",
    "obl", "orphan", "parataxis", "vocative", "xcomp", "root", "punct",
]
ALL_UPOS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
    "VERB", "X",
]
LINKERS = [
    "additionally", "after", "already", "also", "although", "anytime", "anyway", "anyways", "as", "because", "before",
    "but", "cause", "during", "either", "esp", "especially", "etc.", "even", "except", "finally", "first",
    "firstly", "hence", "how", "however", "if", "including", "just", "knowing", "left", "next",
    "only", "overall", "respectively", "seeing", "since", "so", "still", "than", "then", "therefore", "though",
    "throughout", "thus", "too", "unfortunately", "unless", "until", "upon", "well", "what", "when", "whenever",
    "where", "whether", "which", "while", "who", "without", "yet",
]
MWE_TYPES = ("swes", "smwes", "wmwes")
TOP_LEVEL_CATEGORIES = {Categories.Linker, Categories.ParallelScene, Categories.Function, Categories.Ground,
                        Categories.Punctuation}


def read_amr_roles(role_type):
    file_name = "have-" + role_type + "-role-91-roles-v1.06.txt"
    if not os.path.exists(file_name):
        url = "http://amr.isi.edu/download/lists/" + file_name
        try:
            urllib.request.urlretrieve(url, file_name)
        except OSError as e:
            raise IOError(f"Must download {url} and have it in the current directory when running the script") from e
    with open(file_name) as f:
        return [line.split()[1] for line in map(str.strip, f) if line and not line.startswith(("#", "MAYBE"))]


AMR_ROLE = sum((read_amr_roles(role_type) for role_type in ("org", "rel")), [])
ASPECT_VERBS = ['start', 'stop', 'begin', 'end', 'finish', 'complete', 'continue', 'resume', 'get', 'become']
RELATIONAL_PERSON_SUFFIXES = ('er', 'ess', 'or', 'ant', 'ent', 'ee', 'ian', 'ist')


class ConllulexToUccaConverter:
    def __init__(self, enhanced: bool = False, map_labels: bool = False, train: bool = False, model: str = None,
                 **kwargs):
        """
        :param enhanced: whether to use enhanced dependencies rather than basic dependencies
        :param map_labels: whether to translate UD relations to UCCA categories
        :param train: train model to predict UCCA categories
        :param model: input/output filename for model predicting UCCA categories
        """
        del kwargs
        self.enhanced = enhanced
        self.map_labels = map_labels
        self.depedit = DepEdit(DEPEDIT_TRANSFORMATIONS)
        self.train = train
        self.model = model
        if self.model:
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore").fit(list(list(x) for x in zip_longest(
                ALL_DEPREL, ALL_UPOS, sorted(ALL_SS), sorted(ALL_LEXCATS), MWE_TYPES,
                [y for x, y in vars(layer1.NodeTags).items() if y and not x.startswith("__")],
                fillvalue="")))
            if self.train:
                self.classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
            else:
                print(f"Loading model from '{self.model}'", file=sys.stderr)
                self.classifier = joblib.load(self.model)
        else:
            self.classifier = None
        self.features = []
        self.labels = []

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

        # Link tokens to their single/multi-word expressions and vice versa
        exprs = {expr_type: sent[expr_type].values() for expr_type in MWE_TYPES}
        for expr_type, expr_values in exprs.items():
            for expr in expr_values:
                for tok_num in expr["toknums"]:
                    node = nodes[tok_num]
                    node.exprs[expr_type] = expr
                    expr.setdefault("nodes", []).append(node)

        # Create primary UCCA tree
        sorted_nodes = topological_sort(nodes)
        l1 = layer1.Layer1(passage)
        remote_edges = []
        for node in sorted_nodes:
            if node.incoming:
                edge, *remotes = node.incoming
                remote_edges += remotes
                if node.is_analyzable():
                    tags = self.map_label(node, edge)
                    node.unit = node.preterminal = l1.add_fnode_multiple(edge.head.unit, [(tag,) for tag in tags])
                    node.set_extra(self.train)
                    if node.is_possessive_rel() or any(edge.dep.is_analyzable() for edge in node.outgoing):
                        # Intermediate head node for hierarchy
                        tags = self.map_label(node)
                        node.preterminal = l1.add_fnode_multiple(node.preterminal, [(tag,) for tag in tags])
                        node.set_extra(self.train)
                    if node.is_possessive_rel():
                        remote_edges.append(Edge(edge.dep, edge.head, Categories.Participant))
                else:  # Unanalyzable: share preterminal with head
                    node.preterminal = edge.head.preterminal
                    node.unit = edge.head.unit

        # Join strong multi-word expressions to one unanalyzable unit
        for smwe in exprs["smwes"]:
            smwe_nodes = smwe["nodes"]
            head = min(smwe_nodes, key=sorted_nodes.index)  # Highest in the tree
            if (smwe["lexcat"], smwe["ss"]) in UNANALYZABLE_MWE_LEXCAT_SS:
                for node in smwe_nodes:
                    node.preterminal = head.preterminal
            elif smwe["lexcat"] in ("V.VID", "V.IAV"):
                unit_ids = {node.unit.ID for node in smwe_nodes}
                old_node = head.unit
                head.unit = l1.add_fnode(old_node, Categories.Process)
                for edge in old_node.outgoing:
                    if edge.child.ID in unit_ids:
                        head.unit.add_multiple([(tag,) for tag in edge.tags], edge.child, edge_attrib=edge.attrib)
                        old_node.remove(edge)

        # Create remote edges if there are any reentrancies (none if not using enhanced deps)
        for edge in remote_edges:
            parent = edge.head.unit or l1.heads[0]  # Use UCCA root if no unit set for node
            child = next((e.dep.preterminal for e in edge.dep.outgoing if e.basic_deprel == "nsubj"),
                         edge.dep.preterminal)
            if child not in parent.children and parent not in child.iter() and (  # Avoid cycles and multi-edges
                    edge.dep.basic_deprel != "conj" or child not in parent.iter()):
                l1.add_remote_multiple(parent, self.map_label(edge.dep, edge), child)

        # Link preterminals to terminals
        for node in tokens:
            node.preterminal.add(Categories.Terminal, node.terminal)

        # Postprocess graph
        for unit in l1.all:
            self.postprocess(unit)

        return passage

    def map_label(self, node: "Node", edge: Optional["Edge"] = None) -> List[str]:
        """
        Map UD relation label to UCCA categories for a corresponding edge.
        :param node: dependency Node that this label corresponds to, containing a `tok' attribute, which is a dict
        :param edge: optionally, a dependency Edge that the created edge corresponds to
        :return: mapped UCCA categories
        """
        if edge is None:
            deprel = basic_deprel = "head"
        else:
            deprel = edge.deprel
            basic_deprel = edge.basic_deprel
        if not self.map_labels:
            return [deprel]
        if self.model:
            features = node.extract_features(basic_deprel=basic_deprel)
            if not self.train:
                label = self.classifier.predict(self.one_hot_encoder.transform([features]))
                return [ID2CATEGORY[np.item(label)]]
        mapped = [UD_TO_UCCA.get(basic_deprel, deprel)]
        # Use supersenses to find Scene-evoking phrases and select labels accordingly
        if Categories.Center in mapped:
            if node.is_scene_verb() or node.is_scene_noun() or node.lexcat == "DISC":
                mapped = [Categories.Process]
            elif node.lexcat == "ADJ" or node.ss in ("n.STATE", "n.ATTRIBUTE", "n.FEELING"):
                mapped = [Categories.State]
            elif node.tok["upos"] == "VERB":
                if node.ss == "v.stative":
                    objs = [e.dep for e in node.outgoing if e.basic_deprel == "obj"]
                    if objs and objs[0].is_scene_noun():
                        mapped = [Categories.Function]
                    elif node.lexlemma in ("be", "have"):
                        mapped = [Categories.State]
                    else:
                        mapped = [Categories.Process]
                elif node.lexcat in ("V.LVC.full", "V.VID"):
                    mapped = [Categories.Function]
                elif node.lexcat == "V.LVC.cause":
                    mapped = [Categories.Adverbial]
            elif node.is_possessive_rel():
                mapped = [Categories.State, Categories.Participant]
            elif node.is_copular_fragment():
                mapped = [Categories.Participant]
        elif basic_deprel == "conj":
            if node.head.unit and Categories.ParallelScene not in node.head.unit.ftags:
                mapped = [Categories.Center]
        elif not {Categories.Adverbial, Categories.Relator}.intersection(mapped) and (
                node.lexlemma in LINKERS or node.lexcat == "DISC" or node.ss == "p.Purpose"):
            mapped = [Categories.Linker]
        elif node.ss == 'n.TIME':
            mapped = [Categories.Time]
        elif node.lexcat == 'NUM':
            mapped = [Categories.Quantifier]
        elif node.is_possessive_rel() or Categories.Adverbial in mapped and node.ss == "p.Approximator":
            mapped = [Categories.Elaborator]
        elif Categories.Elaborator in mapped:
            if node.head.is_scene_noun():
                mapped = [Categories.Adverbial]
            elif node.head.is_copular_fragment():
                mapped = [Categories.State]
        elif basic_deprel == "mark" and node.lexlemma != "to":
            mapped = [Categories.Relator]
        if basic_deprel == "vocative":
            mapped.append(Categories.Ground)
        return mapped

    def evaluate(self, converted_passage, sent, reference_passage, report=None):
        if report or self.train:
            toks = {tok["#"]: tok for tok in sent["toks"]}
            exprs = {frozenset(expr["toknums"]): (expr_id, expr_type, expr) for expr_type in ("swes", "smwes", "wmwes")
                     for expr_id, expr in sent[expr_type].items()}
            converted_units, reference_units = [{evaluation.get_yield(unit): unit
                                                 for unit in passage.layer(layer1.LAYER_ID).all}
                                                for passage in (converted_passage, reference_passage)]
            for positions in sorted(set(exprs).union(converted_units).union(reference_units)):
                if positions:
                    expr_id, expr_type, expr = exprs.get(positions, ("", "", {}))
                    ref_unit = reference_units.get(positions)
                    pred_unit = converted_units.get(positions)
                    tokens = [toks[i] for i in sorted(positions)]
                    if report:
                        def _join(k):
                            return " ".join(tok[k] for tok in tokens)

                        def _yes(x):
                            return "Yes" if x else ""

                        def _unit_attrs(x):
                            return [x and x.ID, x and x.extra.get("tree_id"),
                                    x and "|".join(sorted(getattr(x, "ftags", [x.ftag]) or ())),
                                    x and "|".join(sorted(t for e in x.incoming if e.attrib.get("remote")
                                                          for t in e.tags)),
                                    _yes(x and len(x.terminals) > 1), x and str(x)]

                        terminals = reference_passage.layer(layer0.LAYER_ID).all
                        fields = [
                            reference_passage.ID,
                            _join("word") or " ".join(terminals.by_position(p).text for p in sorted(positions)),
                            _join("deprel"), _join("upos"), _join("edeps"),
                            expr_id, expr_type, expr.get("lexcat"), expr.get("ss"), expr.get("ss2"),
                            _yes(len({tok["head"] for tok in tokens} - positions) <= 1),
                        ]
                        fields += _unit_attrs(ref_unit) + _unit_attrs(pred_unit)
                        fields += [
                            pred_unit and pred_unit.extra.pop("node", None) or "",
                            _yes(pred_unit and pred_unit.extra.pop("scene_noun", None))
                        ]
                        print(*[f or "" for f in fields], file=report, sep="\t")
                    if self.train and ref_unit and pred_unit and ref_unit.ftag:
                        features = pred_unit.extra.pop("features", None)
                        if features:
                            self.features.append(features)
                            self.labels.append(CATEGORY2ID[ref_unit.ftag])
        return evaluation.evaluate(converted_passage, reference_passage)

    def fit(self):
        if self.train:
            print(f"Fitting model on {len(self.labels)} instances...", file=sys.stderr)
            self.classifier.fit(self.one_hot_encoder.transform(self.features), self.labels)
            joblib.dump(self.classifier, self.model)
            print(f"Saved to '{self.model}'", file=sys.stderr)

    @staticmethod
    def postprocess(unit: layer1.FoundationalNode):
        if unit.participants:
            for edge in unit:
                for category in edge.categories:
                    if category.tag == Categories.Center:
                        category.tag = Categories.Process
        if unit.centers:
            for edge in unit:
                for category in edge.categories:
                    if category.tag in (Categories.Process, Categories.State, Categories.Adverbial):
                        category.tag = Categories.Elaborator
        if unit.parallel_scenes:
            for edge in unit:
                for category in edge.categories:
                    if category.tag == Categories.Connector:
                        category.tag = Categories.Linker
                    # if edge.child.outgoing and edge.child.is_scene():
                    #     category.tag = Categories.ParallelScene
            # non_top_level = [edge for edge in unit if not TOP_LEVEL_CATEGORIES.intersection(edge.tags)]
            # if non_top_level:
            #     child = unit.layer.add_fnode(unit, Categories.ParallelScene)
            #     for edge in non_top_level:
            #         child.add_multiple([(tag,) for tag in edge.tags], edge.child, edge_attrib=edge.attrib)
            #         unit.remove(edge)
        else:  # if not (unit.is_scene() and unit.participants and Categories.ParallelScene in unit.ftags):
            for edge in unit:
                for category in edge.categories:
                    if category.tag == Categories.Linker:
                        category.tag = Categories.Relator if unit.is_scene() else Categories.Connector
        if not unit.incoming:
            for edge in unit:
                for category in edge.categories:
                    if category.tag not in TOP_LEVEL_CATEGORIES:
                        category.tag = Categories.ParallelScene
        if unit.is_scene() or Categories.ParallelScene in (unit.ftags or ()):
            for edge in unit.outgoing:
                if {Categories.ParallelScene, Categories.Linker}.intersection(edge.tags):
                    if not edge.attrib.get("remote"):
                        unit.fparent.add_multiple([(tag,) for tag in edge.tags], edge.child, edge_attrib=edge.attrib)
                    unit.remove(edge)
        if unit.terminals:
            for child in unit.children:
                if child.layer.ID != layer0.LAYER_ID:
                    for grandchild in child.children:
                        unit.add(Categories.Terminal, grandchild)
                    unit.remove(child)
                    if not child.incoming:
                        child.destroy()
        if not unit.centers and not unit.is_scene():
            for edge in unit:
                for category in edge.categories:
                    if category.tag == Categories.Connector:
                        category.tag = Categories.Center
        for edge in unit.outgoing:
            if edge.attrib.get("remote") and \
                    {Categories.ParallelScene, Categories.Connector, Categories.Linker}.intersection(edge.tags):
                unit.remove(edge)


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
        self.incoming: List[Edge] = []  # List of Edges from heads
        self.outgoing: List[Edge] = []  # List of Edges to dependents
        self.level = self.heads_visited = None  # For topological sort
        self.unit = self.preterminal = None  # Corresponding UCCA units
        self.features = None  # For classification of categories

    def link(self, nodes: List["Node"], enhanced: bool = False) -> None:
        """
        Set incoming and outgoing edges after all Nodes have been created.
        :param nodes: List of all Nodes in the sentences, ordered by position (starting with the root in position 0)
        :param enhanced: whether to use enhanced dependencies rather than basic dependencies
        """
        if self.tok:
            self.incoming = [Edge(nodes[self.tok["head"]], self, self.tok["deprel"])]
            if enhanced:
                # DepEdit does not modify enhanced dependencies. Avoid adding back stale edeps by filtering them.
                excluded = DEPEDIT_TRANSFORMED_DEPRELS.intersection({self.tok["deprel"]}) | {"ref"}
                self.incoming += [Edge(nodes[int(head)], self, deprel, enhanced=True) for head, _, deprel in
                                  [edep.partition(":") for edep in self.tok["edeps"].split("|")]
                                  if "." not in head and int(head) != self.tok["head"] and deprel not in excluded]
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
        return self.incoming[0].basic_deprel if self.incoming else None

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
        return None

    @property
    def lexcat(self) -> Optional[str]:
        for expr in self.exprs.values():
            lexcat = expr.get("lexcat")
            if lexcat:
                return lexcat
        return None

    @property
    def lexlemma(self) -> Optional[str]:
        for expr in self.exprs.values():
            lexlemma = expr.get("lexlemma")
            if lexlemma:
                return lexlemma
        return None

    def is_analyzable(self) -> bool:
        """
        Determine if the token requires a preterminal UCCA unit. Otherwise it will be attached to its head's unit.
        """
        return self.basic_deprel in ANALYZABLE_DEPREL or self.basic_deprel not in UNANALYZABLE_DEPREL and not (
                self.head.tok and len({self.tok["upos"], self.head.tok["upos"], "PROPN"}) == 1
                and self.smwe == self.head.smwe or
                (self.tok["upos"] in ("PUNCT", "NUM") or self.ss == "n.TIME") and self.head.ss == "n.TIME" and
                self.head.tok and self.head.tok["upos"] == "PROPN")

    def is_scene_verb(self) -> bool:
        """
        Determine if the node evokes a scene, which affects its UCCA category and the categories of units linked to it
        """
        return self.tok["upos"] == "VERB" and (
                self.basic_deprel not in ("aux", "cop", "advcl", "conj", "discourse", "list", "parataxis")) and (
                self.lexcat not in ("V.LVC.cause", "V.LVC.full")) and not (
                self.ss == "v.change" and self.tok['lemma'] in ASPECT_VERBS)

    def is_scene_noun(self) -> bool:
        if self.ss == "n.PERSON":
            lemma = self.tok['lemma']
            return not self.is_proper_noun() and (lemma.endswith(RELATIONAL_PERSON_SUFFIXES) or
                                                  lemma in AMR_ROLE + RELNOUNS)
        return self.ss in ("n.ACT", "n.EVENT", "n.PHENOMENON", "n.PROCESS")

    def is_proper_noun(self):
        return self.tok['upos'] == 'PROPN' or self.tok['xpos'].startswith('NNP')

    def is_possessive_rel(self):
        return self.lexcat.endswith("POSS") and self.ss in ("p.SocialRel", "p.OrgRole")

    def is_copular_fragment(self):
        return self.tok["upos"] == "NOUN" and self.basic_deprel == "root" and not any(
            e.basic_deprel in ("nsubj", "csubj", "expl") for e in self.outgoing)

    def extract_features(self, basic_deprel: Optional[str] = None) -> np.ndarray:
        self.features = np.array([
            basic_deprel or self.basic_deprel,
            self.tok["upos"],
            self.ss,
            self.lexcat,
            sorted(self.exprs)[0],
            self.unit.tag if self.unit else "",
        ])
        return self.features

    def set_extra(self, train: bool):
        if train and self.features is not None:
            self.preterminal.extra["features"] = list(self.features)
        self.preterminal.extra["node"] = str(self)
        self.preterminal.extra["scene_noun"] = self.is_scene_noun()

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
    def basic_deprel(self) -> Optional[str]:
        """
        Remove any relation subtypes (":" and anything following it) from dependency relation.
        :return: basic dependency relation str
        """
        return self.deprel.partition(":")[0]

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
        if all(edge.enhanced for edge in node.outgoing):  # Start from the leaves
            remaining.append(node)
    while remaining:
        node = remaining.pop()
        if node.level is None:  # Not done with this node yet
            incoming = [edge for edge in node.incoming if not edge.enhanced]
            if node.incoming:  # Got a head
                remaining_heads = [edge.head for edge in incoming if
                                   edge.head.level is None and edge.head not in node.heads_visited]
                if remaining_heads:
                    node.heads_visited.update(remaining_heads)  # To avoid cycles
                    remaining += [node] + remaining_heads
                    continue
                node.level = 1 + max(edge.head.level or 0 for edge in incoming)  # Done with heads
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


def main(args: argparse.Namespace) -> None:
    if args.train and not args.model:
        argparser.error("--train requires --model")
    if args.report and not args.evaluate:
        argparser.error("--report requires --evaluate")
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
    if args.evaluate or args.train:
        passages = ((converted.get(ref_passage.ID), ref_passage) for ref_passage in get_passages(args.evaluate))
        if args.report:
            report = open(args.report, "w", encoding="utf-8")
            print("sent_id", "text", "deprel", "upos", "edeps", "expr_id", "expr_type", "lexcat", "ss", "ss2",
                  "subtree", "ref_unit_id", "ref_tree_id", "ref_category", "ref_remote", "ref_unanalyzable",
                  "ref_annotation", "pred_unit_id", "pred_tree_id", "pred_category", "pred_remote", "pred_unanalyzable",
                  "pred_annotation", "pred_node", "pred_scene_noun",
                  file=report, sep="\t")
        else:
            report = None
        scores = [converter.evaluate(converted_passage, sent, reference_passage, report)
                  for (converted_passage, sent), reference_passage in
                  tqdm(filter(itemgetter(0), passages), unit=" passages", desc="Evaluating", total=len(converted))]
        converter.fit()
        if report:
            report.close()
        evaluation.Scores.aggregate(scores).print()
        print(f"Evaluated {len(scores)} out of {len(converted)} sentences ({100 * len(scores) / len(converted):.2f}%).",
              file=sys.stderr)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Convert .conllulex files to UCCA")
    argparser.add_argument("filenames", nargs="+", help=".conllulex or .json STREUSLE file name(s) to convert")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
    argparser.add_argument("-f", "--format", choices=("xml", "pickle", "json"), default="xml", help="output format")
    argparser.add_argument("-v", "--verbose", action="store_true", help="extra information")
    argparser.add_argument("-n", "--no-write", action="store_false", dest="write", help="do not write files")
    argparser.add_argument("-e", "--enhanced", action="store_true", help="use enhanced dependencies rather than basic")
    argparser.add_argument("-m", "--map-labels", action="store_true", help="predict UCCA categories for edge labels")
    argparser.add_argument("-t", "--train", action="store_true", help="train model to predict UCCA categories")
    argparser.add_argument("--normalize", action="store_true", help="normalize UCCA passages after conversion")
    argparser.add_argument("--extra-normalization", action="store_true", help="apply extra UCCA normalization")
    argparser.add_argument("--evaluate", help="directory/filename pattern of gold UCCA passage(s) for evaluation")
    argparser.add_argument("--report", help="output filename for report of units, subtrees and multi-word expressions")
    argparser.add_argument("--model", help="input/output filename for model predicting UCCA categories")
    main(argparser.parse_args())
