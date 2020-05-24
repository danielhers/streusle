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
import urllib
from collections import Counter
from itertools import groupby, chain
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
from relnoun_lists import RELNOUNS

import random

SENT_ID = "sent_id"
MWE_TYPES = ("swes", "smwes", "wmwes")


"""
Various syntactic verb/adjective predicates are treated as semantically secondary ("D")
to an embedded predicate, typically an xcomp.

Guidelines pp. 23-24:

BEGINNING: begin,start,finish,complete,continue(with)
TRYING: try,attempt,succeed,fail,practice
WANTING: want,wish(for);hope(for);need,require;expect;intend;pretend
POSTPONING: postpone,delay,defer,avoid
MAKING: make,force,cause,tempt;let,permit,allow,prevent,spare,ensure
HELPING: help,aid,assist

A manually filtered list of xcomp governors from training data:
{'have to': 70, 'want': 45, 'have': 28, 'try': 23, 'need': 23, 'get': 22,
'make': 18, 'let': 16, 'able': 15, 'help': 12, 'hope': 9, 'keep': 8, 'use to': 6, 'willing': 6,
'be suppose to': 5, 'continue': 5, 'end up': 5, 'would like': 5, 'stop': 5}
"""
XCOMP_SECONDARY_HEADS = set(('begin,end,start,stop,finish,commence,complete,continue,'
    'try,attempt,succeed,manage,fail,practice,want,desire,wish,hope,need,require,expect,intend,pretend,'
    'postpone,delay,defer,avoid,make,force,cause,tempt,let,permit,allow,prevent,spare,ensure,'
    'help,aid,assist,'
    'have to,have,get,become,keep,stay,remain,use to,be suppose to,end up,would like,'
    'willing,eager,ready,prepared,inclined,likely').split(','))
    # easy,hard,tough: tough-constructions are treated sometimes as ccomp, sometimes as advcl

# Rules to assimilate UD to UCCA, based on https://www.aclweb.org/anthology/N19-1047/
DEPEDIT_TRANSFORMATIONS = ["\t".join(transformation) for transformation in [
    ("func=/.*/;func=/cc/;func=/conj|root/", "#1>#3;#3>#2", "#1>#2"),         # raise cc        over conj, root
    #("func=/.*/;func=/mark/;func=/advcl/", "#1>#3;#3>#2", "#1>#2"),           # raise mark      over advcl
    # causes problems with "went out of their way to<--take care of him"
    ("func=/.*/;func=/advcl/;func=/appos|root/", "#1>#3;#3>#2", "#1>#2"),     # raise advcl     over appos, root
    ("func=/.*/;func=/appos/;func=/root/", "#1>#3;#3>#2", "#1>#2"),           # raise appos     over root
    ("func=/.*/;func=/conj/;func=/parataxis|root/", "#1>#3;#3>#2", "#1>#2"),  # raise conj      over parataxis, root
    ("func=/.*/;func=/parataxis/;func=/root/", "#1>#3;#3>#2", "#1>#2"),       # raise parataxis over root
]]
DEPEDIT_TRANSFORMED_DEPRELS = set(re.search(r"(?<=func=/)\w+", t).group(0) for t in DEPEDIT_TRANSFORMATIONS)



SNACS_TEMPORAL = ('p.Temporal','p.Time','p.StartTime','p.EndTime','p.Frequency','p.Interval')


def read_amr_roles(role_type):
    file_name = "have-" + role_type + "-role-91-roles-v1.06.txt"
    if not os.path.exists(file_name):
        url = "http://amr.isi.edu/download/lists/" + file_name
        try:
            urllib.request.urlretrieve(url, file_name)
        except OSError as e:
            raise IOError(f"Must download {url} and have it in the current directory when running the script") from e
    with open(file_name) as f:
        entries = set()
        for line in map(str.strip, f):
            if line and not line.startswith(("#", "MAYBE")):
                entry = line.split()[1]
                if '-' in entry:    # tokenize hyphens (and make them optional)
                    entries.add(entry.replace('-', ' - '))
                    entries.add(entry.replace('-', ' '))
                else:
                    entries.add(entry)
        return sorted(entries)


AMR_ROLE = {role_type: read_amr_roles(role_type) for role_type in ("org", "rel")}
ASPECT_VERBS = ['start', 'stop', 'begin', 'end', 'finish', 'complete', 'continue', 'resume', 'get', 'become']

RELATIONAL_PERSON_SUFFIXES = ('er', 'ess', 'or', 'ant', 'ent', 'ee', 'ian', 'ist')
# With simple suffix matching, false positives: fee, flower, list. Require a preceding nonadjacent vowel?

def nonremote(edges):
    edge, = [e for e in edges if not e.attrib.get('remote')]
    return edge

def pass_through_C(unit):
    if unit.ftag=='C':
        return unit.fparent
    return unit

def children_with_cat(unit, cat, remote=False):
    if not remote:
        return [c for c in unit.children if isinstance(c, layer1.FoundationalNode) and c.fparent is unit and c.ftag==cat]
        # the isinstance() check is necessary for some of the reference parses, e.g. "Raging Taco Raging Burrito" missing "&"
    return [c for c in unit.children if any(e.tag==cat for e in c.incoming[1:])]

cwc = children_with_cat

def describe_context(units):
    if not units: return ''
    if len(units)>1:
        return "|".join(sorted(pass_through_C(u).ftag or '-' for u in units))
    u = pass_through_C(units[0])
    par = u.fparent
    if u.ftag in ('P','S'):
        return (f'{u.ftag} ~ A={len(cwc(par,"A"))}({len(cwc(par,"A",True))})'
                          f' T={len(cwc(par,"T"))}({len(cwc(par,"T",True))})'
                          f' D={len(cwc(par,"D"))}({len(cwc(par,"D",True))})')
    elif u.ftag == 'E':
        return (f'{u.ftag} ~ C={len(cwc(par,"C"))}({len(cwc(par,"C",True))})'
                          f' Q={len(cwc(par,"Q"))}({len(cwc(par,"Q",True))})'
                          f' E={len(cwc(par,"E"))-1}({len(cwc(par,"E",True))})')
    elif u.ftag == 'Q':
        return (f'{u.ftag} ~ C={len(cwc(par,"C"))}({len(cwc(par,"C",True))})'
                          f' Q={len(cwc(par,"Q"))-1}({len(cwc(par,"Q",True))})'
                          f' E={len(cwc(par,"E"))}({len(cwc(par,"E",True))})')
    return u.ftag

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

        def print_dep_parse(nodes):
            print([(n,n.head,n.deprel) for n in nodes])


        # Link tokens to their single/multi-word expressions and vice versa
        exprs = {expr_type: sent[expr_type].values() for expr_type in MWE_TYPES}
        for expr_type, expr_values in exprs.items():
            for expr in expr_values:
                for tok_num in expr["toknums"]:
                    node = nodes[tok_num]
                    node.exprs[expr_type] = expr
                    expr.setdefault("nodes", []).append(node)


        unit2expr = {}
        node2unit = {}
        node2unit_for_advbl_attachment = {}   # exceptions to node2unit when attaching syntactic dependents:
        # units for dependents of (node) should attach under (unit)

        def parent_unit_for_dep(node: Node, deprel: str):
            # deprels that are adverbial, i.e., attach to verbs or copular predicates
            ADV_RELS = ('nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp',
                        'obl', 'vocative', 'expl', 'dislocated',
                        'advcl', 'advmod', 'discourse', 'aux', 'cop', 'mark')
            isAdverbialDeprel = deprel in ADV_RELS or deprel.startswith(tuple(r+':' for r in ADV_RELS))
            if isAdverbialDeprel:
                parent = node2unit_for_advbl_attachment.get(node, node2unit.get(node))
            else:
                parent = node2unit.get(node)
            return parent

        def expr_head(expr: dict) -> 'Node':
            """Given an expression, return the first (usually only)
            dependency node that still has a parent (which will be outside the expression)."""
            nodes = expr["nodes"]
            headed_nodes = [n for n in nodes if n.head is not None]
            #if len(headed_nodes)>1:
            #    print('multiheaded:',expr)
            return headed_nodes[0]

        def num_dep_parents(expr: dict) -> 'Node':
            nodes = expr["nodes"]
            headed_nodes = [n for n in nodes if n.head is not None]
            #if len(headed_nodes)>1:
            #    print('multiheaded:',expr)
            return len(headed_nodes)

        l1 = layer1.Layer1(passage)
        dummyroot = l1.add_fnode(None, 'DUMMYROOT')

        # Preprocessing:
        # For cycles involving an MWE: MWE.1 -> x -> MWE.2(expr_head),
        # break the MWE. (It's OK to have MWE.1(expr_head) -> x -> MWE.2, e.g. "take -> care -> x -> of")
        # e.g. "5 out of 5_stars", "kitchen and wait_staff"
        for n in nodes:
            mwe = n.smwe
            if not mwe: continue
            h = expr_head(mwe)
            if h.head and h.head.head and h.head not in mwe["nodes"] and h.head.head in mwe["nodes"]:
                for node in mwe["nodes"]:
                    del node.exprs["smwes"]
                    node.exprs["swes"] = {"toknums": [node.position], "nodes": [node],
                        "lexlemma": node.tok["lemma"],
                        "ss": mwe["ss"], "ss2": mwe["ss2"],
                        "lexcat": mwe['lexcat']}
                    #print(node.exprs)
                    sent['swes'][str(node.position)] = node.exprs["swes"]
                    # keep the lexcat and supersenses, though they may be wrong
                del sent["smwes"][[k for k,v in sent["smwes"].items() if v is mwe][0]]



        for expr in chain(sent['swes'].values(), sent['smwes'].values()):

            lvc_scene_lus = []
            if 'LVC' in expr['lexcat']: # LVC.full or LVC.cause
                # decide the semantic head (scene-evoker)
                lvc_scene_lus = [n for n in expr['nodes'] if n.tok['upos'] not in ('VERB', 'DET')]
                if not lvc_scene_lus:
                    # a couple are V+V (annotation errors?); take the last word
                    lvc_scene_lus = [expr['nodes'][-1]]
                elif len(lvc_scene_lus)>1:
                    # e.g. "went_on_a_trip"
                    lvc_scene_lus = [lvc_scene_lus[-1]]


            if lvc_scene_lus:
                # LVCs are scene-evoking
                outerlu = l1.add_fnode(dummyroot, '+')
                n = expr['nodes'][0]
                unit2expr[outerlu] = n.smwe # could use walrus operator here

                for node in expr['nodes']:
                    if node not in lvc_scene_lus:
                        # light verb or determiner in LVC
                        lu = l1.add_fnode(outerlu, 'D' if expr['lexcat']=='V.LVC.cause' else 'F')
                        lu.add(layer1.EdgeTags.Terminal, node.terminal)
                        # point to parent unit because things should never attach under F units
                        unit2expr[lu] = n.smwe
                        node2unit[node] = outerlu
                    else:   # no "LU" for LVC
                        lu = l1.add_fnode(outerlu, 'UNA')
                        lu.add(layer1.EdgeTags.Terminal, node.terminal)
                        unit2expr[lu] = node.smwe
                        node2unit[node] = outerlu

            elif expr['lexcat']=='PUNCT':
                lu = l1.add_fnode(dummyroot, 'U')
                n = expr['nodes'][0]
                assert n.swe
                unit2expr[lu] = n.swe
                lu.add(layer1.EdgeTags.Terminal, n.terminal)
                node2unit[n] = lu
            else:
                outerlu = l1.add_fnode(dummyroot, 'LU')
                lu = l1.add_fnode(outerlu, 'UNA')
                n = expr['nodes'][0]
                unit2expr[outerlu] = n.swe or n.smwe # could use walrus operator here
                unit2expr[lu] = n.swe or n.smwe

                for node in expr['nodes']:
                    lu.add(layer1.EdgeTags.Terminal, node.terminal)
                    node2unit[node] = outerlu

            for node in expr['nodes']:
                # delete within-MWE dep edges so we don't process them later
                h = node.head   # could use walrus operator
                if h in expr['nodes']:
                    e = node.incoming_basic[0]
                    node.incoming_basic.remove(e)
                    h.outgoing_basic.remove(e)

        # TODO: treat all 'discourse' subtrees as G with no internal structure, like an MWE?
        # TODO: turn remaining 'fixed' attachments into UNA ('so that')

        # decide whether each LU is a scene-evoker ("+") or not ("-")
        # adjectives and predicative prepositions are marked as "S"

        for u in dummyroot.children:
            cat = u.ftag
            if cat=='LU':
                isSceneEvoking = None
                expr = unit2expr[u]

                # TODO: check if UNA nodes are scene-evoking

                n = expr_head(expr)
                if n.lexcat=='N':
                    if n.tok['upos']=='PROPN':
                        isSceneEvoking = False
                    elif n.ss in ("n.ACT", "n.PHENOMENON", "n.PROCESS") or n.ss=='n.EVENT' and n.lexlemma not in ('time','day','night','morning','evening','afternoon'):
                        isSceneEvoking = True   # these will default to P. S evokers like n.ATTRIBUTE below
                    elif n.ss in ('n.PERSON','n.GROUP') and n.lexlemma in AMR_ROLE["rel"] + AMR_ROLE["org"] + RELNOUNS:
                        # relational noun: employee, sister, friend
                        # TODO: really a hybrid scene-nonscene unit. To most syntactic cxns, except perhaps possessives/compounds/PPs,
                        # is is like a non-scene nominal. E.g. we don't want coordination involving a relational noun ("wife and I") to be treated as linkage.
                        isSceneEvoking = True
                    elif n.ss in ('n.PERSON',) and n.lexlemma.endswith(RELATIONAL_PERSON_SUFFIXES): #and ' ' not in n.lexlemma and not n.lexlemma.endswith(('ment','ness')) and re.search(r'[aeiou][^aeiou]+[aeiou]', n.lexlemma, re.I):
                        # probably relational noun: last vowel (cluster) belongs to a suffix, and there is another preceding vowel cluster separated by a consonant
                        # false positives: infant
                        # omit n.GROUP, which includes false positives like "restaurant" and "business"
                        isSceneEvoking = True
                    # TODO: more heuristics from https://github.com/danielhers/streusle/blob/streusle2ucca-2019/conllulex2ucca.py#L585
                elif n.ss and n.ss.startswith('v.'):
                    isSceneEvoking = True
                    # TODO: more heuristics from https://github.com/danielhers/streusle/blob/streusle2ucca-2019/conllulex2ucca.py#L580
                elif n.lexcat=='AUX' and not n.deprel.startswith(('aux','cop')):
                    # auxiliary or copula has been promoted to head (e.g. VP Ellipsis)
                    # "We were told that we couldn't today because they were closing soon."
                    # TODO: really we want an implicit unit
                    isSceneEvoking = True
                else:
                    isSceneEvoking = False

                u._fedge().tag = '+' if isSceneEvoking else '-'

                if n.lexcat=='ADJ' and n.lexlemma not in ('else','such') and n.deprel!='discourse':
                    u._fedge().tag = 'S' # assume this is a state. However, adjectives modifying scenes (D) should not be scene-evoking---correct below
                elif n.deprel=='expl' and n.lexlemma=='there':  # existential there
                    u._fedge().tag = 'S'
                elif n.lexlemma=='be' and any(c.lexlemma=='there' for c in n.children_with_rel('expl')):
                    u._fedge().tag = '-'    # copula serves as clause-root for expletive 'it' and existential 'there'
                    # copula rule below will change this to F
                    exist = n.children_with_rel('expl')[0]  # walrus
                    node2unit_for_advbl_attachment[n] = node2unit[exist]
                    # make 'there' the dep head and detach be-verb/copula
                    e_outer = n.incoming_basic[0]
                    e_inner = exist.incoming_basic[0]
                    assert e_inner.deprel=='expl'
                    # detach copula
                    n.incoming_basic.remove(e_outer)
                    n.outgoing_basic.remove(e_inner)
                    exist.incoming_basic.remove(e_inner)
                    # attach 'there' where copula was
                    e_outer.dep = exist
                    exist.incoming_basic.append(e_outer)
                    # attach copula where 'there' was
                    e_inner.head = exist
                    e_inner.dep = n
                    e_inner.deprel = 'cop'
                    n.incoming_basic.append(e_inner)
                    exist.outgoing_basic.append(e_inner)
                elif n.lexcat=='ADV' and n.deprel!='discourse' and n.children_with_rel('cop'):
                    u._fedge().tag = 'S'
                elif n.lexcat=='DISC' and n.lexlemma=='thanks':
                    u._fedge().tag = 'P'    # guidelines p. 22
                elif 'heuristic_relation' in expr and expr['heuristic_relation']['config'].startswith('predicative') and n.lexlemma!='nothing but' and not (n.ss2=='p.Extent' and n.lexlemma=='as'):
                    # a predicative SNACS relation evoked by u
                    # skip nothing_but, which is treated semantically as P but syntactically as coordination, which causes problems
                    h = n.head  # TODO: in an idiomatic PP this will not be the head noun, it will be ITS head (because we modified the dep parse for MWEs)
                    if n.deprel=='case' or not (h and h.children_with_rel('case')):    # exclude advmod if sister to case, like "*back* between"
                        u._fedge().tag = 'S' # predicative PP, so preposition is scene-evoking
                        if num_dep_parents(expr)>1:
                            pass    # e.g. "in the midst of"
                        elif n.deprel!='root' and not n.deprel.startswith('root:') and n.lexcat!='PP' and expr['heuristic_relation']['obj'] is not None:
                            assert h.lexcat in ('N','PRON','NUM','ADJ'),(h,h.lexcat,str(u),str(l1.root))
                            # adverbial modifiers of the NOMINAL this P case-marks should be semantically under the SNACS-evoked unit
                            node2unit_for_advbl_attachment[h] = u # make preposition unit the attachment site for ADVERBIAL dependents
                elif n.deprel=='cop' and n.head.lexcat in ('N','PRON','NUM') and not n.head.children_with_rel('case'):    # predicate nominal
                    # make copula the head
                    # TODO: exception if nominal is scene-evoking (e.g. "friend", pred. possessive)
                    u._fedge().tag = 'S'
                    h = n.head
                    node2unit_for_advbl_attachment[h] = u
                    # invert headedness of copula and nominal
                    # no need to move other dependents thanks to parent_unit_for_dep()
                    # n = copula, h = nominal
                    assert not n.children
                    # - delete "cop" deprel
                    e = n.incoming_basic[0]
                    n.incoming_basic.remove(e)
                    h.outgoing_basic.remove(e)
                    # - nominal deprel moves to copula
                    e = h.incoming_basic[0]
                    h.incoming_basic.remove(e)
                    e.dep = n
                    n.incoming_basic.append(e)
                    # - make nominal the obj of copula
                    e = Edge(head=n,dep=h,deprel='obj')
                    n.outgoing_basic.append(e)
                    h.incoming_basic.append(e)
                elif n.ss in ("n.ATTRIBUTE", "n.FEELING", "n.STATE"):
                    u._fedge().tag = 'S'


        def decide_nominal_cat(n: Node, parent_unit_cat: str):
            if n.ss=='n.TIME' or n.deprel=='nmod:tmod' or any(c.ss in SNACS_TEMPORAL for c in n.children):
                newcat = 'T' if parent_unit_cat in ('+','S','P') else 'E'
                # TODO: "books *from the 1900s*" would be E?
            elif n.ss in ('p.Approximator','p.Manner','p.Extent') or any(c.ss in ('p.Manner','p.Extent') for c in n.children):
                newcat = 'D' if parent_unit_cat in ('+','S','P') else 'E'
            else:
                newcat = 'A' if parent_unit_cat in ('+','S','P') else 'E'
            return newcat


        ''' # from 2019 converter
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
                                                      lemma in AMR_ROLE["rel"] + AMR_ROLE["org"] + RELNOUNS)
            return self.ss in ("n.ACT", "n.EVENT", "n.PHENOMENON", "n.PROCESS")
        '''





        printMe = False
        if "asdfKim's" in sent['text']:
            printMe = True
            print('000000000', l1.root)



        # functional modifiers, discourse/G

        for u in dummyroot.children:
            cat = u.ftag
            if cat not in ('+', '-', 'S', 'LVC'):
                continue
            assert u in unit2expr,(str(l1.root),str(u))
            expr = unit2expr[u]

            n = expr_head(expr)
            h = n.head
            if h is None:
                continue
            r = n.deprel

            if r in ('fixed','flat'):
                print('Fixed/flat:', n,r,h, sent['text'])

            if r in ('det','aux','cop') or r.startswith(('det:','aux:','cop:')):
                if r=='det' or r.startswith('det:'):
                    assert cat=='-',(str(u),str(l1.root))
                # o.w. USUALLY cat=='-', but there are some uses of 'be' and 'get' marked as v.stative or v.change

                hu = parent_unit_for_dep(h, r)
                assert hu is not None,()
                assert hu is not u,(str(n),r,str(h),str(u),str(l1.root))
                dummyroot.remove(u)
                if r.startswith('det') and expr['lexlemma'] not in ('a','an','the'):
                    hu.add('E', u)  # demonstrative dets are E
                elif r.startswith('aux') and n.lexlemma not in ('be','have','do','will','get'):
                    hu.add('D', u)  # modal auxes are D (but not tense auxes)
                else:
                    # F regardless of whether head's unit is scene-evoking
                    hu.add('F', u)
                    node2unit[n] = hu   # point to parent unit because things should never attach under F units
                #print(f'node2unit[{n}]: {node2unit[n]} -> {hu}')
            elif cat=='-' and (r in ('discourse','vocative') or (n.lexcat=='INTJ' and r=='root')):
                #assert cat=='-',(cat,n,r,h)
                # keep at root
                u._fedge().tag = 'G'

        if printMe:
            print('111111111', l1.root)

        for u in dummyroot.children:
            if u not in dummyroot.children: # removed in a previous iteration
                continue
            cat = u.ftag
            #if cat not in ('+', '-', 'S', 'LVC'):
            #    continue
            assert u in unit2expr,(str(l1.root),str(u))
            expr = unit2expr[u]

            n = expr_head(expr)
            h = n.head
            r = n.deprel


            if h is None:
                continue

            hu = parent_unit_for_dep(h, r)
            assert hu is not u,(n,r,h,str(u),str(l1.root))
            if r in ('amod','advmod','acl','advcl') or r.startswith(('amod:','advmod:','acl:','advcl:')):    # includes acl:relcl
                if cat=='-':
                    hucat = hu.ftag
                    dummyroot.remove(u)
                    if hucat in ('+','S','P'):
                        if n.lexlemma in ("here", "there", "nowhere", "somewhere", "anywhere", "everywhere") and n.tok['upos']=='ADV':
                            # locative pro-adverbs are "A"
                            hu.add('A', u)
                        elif n.ss in SNACS_TEMPORAL or n.ss=='n.TIME':
                            hu.add('T', u)
                        elif n.lexlemma in ("now", "yesterday", "tomorrow", "early", "earlier", "late", "later",
                            "before", "previous", "previously", "after", "subsequent", "subsequently", "recent", "recently",
                            "soon", "eventual", "eventually", "already", "yet",
                            "often", "frequent", "infrequent", "frequently", "infrequently", "rare", "rarely", "current", "ongoing",
                            "sometime", "anytime", "everytime", "ever", "never"):
                            # temporal deictic and frequency adjectives/adverbs are "T"
                            hu.add('T', u)
                        else:
                            hu.add('D', u)
                    else:
                        hu.add('E', u)
                elif cat=='S' or cat=='+':
                    # adjectives are treated as S thus far
                    # cat=='+' for e.g. a *personalized* gift
                    hucat = hu.ftag
                    dummyroot.remove(u)

                    if hucat=='S' and (r=='amod' or r.startswith('amod:')):  # probably an ADJ compound, like African - American
                        hu.add('C', u)  # put "African" as C under "American"---that will later become C too
                    elif n.children_with_rels('mark') and n.children_with_rels('mark')[0].lexlemma not in ('to','that'):
                        # OMG walrus
                        marker = n.children_with_rels(('case','mark'))[0]
                        # marker should be a linker, potentially an MWE like "for the sake of" (Purpose)
                        # so we want to attach as SISTER to head, not under it
                        (hu.fparent or hu).add(cat, u)
                    elif n.children_with_rels(('case','mark')) and n.children_with_rels(('case','mark'))[0].ss in ('p.Purpose','p.Explanation','p.ComparisonRef'):
                        # like above case
                        marker = n.children_with_rels(('case','mark'))[0]
                        (hu.fparent or hu).add(cat, u)
                    elif hucat in ('-','A','E'):  # E-scene. make node for scene to attach as E
                        scn = l1.add_fnode(hu, 'E')
                        scn.add(cat, u)
                    elif hucat in ('+','S','P'):
                        hu.add('D', u)  # e.g. aspectual verb particle
                    else:
                        # maybe not a problem--allow the E strategy under all non-scene units?
                        # TODO: hucat=='S' + advmod
                        # TODO: 'when' advcl is apparently an annotation error
                        # '10 more minutes'
                        #assert h.lexlemma=='when',(hucat,str(hu),cat,str(u),n,n.deprel,n.head,str(l1.root))

                        # e.g. "Thanks for *doing* such great work..."
                        # make an A-scene
                        # TODO: revisit
                        #newu = l1.add_fnode(hu, 'A')
                        #newu.add(cat, u)
                        hu.add(cat, u)
                else:
                    assert False,(n,r,h,sent['text'])
            elif r=='nummod':
                hucat = hu.ftag
                dummyroot.remove(u)
                hu.add('Q' if hucat not in ('+','S','P') else 'D', u)
            elif r=='nmod:npmod' and expr['lexlemma'].endswith('self'):
                # the surgery *itself*: attach as F (guidelines p. 31, "Reflexives")
                assert cat=='-'
                dummyroot.remove(u)
                hu.add('F', u)
            elif (r=='case' or r=='nmod:poss') and n.lexcat in ('P', 'PP', 'POSS', 'PRON.POSS', 'INF.P') and n.ss not in ('p.Explanation','p.Purpose','p.ComparisonRef') and not (n.ss2=='p.Extent' and n.lexlemma=='as'):
                assert 'heuristic_relation' in expr,(expr,sent[SENT_ID],sent['text'],'Need to run govobj.py?')
                go = expr['heuristic_relation']
                govi = go['gov']
                obji = go['obj']
                config = go['config']
                if config.startswith('predicative'):
                    # predicative PP: prep is the scene-evoker
                    assert cat=='S',str(l1.root)
                    # move its gov and obj underneath, unless gov is scene-evoking
                    if govi is not None:
                        gov = nodes[govi]
                        govu = node2unit[gov]
                        govcat = govu.ftag
                        #assert govcat not in ('+','S','P'),(govcat,str(govu),str(l1.root))
                        assert govu.fparent is dummyroot,(govcat,str(govu),str(l1.root))
                        dummyroot.remove(govu)
                        u.add(govcat, govu)
                        #print(govcat,str(govu),str(l1.root))
                    if obji is not None and 'stranded' not in config:
                        # TODO: stranded preps require a remote unit
                        obj = nodes[obji]
                        obju = node2unit[obj]
                        objcat = obju.ftag
                        #assert objcat not in ('+','S','P'),(objcat,str(obju),str(l1.root))
                        assert obju.fparent is dummyroot,(str(u),go,objcat,str(obju),str(l1.root))
                        if objcat=='-':
                            objcat = decide_nominal_cat(n, '+') # TODO: check '+' is correct for pred. complement
                        dummyroot.remove(obju)
                        u.add(objcat, obju)
                        #assert False,(objcat,str(obju),str(l1.root))
                    # TODO: T and D PPs (The meeting was on Thursday)
                elif obji is not None:
                    obj = nodes[obji]
                    obju = node2unit[obj]
                    dummyroot.remove(u)
                    if n.lexcat=='POSS' and n.ss in ('p.Possessor',):
                        # ownership -> possessive particle is scene-evoking
                        gov = nodes[govi]
                        govu = node2unit[gov]

                        #govu.add('S', u)
                        objcats = obju.ftags
                        obju.fparent.remove(obju)
                        escn = l1.add_fnode(govu, 'E')   # possessive = E-scene
                        #govu.add_multiple([('_',),('S',)], u)   # _ to block further rules from treating this as a main scene wrapped in H
                        escn.add('S', u)
                        possscn = u
                        if objcats==['E'] or objcats==['-']:
                            u.add('A', obju)
                        else:
                            objwrapu = l1.add_fnode(u, 'A')
                            objwrapu.add_multiple(list(map(tuple, objcats)), obju)
                        # remote from possession scene to the lexical head of govu
                        # e.g. [E [S [A [UNA Kim]] [UNA 's] [A* business cards] ] ] [C [UNA business cards]]
                        t, = layer1._multiple_children_by_tag(govu, 'UNA')
                        l1.add_remote(possscn, 'A', t)
                    #elif n.ss=='p.Extent':
                    #    assert False,(str(n),str(l1.root))
                    else:   # make prep/possessive a relator under its object
                        obju.add('R', u)
                elif config=='possessive':  # possessive pronoun
                    gov = nodes[govi]
                    govu = node2unit[gov]
                    gucat = govu.ftag

                    dummyroot.remove(u)
                    if gucat in ('+','S','P'):  # scene-evoking head noun: relational or eventive
                        govu.add('A', u)
                    elif n.ss=='p.Originator' and gov.ss=='n.COMMUNICATION':
                        # possessive + communication noun ambiguous between process and content ("my answer/order/request/...")
                        # TODO: change the communication noun to P (maybe do this above)
                        govu.add('A', u)
                    elif n.ss in ('p.Possessor','p.Originator','p.Recipient','p.SocialRel','p.OrgMember'):
                        # possessive pron marked S+A embedded in E
                        # - Possessor = ownership scene
                        # - SocialRel with non-relational head noun (e.g. "MY tire guy")
                        # - OrgMember (e.g. "my company")
                        # - Originator (e.g. "their food")
                        # TODO: double-check this is the right policy for OrgMember (data appear to be inconsistent)
                        possscn = l1.add_fnode(govu, 'E')
                        possscn.add_multiple([('A',),('S',)], u)   #A|S. Put the A first so other rules don't treat this like a scene and wrap in H

                        # remote to the lexical head of govu
                        t, = layer1._multiple_children_by_tag(govu, 'UNA')
                        l1.add_remote(possscn, 'A', t)
                    else:
                        govu.add('E', u)
                else:
                    assert False,expr
            elif r in ('nmod','compound') or r.startswith(('nmod:','compound:')):    # misc nmod--see rules for obl
                # TODO: scene nmod
                # TODO: 'X of Y' nmod relations (p. 13)
                hucat = hu.ftag
                if cat not in ('+','S','P'):
                    assert cat=='-',(cat,str(u))
                    newcat = decide_nominal_cat(n, hucat)
                    dummyroot.remove(u)
                    hu.add(newcat, u)
                elif hucat in ('+','S','P'):
                    # scene modifying scene: D seems like the best bet
                    dummyroot.remove(u)
                    hu.add('D', u)
                else:
                    # scene modifying nonscene: E-scene
                    newu = l1.add_fnode(hu, 'E')
                    dummyroot.remove(u)
                    newu.add(cat, u)
                    # TODO: consider adding head noun under newu as remote
            elif r=='appos' and cat not in ('+','S','P'):
                if n.tok["upos"]=='PROPN':
                    # TODO: make the head an E
                    #assert False,('PROPN as C in apposition:',cat,n.lexlemma,h.lexlemma)
                    pass
                if True:
                    # make this an E
                    assert cat=='-',(cat,n,str(u),hucat,h,str(hu))
                    dummyroot.remove(u)
                    hu.add('E', u)

        if printMe:
            print('222222222', l1.root)

        # TODO: consider order of operations for case vs. nmod/amod/etc.

        # if 'feed my cat' in sent['text'] or 'nothing but frustrating' in sent['text']:
        #     print(l1.root)

        # argument structure

        for u in dummyroot.children:
            cat = u.ftag
            #if cat not in ('+', '-', 'S', 'LVC'):
            #    continue
            assert u in unit2expr,(str(l1.root),str(u))
            expr = unit2expr[u]
            n = expr_head(expr)
            r = n.deprel
            h = n.head

            if h is None:
                continue
            hu = parent_unit_for_dep(h, r)
            hucat = hu.ftag if hu else None
            if printMe and 'for' in str(u):
                print('xxxxxxxxx', n,n.ss,r,h,cat,str(u),hucat,str(hu), '      ', str(l1.root))
            if cat=='-' and (r in ('nsubj','obj','iobj','xcomp') or r.startswith(('nsubj:','obj:','iobj:','xcomp:'))):
                # N.B. nominal xcomp can occur with verbs like "make", "call", "name", "deem", "consider", etc.
                if hucat in ('+','S','P'):
                    # am I a relative pronoun?
                    if h.deprel=='acl:relcl' and n.lexcat=='PRON' and n is sorted(h.children, key=lambda x: x.position)[0] and n.lexlemma in ('that','who','whom','what','when','where','why','how'):
                        # leftmost dependent of relative clause head and a relative pronoun type...so probably
                        # relative pronouns should be R
                        newcat = 'R'
                    elif hucat in ('+','S','P'):
                        newcat = 'A'
                    else:
                        newcat = 'E'

                    dummyroot.remove(u)
                    hu.add(newcat, u)
                else:
                    # TODO: predicate nominals
                    #assert False,(str(u),h,node2unit[h],hucat,str(l1.root))
                    pass
            elif cat in ('+','S','P') and hucat in ('+','S','P') and (r in ('nsubj','obj','iobj','csubj','ccomp','xcomp') or r.startswith(('nsubj:','obj:','iobj:','csubj:','ccomp:','xcomp:'))):
                # syntactically core args
                if n.head.lexlemma in XCOMP_SECONDARY_HEADS:
                    subsceneu = l1.add_fnode(hu, '^')   # "^": an indication to promote the embedded scene to primary and demote the parent scene head to "D"
                else:
                    # TODO: remote, typically
                    subsceneu = l1.add_fnode(hu, 'A')
                dummyroot.remove(u)
                subsceneu.add(cat, u)
            elif r=='expl' or r.startswith('expl:'):
                assert expr["lexlemma"] in ('it','there'),(expr,sent['text'])
                if expr["lexlemma"]=='it':
                    assert cat=='-'
                    # expletive 'it' -> F
                    dummyroot.remove(u)
                    hu.add('F', u)
                elif expr["lexlemma"]=='there':
                    assert cat=='S'
                    # existential 'there' -> S
                    dummyroot.remove(u)
                    hu.add('S', u)
            elif r=='mark' and n.lexlemma in ('to','that') and n.ss!='p.Purpose':
                if hucat=='-':
                    print('Weird attachment of inf/complementizer to non-scene unit',expr,str(hu),str(l1.root))
                dummyroot.remove(u)
                hu.add('F' if n.lexlemma=='to' else 'R', u) # infinitive TO = F, complementizer THAT = R
            elif cat=='-' and (r=='mark' or n.lexcat == "DISC" or n.ss in ('p.Purpose','p.Explanation','p.ComparisonRef')):
                # adpositional or infinitive linker
                # (coordination linkers are handled below)
                dummyroot.remove(u)

                # p.Purpose can also be used with a non-scene unit object,
                # in which case it should be R rather than L
                if n.ss=='p.Purpose':
                    obji = expr['heuristic_relation'].get('obj')
                    obju = node2unit[nodes[obji]]
                    if obju.ftag not in ('+','P','S'):  #(obju.ftag,str(obju),sent['text'])
                        obju.add('R', u)
                        continue

                if hu is None:
                    dummyroot.add('L', u)
                else:
                    # add as sister to dependency head
                    (hu.fparent or hu).add('L', u)
            elif r=='obl' or r.startswith('obl:'):
                if 'wait staff' in str(u):
                    print(n,r,h,cat,str(u),hucat,str(hu))
                if cat not in ('+','S','P'):
                    assert cat=='-',(cat,str(u))
                    if any(c.ss in ('p.Purpose','p.Explanation','p.ComparisonRef') for c in n.children_with_rels(('case','mark'))):
                        continue #print(str(n)) # skip now, make this a Linker below

                    if n.ss=='n.TIME' or r=='obl:tmod' or any(c.ss in SNACS_TEMPORAL for c in n.children):
                        newcat = 'T'
                    elif n.ss in ('p.Approximator','p.Manner','p.Extent') or any(c.ss in ('p.Manner','p.Extent') for c in n.children):
                        newcat = 'D'    # manner adverbials
                    else:
                        newcat = 'A'

                    dummyroot.remove(u)
                    if hu is None:
                        hu = dummyroot
                    hu.add(newcat, u)
                else:
                    # TODO: not sure about this
                    dummyroot.remove(u)
                    if hu is None:
                        hu = dummyroot
                    hu.add('D', u)


        # for u in dummyroot.children:
        #     cat = u.ftag
        #     assert u in unit2expr,(str(l1.root),str(u))
        #     expr = unit2expr[u]
        #     n = expr_head(expr)
        #     r = n.deprel
        #     if cat=='-' and r in ('root','parataxis'):  # be sure not to filter to non-None heads
        #         u._fedge().tag = 'H'    # sentence/fragment is H by default





        # coordination and connectives

        if printMe:
            print('333333333', l1.root)

        for u in l1.all:
            assert u is dummyroot.fparent or u.incoming and (all(e.tag==layer1.EdgeTags.Terminal for e in u.incoming) or nonremote(u.incoming)),(str(u),str(l1.root))


        def dfs_order_subtree(unit):
            """Traverse primary foundational unit descendents in DFS order"""
            items = [unit]
            for e in unit.outgoing:
                if not e.attrib.get('remote') and e.tag!=layer1.EdgeTags.Terminal:
                    items.extend(dfs_order_subtree(e.child))
            return items

        # by now we should have processed all dependencies except for conj and cc
        # so we can safely wrap the coordination structure in [COORD [X ...] [CONJ ...] [Y ...]]
        # where X and Y are conjuncts and CONJ is the conjunction/connective (N or L)

        # traverse top-down so none of the conjuncts will themselves be coordination structures;
        # the conjunct's head will correspond to a lexically-evoked unit
        processed_conj_heads = []   # to make sure we don't revisit a node via multiple units
        for u in dfs_order_subtree(dummyroot):
            try:
                assert u is dummyroot.fparent or u.incoming and (all(e.tag==layer1.EdgeTags.Terminal for e in u.incoming) or nonremote(u.incoming)),(str(u),str(l1.root))
            except:
                print(str(u),str(l1.root))
                raise

            if u not in unit2expr:
                continue
            expr = unit2expr[u]
            n = expr_head(expr)
            cat = u.ftag

            if not n.children_with_rels(('cc','cc:preconj','conj')) or n in processed_conj_heads:
                continue

            # n is the first conjunct node, with other conjuncts and ccs as daughters

            # h = n.head
            # r = n.deprel
            # hu = parent_unit_for_dep(h, r)
            hu = u.fparent
            if any(node2unit[c] is u or node2unit[c] is hu for c in n.children_with_rel('conj')):
                # due presumably to an MWE, one of the conjuncts also belongs to this unit or the parent unit
                # so go up a level
                hu = hu.fparent

            hucat = hu.ftag

            if printMe:
                print('yyyyyyyyy', n,n.ss,n.deprel,n.head,cat,str(u),hucat,str(hu), '      ', str(l1.root))


            # create coordination wrapper unit under n's parent
            coordu = l1.add_fnode(hu, f'{cat}(COORD)')
            # move u underneath
            u.fparent.remove(u)
            conjcat = cat if cat!='-' else 'C'
            coordu.add(conjcat, u)  # TODO: revisit cat and hucat


            processed_children = []
            for c in n.children_with_rel('conj'):
                conj = node2unit[c]
                assert conj is not coordu.fparent,c
                if conj not in processed_children:
                    ccat = conjcat if conj.ftag=='-' else conj.ftag
                    assert ccat!='-',(ccat,str(conj))
                    conj.fparent.remove(conj)
                    coordu.add(ccat, conj)
                    processed_children.append(conj)

            for c in n.children_with_rels(('cc','cc:preconj')):
                cc = node2unit[c]
                if cc not in processed_children:
                    # skip if UNA containing both coordinator and conjunct (processed above), e.g. "and company"
                    ccat = 'L' if conjcat in ('+','P','S') else 'N'
                    cc.fparent.remove(cc)
                    coordu.add(ccat, cc)
                    processed_children.append(cc)

            processed_conj_heads.append(n)


            if printMe:
                print('444444444', l1.root)


        # decide S or P for remaining "+" scenes



        hasSec = any(n for n in l1.all if n.ftag in ('^',))



        for u in l1.all:
            assert u is dummyroot.fparent or (u.incoming and nonremote(u.incoming)),(str(u),str(l1.root))
            if u.ftag and u.ftag=='+':  # TODO: startswith('+') for +(COORD)
                rest = u.ftag[1:]   # could be +(COORD)
                expr = unit2expr[u]
                n = expr_head(expr)
                if n.ss=='v.stative' and n.lexlemma in ('be','have'):
                    u._fedge().tag = 'S' + rest
                elif n.ss.startswith('v.'):
                    u._fedge().tag = 'P' + rest
                elif n.ss in ("n.ACT", "n.PHENOMENON", "n.PROCESS", "n.EVENT"):
                    u._fedge().tag = 'P' + rest
                elif n.lexcat=='AUX':   # e.g. due to ellipsis
                    u._fedge().tag = 'P' + rest
                else:
                    u._fedge().tag = 'S' + rest
                    print('Weird scene, defaulting to S:',u,u.ftag,n,expr)

        #assert not lvc_scene_lus,l1.root

        # Where an embedded scene was marked as "^", it means the main scene predicate
        # should be considered secondary and the embedded scene promoted.
        for u in l1.all:
            if u.ftag and u.ftag.startswith('^'):
                assert u.ftag=='^',str(l1.all)
                pu = u.fparent
                for c in pu.children:
                    if c.ftag=='UNA':
                        # UNA and putative scene head--demote to D
                        pu.remove(c)
                        pu.add('D', c)
                if pu.ftag in ('S','P'): # exclude S(COORD) and P(COORD) to avoid complications
                    # merge the child scene with the parent scene
                    pu.remove(u)
                    if len(u.children)==1 and u.children[0].ftag in ('S','P'):
                        # promote scene type to parent unit
                        c = u.children[0]
                        pu._fedge().tag = c.ftag
                        u.remove(c)
                        u = c
                    for c in u.children:
                        ccat = c.incoming[0].tag
                        u.remove(c)
                        pu.add(ccat, c)

        #if hasSec:
        #    print('__',l1.root, file=sys.stderr)

        if printMe:
            print('555555555', l1.root)

        # ARTICULATION
        # We currently have a version of the UCCA graph where units are headed
        # by bare terminals. They need proper categories (S, P, C, etc.)
        # and their parent units need to be adjusted, as well as coordination
        # structures.

        # Traverse the graph bottom-up:
        # Raise terminals/UNA so they have a unary unit with a foundational unit category
        # If parent unit is a scene (P or S), use that category label
        #    and change the scene unit category to H.
        # Else if the parent unit has category `-`, change it to C.
        # Else create a unary C node to wrap the lexical unit.
        #
        # Leave COORD alone?
        #
        # N.B. This will create some superfluous unary nesting e.g. [A [H ...]], which we remove later.

        h_units_to_relabel = []
        for u in dfs_order_subtree(dummyroot)[::-1]:    # bottom-up

            if u.ftag=='UNA':
                # raise me!

                cat = nonremote(u.incoming).tag

                pu = nonremote(u.incoming).parent   # .fparent only if a foundational node (UNA)
                pucat = pu.ftag

                if pu is dummyroot or pucat in ('+','H'):
                    assert False,(pucat,str(u),str(l1.root))
                elif pucat in ('P','S','H(P)','H(S)'):
                    # copy the category and change the parent unit category to H
                    # temporarily mark the parent H(P) or H(S) in case there are
                    # other siblings to be processed
                    if pucat in ('P','S'):
                        pu._fedge().tag = f'H({pucat})'
                        h_units_to_relabel.append(pu)
                    pucat = pu.ftag
                    newcat = {'H(P)': 'P', 'H(S)': 'S'}[pucat]
                    newu = l1.add_fnode(pu, newcat)
                    pu.remove(u)
                    newu.add(cat, u)
                elif pucat=='-':
                    if len(pu.children)==1: # change to C
                        pu._fedge().tag = 'C'
                        newu = pu
                    else:
                        newu = l1.add_fnode(pu, 'C')
                        pu.remove(u)
                        newu.add(cat, u)
                elif len(pu.children)>1:   # add an intermediate unary C unit
                    newu = l1.add_fnode(pu, 'C')
                    pu.remove(u)
                    newu.add(cat, u)
                else:
                    continue


        if printMe:
            print('666666666', l1.root)

        for hunit in h_units_to_relabel:
            cat = hunit.ftag
            newcat = {'H(P)': 'H', 'H(S)': 'H'}[cat]
            hunit._fedge().tag = newcat



        if printMe:
            print('777777777', l1.root)

        for u in l1.all:
            if u.ftag=='UNA':
                if len(u.fparent.children)==1:  #,(str(u),str(u.fparent),str(l1.root))
                    pass
                else:
                    print('STRUCTURE VIOLATION:', str(l1.root), file=sys.stderr)



        if printMe:
            print('999999999', l1.root)


        # clean up punctuation

        for u in dummyroot.children:
            cat = u.ftag
            if cat=='U':
                expr = unit2expr.get(u)
                if not expr: continue
                n = expr_head(expr)
                h = n.head
                if h is None:
                    continue
                r = n.deprel
                if r=='punct':  # and h.deprel!='root':
                    hu = parent_unit_for_dep(h, r)
                    if hu.fparent:
                        hu = hu.fparent
                    dummyroot.remove(u)
                    hu.add('U', u)




        if printMe:
            print('aaaaaaaaa', l1.root)




        # remove "COORD" designations and replace "+" with "H"
        for u in l1.all:
            if u.incoming:
                cat = nonremote(u.incoming).tag
                cat = cat.replace('(COORD)','')
                if cat=='+':
                    cat = 'H'
                nonremote(u.incoming).tag = cat



        # ensure top-level cat is valid
        toplevel = [c for c in dummyroot.children if c.ftag!='U']
        # top-level units must be H or L, so change - ones to H
        for u in toplevel:
            if u.ftag=='-':
                u._fedge().tag = 'H'
                # TODO: add implicit unit?

        if printMe:
            print('bbbbbbbbb', l1.root)

        # remove "H" under unary parent (except at top level)
        for u in l1.all:
            if u.incoming and nonremote(u.incoming).tag=='H' and u not in toplevel:
                pu = u.fparent
                if len(pu.children)==1:
                    assert pu.ftag in ('A','E','H'),(pu.ftag,str(pu))
                    pu.remove(u)
                    for e in u.outgoing:
                        u.remove(e.child)
                        pu.add_multiple(list(map(tuple, e.tags)), e.child, edge_attrib=e.attrib)


        # Remove "UNA" by merging with unary parent
        # The UNA-child unit may be a remote elsewhere, so instead of discarding it
        # we replace the unary primary parent with it
        for u in l1.all:
            cat = u.ftag
            if cat=='UNA':
                pu = u.fparent
                assert len(pu.children)==1
                assert len(pu.incoming)==1
                pucat = pu.ftag
                gpu = pu.fparent
                gpu.remove(pu)
                gpu.add(pucat, u)





        KEYWORDS = ('good dentists in Fernandina','Esp. the mole','mircles','ufc fighter','kitchen and wait staff',
            'For example','would have been distorted','surgery','eather','feed my cat',
            'of the ants','patrons willing','were back quickly','worst Verizon store',
            'as proper as','Raging Taco')
        if any(kwd in sent['text'] for kwd in KEYWORDS) or random.random()>.9:
            print(l1.root)

        if printMe:
            print(l1.root)
            assert False

        if any(n for n in dummyroot.children if n.ftag in ('+','-')):
            print(l1.root)
            assert False
        elif any(n for n in l1.all if n.ftag in ('+','-')):
            print(l1.root)





        '''
        1. Identify (unanalyzable) units: expressions with lexcats …
            For unanalyzable MWEs, note which token is the syntactic head and check that none of the other tokens in the MWE have dependants outside of the MWE
        2. Intermediate annotation for each lexical unit: scene-evoking or not. For LVCs, mark the syntactic head as an F modifier of the scene-evoking item.
        3. Handle coordinations of non-scene unit evokers, marking coordinators as N and the conjunct heads as C
        4. Form non-scene units with C, E, Q, R, F, T, etc. elements, with special attention to (a) possessives, (b) relational nouns, …
            For prepositions/possessives, consider using govobj information which deals with things like copular predicates and stranding
        5. Handle coordinations where at least one conjunct is scene-evoking
        6. Form scene units headed by each scene-evoking lexical unit: A, D, T, F, etc., with special attention to (a) copula constructions, (b) nominally-evoked scenes and possessive participants, (c) relative clauses, ...
        7. Decide whether each scene-evoking lexical unit is P or S
        8. Remaining parallel scenes and linkage
        9. (G? Anaphora? Implicit units? Secondary edges?)
        '''

        """
        TODO: Uncovered STREUSLE 4.3 annotation errors in training set

        ## LVCs
        - Some LVCs include the article, e.g. have_a_bite
        - has_ 4 studios _planned is tagged as LVC - is "has" a regular verb here (cf. "get")?
        - had_ ... _replace
        - 'go on a trip' marked as an LVC

        - 'get busy': should not be aux (not passive get)
        - 'getting infected': 'getting' is v.body, but that should only apply if an MWE
        - 'such': inconsistent treatment: sometimes ADJ/det (such nice workers)
        - 'so that': mostly not treated as MWE, but always fixed. should change to so_that

        - 'Huge ammount of time wasted time': first 'time' is case
        - 'Beautifully written reviews Doctor, but completely UNTRUE.' Doctor is flat, should be vocative

        - when + clause: when often advmod when it should be mark

        - preposition stranding not consistent: ADP <case@L _ vs. ADP <obl _
            govobj.py workaround
        - ADV <advmod (NOUN|PRON !>cop _)   -- advmod attachments to nouns. many look weird

        - typos: !L=there&!L=it <expl _

        - 'once a week': once as case

        FIXED
        - amod(PRON,ADJ) should be xcomp(VERB,ADJ): "saw it *riddled* with..."
        - typo expl THERE


        STREUSLE/UD LIMITATIONS
        1) Which adverbs are discourse connectives and which are within-clause modifiers.
        2) n.COGNITION, n.COMMUNICATION, n.POSSESSION are broad semantic fields covering entities, states, and processes.
        3) Institutionalized phrases (collection agency) are treated as MWEs in STREUSLE but not UCCA.
        """


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
                                    describe_context(xs),  # context
                                    "|".join(sorted(t for x in xs for e in x.incoming if e.attrib.get("remote")
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
        VERBOSE=False
        return (evaluation.evaluate(converted_passage, reference_passage, errors=VERBOSE, units=VERBOSE, verbose=VERBOSE), reference_passage.ID,
                " ".join(t.text for t in sorted(reference_passage.layer(layer0.LAYER_ID).all, key=attrgetter(
                    "position"))), str(converted_passage), str(reference_passage))


REPORT_HEADERS = [
    "sent_id", "text", "deprel", "upos", "edeps", "expr_id", "expr_type", "lexcat", "ss", "ss2",
    "subtree", "ref_unit_id", "ref_tree_id", "ref_category", "ref_context", "ref_remote", "ref_unanalyzable",
    "ref_annotation", "pred_unit_id", "pred_tree_id", "pred_category", "pred_context", "pred_remote", "pred_unanalyzable",
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
    def children(self) -> List['Node']:
        return [e.dep for e in self.outgoing_basic]

    def children_with_rel(self, rel: str) -> List['Node']:
        return [e.dep for e in self.outgoing_basic if e.deprel==rel]

    def children_with_rels(self, rels: (str,)) -> List['Node']:
        return [e.dep for e in self.outgoing_basic if e.deprel in rels]


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
    def ss2(self) -> Optional[str]:
        for expr in self.exprs.values():
            ss = expr.get("ss2")
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

    @property
    def lexexpr(self) -> dict:
        if self.exprs.get("swes"):   # walrus
            assert "smwes" not in self.exprs
            return self.exprs.get("swes")
        if self.exprs.get("smwes"):
            return self.exprs.get("smwes")
        assert self.position==0,(self,self.exprs)   # must be root if no lexexpr

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
        if False and "Kim's" not in sent['text']:
            converted[sent[SENT_ID]] = None
            continue
        # if len(converted)>10:
        #     break

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
