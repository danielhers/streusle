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

SENT_ID = "sent_id"
MWE_TYPES = ("swes", "smwes", "wmwes")

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

        #if sent[SENT_ID]=='reviews-245160-0004':
        #    assert False,nodes

        # TODO intermediate structure - annotate units with attributes such as "scene-evoking"

        # TODO Create primary UCCA tree
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

        # print dependency parse
        # if sent[SENT_ID]=='reviews-274498-0001':
        #     print_dep_parse(nodes)
        #     print(l1.root)
        #     assert False



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

            # decide main cat
            if expr['lexcat']=='PUNCT':
                mcat = 'U'
            elif lvc_scene_lus:
                mcat = '+'  # LVCs are scene-evoking
            else:
                mcat = 'LU'

            outerlu = toplu = lu = l1.add_fnode(dummyroot, mcat)
            n = expr['nodes'][0]
            unit2expr[lu] = n.swe or n.smwe # could use walrus operator here

            isUNA = False
            if len(expr['nodes'])>1 and not lvc_scene_lus:
                # non-LVC MWE: treat as unanalyzable
                # TODO: dates should also be unanalyzable, but alas, they're not MWEs
                isUNA = True
                toplu = lu = l1.add_fnode(toplu, 'UNA')

            for node in expr['nodes']:
                if lvc_scene_lus and node not in lvc_scene_lus:
                    # light verb or determiner in LVC
                    lu = l1.add_fnode(toplu, 'F')
                else:
                    lu = toplu
                lu.add(layer1.EdgeTags.Terminal, node.terminal)

                unit2expr[lu] = node.swe or node.smwe
                node2unit[node] = outerlu if isUNA else lu
                if lvc_scene_lus and node not in lvc_scene_lus and node2unit[node] is lu:
                    # point to parent unit because things should never attach under F units
                    node2unit[node] = toplu

                # delete within-MWE dep edges so we don't process them later
                h = node.head   # could use walrus operator
                if h in expr['nodes']:
                    e = node.incoming_basic[0]
                    node.incoming_basic.remove(e)
                    h.outgoing_basic.remove(e)

        # TODO: treat all 'discourse' subtrees as G with no internal structure, like an MWE?

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
                    elif n.ss in ("n.ACT", "n.EVENT", "n.PHENOMENON", "n.PROCESS"):
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
                elif 'heuristic_relation' in expr and expr['heuristic_relation']['config'].startswith('predicative') and n.lexlemma!='nothing but' and not (n.ss2=='p.Extent' and n.lexlemma=='as'):
                    # a predicative SNACS relation evoked by u
                    # skip nothing_but, which is treated semantically as P but syntactically as coordination, which causes problems
                    h = n.head  # TODO: in an idiomatic PP this will not be the head noun, it will be ITS head (because we modified the dep parse for MWEs)
                    if n.deprel=='case' or not (h and h.children_with_rel('case')):    # exclude advmod if sister to case, like "*back* between"
                        u._fedge().tag = 'S' # predicative PP, so preposition is scene-evoking
                        if num_dep_parents(expr)>1:
                            pass    # e.g. "in the midst of"
                        elif n.deprel!='root' and not n.deprel.startswith('root:') and n.lexcat!='PP' and expr['heuristic_relation']['obj'] is not None:
                            assert h.lexcat in ('N','PROPN','PRON','NUM','ADJ'),(h,h.lexcat,str(u),str(l1.root))
                            # adverbial modifiers of the NOMINAL this P case-marks should be semantically under the SNACS-evoked unit
                            node2unit_for_advbl_attachment[h] = u # make preposition unit the attachment site for ADVERBIAL dependents
                elif n.deprel=='cop' and n.head.lexcat in ('N','PRON','PROPN','NUM') and not n.head.children_with_rel('case'):    # predicate nominal
                    # copula is head
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


        # fix dep tree to address problem case:
        # conj of two compound modifiers, the 2nd of which forms an MWE with its head
        for n in nodes:
            if n.deprel=='conj' and n.head and n.head.head and n.lexexpr is not n.head.lexexpr and n.lexexpr is n.head.head.lexexpr:
                print('Removing nasty conj-modifier-in-MWE',sent[SENT_ID],n,n.head,n.head.head)
                # x --a-> y --conj-> n, where x & n form an MWE that does not include y
                # transform to
                # n (<-a--) x --conj-> y
                # where the <-a-- edge is omitted because it would be MWE-internal
                y = n.head
                x = n.head.head

                e_other = y.incoming_basic[0]
                y.incoming_basic.remove(e_other)
                x.outgoing_basic.remove(e_other)

                e_conj = n.incoming_basic[0]
                n.incoming_basic.remove(e_conj)
                y.outgoing_basic.remove(e_conj)
                e_conj.head = x
                e_conj.dep = y
                x.outgoing_basic.append(e_conj)
                y.incoming_basic.append(e_conj)

                # In our convention the cc attaches to the conj head.
                # Initially that was y, so we move it to x.
                for cc in y.children_with_rel('cc'):
                    e_cc = cc.incoming_basic[0]
                    y.outgoing_basic.remove(e_cc)
                    e_cc.head = x
                    x.outgoing_basic.append(e_cc)

                print_dep_parse(nodes)



        #if 'surgery' in sent['text']:
        #    print(l1.root)
        # if sent[SENT_ID]=='reviews-274498-0001':
        #     print_dep_parse(nodes)
        #     print(l1.root)
        #     assert False

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
            elif r=='discourse' and cat=='-':
                #assert cat=='-',(cat,n,r,h)
                # keep at root
                u._fedge().tag = 'G'


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
            if h is None:
                continue
            r = n.deprel
            hu = parent_unit_for_dep(h, r)
            assert hu is not u,(n,r,h,str(u),str(l1.root))
            if r in ('amod','advmod','acl') or r.startswith(('amod:','advmod:','acl:')):
                if cat=='-':
                    hucat = hu.ftag
                    dummyroot.remove(u)
                    hu.add('D' if hucat in ('+','S') else 'E', u)
                elif cat=='S' or cat=='+':
                    # adjectives are treated as S thus far
                    # cat=='+' for e.g. a *personalized* gift
                    hucat = hu.ftag
                    dummyroot.remove(u)

                    if hucat=='S' and (r=='amod' or r.startswith('amod:')):  # probably an ADJ compound, like African - American
                        hu.add('C', u)  # put "African" as C under "American"---that will later become C too
                    elif hucat in ('-','A','E'):  # E-scene. make node for scene to attach as E
                        scn = l1.add_fnode(hu, 'E')
                        scn.add('S', u)
                    elif hucat in ('+','S'):
                        # if hucat!='+':
                        #     print('Decide whether to allow advmod under S',hucat,str(hu))
                        #     print(cat,str(u),str(l1.root))
                        hu.add('D', u)  # e.g. aspectual verb particle
                    else:
                        # maybe not a problem--allow the E strategy under all non-scene units?
                        # TODO: hucat=='S' + advmod
                        # TODO: 'when' advcl is apparently an annotation error
                        # '10 more minutes'
                        #assert h.lexlemma=='when',(hucat,str(hu),cat,str(u),n,n.deprel,n.head,str(l1.root))
                        pass
            elif r=='nummod':
                hucat = hu.ftag
                dummyroot.remove(u)
                hu.add('Q' if hucat=='-' else 'D', u)
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
                        dummyroot.remove(obju)
                        u.add(objcat, obju)
                        #assert False,(objcat,str(obju),str(l1.root))
                    # TODO: T and D PPs (The meeting was on Thursday)
                elif obji is not None:
                    obj = nodes[obji]
                    obju = node2unit[obj]
                    dummyroot.remove(u)
                    if n.lexcat=='POSS' and n.ss in ('n.Possessor',):
                        # ownership -> possessive particle is scene-evoking
                        gov = nodes[govi]
                        govu = node2unit[gov]
                        govu.add('S', u)
                        u.add('A', obju)
                        # there would also be a remote edge from the possession scene to the gov
                    #elif n.ss=='p.Extent':
                    #    assert False,(str(n),str(l1.root))
                    else:   # make prep/possessive a relator under its object
                        obju.add('R', u)
                elif config=='possessive':
                    # possessive pronoun
                    # TODO: scene-evoking head noun
                    # TODO: ownership (S+A)

                    # default to E
                    dummyroot.remove(u)
                    gov = nodes[govi]
                    govu = node2unit[gov]
                    govu.add('E', u)
                else:
                    assert False,expr
            elif r in ('nmod','compound') or r.startswith(('nmod:','compound:')):    # misc nmod--see rules for obl
                # TODO: scene nmod
                hucat = hu.ftag
                if cat not in ('+','S','P'):
                    assert cat=='-',(cat,str(u))
                    if n.ss=='n.TIME' or r=='nmod:tmod' or any(c.ss in SNACS_TEMPORAL for c in n.children):
                        newcat = 'T' if hucat in ('+','S','P') else 'E'
                        # TODO: "books *from the 1900s*" would be E?
                    elif n.ss in ('p.Approximator','p.Manner','p.Extent') or any(c.ss in ('p.Manner','p.Extent') for c in n.children):
                        newcat = 'D' if hucat in ('+','S','P') else 'E'
                    else:
                        newcat = 'A' if hucat in ('+','S','P') else 'E'
                    dummyroot.remove(u)
                    hu.add(newcat, u)
            elif r=='appos' and cat not in ('+','S','P'):
                if n.lexcat=='PROPN':
                    # TODO: make the head an E
                    assert False,('PROPN as C in apposition:',cat,n.lexlemma,h.lexlemma)
                else:
                    # make this an E
                    assert cat=='-',(cat,n,str(u),hucat,h,str(hu))
                    dummyroot.remove(u)
                    hu.add('E', u)

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
            h = n.head
            if h is None:
                continue
            r = n.deprel
            hu = parent_unit_for_dep(h, r)
            hucat = hu.ftag if hu else None
            if cat=='-' and (r in ('nsubj','obj','iobj') or r.startswith(('nsubj:','obj:','iobj:'))):

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
                # xcomp would involve a remote
                participantu = l1.add_fnode(hu, 'A')
                dummyroot.remove(u)
                participantu.add(cat, u)
            elif r=='expl' or r.startswith('expl:'):
                assert cat=='-'
                assert expr["lexlemma"] in ('it','there'),(expr,sent['text'])
                if expr["lexlemma"]=='it':
                    # expletive 'it' -> F
                    dummyroot.remove(u)
                    hu.add('F', u)
                elif expr["lexlemma"]=='there':
                    # existential 'there' -> S
                    dummyroot.remove(u)
                    hu.add('S', u)
            elif r=='mark' and n.lexlemma in ('to','that') and n.ss != "p.Purpose":
                if hucat=='-':
                    print('Weird attachment of inf/complementizer to non-scene unit',expr,str(hu),str(l1.root))
                dummyroot.remove(u)
                hu.add('F' if n.lexlemma=='to' else 'R', u) # infinitive TO = F, complementizer THAT = R
            elif r=='obl' or r.startswith('obl:'):
                # TODO: scene obl

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
                    hu.add(newcat, u)


        # coordination and connectives

        for u in dummyroot.children:
            cat = u.ftag
            #if cat not in ('+', '-', 'S', 'LVC'):
            #    continue
            assert u in unit2expr,(str(l1.root),str(u))
            expr = unit2expr[u]
            n = expr_head(expr)
            h = n.head
            if h is None:
                continue
            r = n.deprel
            hu = parent_unit_for_dep(h, r)
            if hu is None:
                continue
            hucat = hu.ftag
            if r=='conj':
                extrau = l1.add_fnode(hu, 'CONJ')
                dummyroot.remove(u)
                extrau.add(cat, u)   # later: decide whether this is connected by N or L, and raise as sibling of first conjunct
            elif cat=='-' and (n.lexcat == "DISC" or n.ss in ('p.Purpose','p.Explanation','p.ComparisonRef') or r=='cc' or r.startswith('cc:') or (r=='mark' and n.lexlemma not in ('to','that','which'))):
                dummyroot.remove(u)
                if r=='cc' and hucat not in ('+','P','S'):
                    # Decide N vs. L based on the first conjunct's scene status
                    # N.B. Depedit moved cc to be under the first conjunct
                    # Coordination of non-scene unit evokers: coordinators as N and the conjunct heads as C
                    hu.add('N', u)
                else: # linker
                    if hu is None:
                        newu = l1.add_fnode(None, 'L')
                        newu.add(u)
                    else:
                        # add as sister to dependency head
                        hu.fparent.add('L', u)


        # decide S or P for remaining "+" scenes

        for u in l1.all:
            if u.ftag=='+':
                expr = unit2expr[u]
                n = expr_head(expr)
                if n.ss=='v.stative' and n.lexlemma in ('be','have'):
                    u._fedge().tag = 'S'
                elif n.ss.startswith('v.'):
                    u._fedge().tag = 'P'
                elif n.ss in ("n.ACT", "n.EVENT", "n.PHENOMENON", "n.PROCESS"):
                    u._fedge().tag = 'P'
                elif n.lexcat=='AUX':   # e.g. due to ellipsis
                    u._fedge().tag = 'P'
                else:
                    assert False,(u,u.ftag,n,expr)

        #assert not lvc_scene_lus,l1.root

        # any P or S unit that is under the root: wrap in H
        for u in dummyroot.children:
            ucat = u.ftag
            if ucat in ('P','S'):
                dummyroot.remove(u)
                pu = l1.add_fnode(dummyroot, 'H')
                pu.add(ucat, u)


        # TODO: raise lexical units (terminals/UNA) without unary parents
        # to be sister and acquire the parent's category
        #    [A [P [R that] [A I] [F had] [F a] [D routine] surgery]]
        # -> [A [R that] [P [A I] [F had] [F a] [D routine] [P surgery]]]
        #
        #    [P [A I] [F had] [F a] [D routine] surgery]
        # -> [H [A I] [F had] [F a] [D routine] [P surgery]]

        # ensure all lexical units have a unary parent unit

        # apparently terminal nodes cannot remove from Foundational Layer units once added,
        # so this will be a bit convoluted to accomodate non-UNA lexical expressions

        # BEFORE
        #    [P [A I] [F had] [F a] [D routine] surgery] [- [R for] [F an] [UNA ingrown toenail] ] [U .] ]
        #    ^pu                                         ^u ^cu
        # insert new wrapper node as "-", then [R for] and [F an] move up under the wrapper node,
        # then move lexical node under the wrapper node as "C"
        # AFTER
        #    [P [A I] [F had] [F a] [D routine] surgery] [- [R for] [F an] [C [UNA ingrown toenail] ] ] [U .] ]
        #                                                ^newu = pu


        # BEFORE
        # [DUMMYROOT ...   [P [A I] [D actually] forgot [A [P [R to] feed [A [E my] cat] ] ]  ...]
        # ^pu              ^u

        for u in l1.all:
            if len(u.children)>1 and any(cu.incoming[0].tag in (layer1.EdgeTags.Terminal, 'UNA') for cu in u.children):
                pu = u.fparent
                assert pu,(str(u),str(l1.root))
                if len(pu.children)>1:
                    # we need to create a new node to wrap just the current contents of u
                    ucat = u.ftag
                    newu = l1.add_fnode(pu, ucat)
                    pu.remove(u)
                    newu.add('C', u)
                    pu = newu
                for cu in u.children:
                    if cu.incoming[0].tag not in (layer1.EdgeTags.Terminal, 'UNA'):
                        # move up
                        cucat = cu.ftag
                        u.remove(cu)
                        pu.add(cucat, cu)
                # now u will consist only of terminals


        def dfs_order_subtree(unit):
            items = [unit]
            for u in unit.children:
                items.extend(dfs_order_subtree(u))
            return items

        # traversing top down, move CONJ up and decide H or C based on L/N sister

        # [- [R to] [E his] [C friends] [N [UNA rather than] ] [CONJ [E his] [- customers] ] ]
        # CONJ is non-unary -> change CONJ to C

        for u in dfs_order_subtree(dummyroot):
            if isinstance(u,layer1.FoundationalNode) and u.ftag=='CONJ':
                pu = u.fparent
                pucat = pu.ftag
                if len(u.children)>1 and pucat not in ('+','P','S','H'):   # need to keep this unit span but change CONJ to H/C
                    if any(sib.ftag in ('H','L') for sib in pu.children):
                        assert not any(sib.ftag=='N' for sib in pu.children),(str(gpu),sent['text'])
                        newcat = 'H'    # scene linkage
                    else:
                        newcat = 'C'    # conjoined non-scenes
                    u._fedge().tag = newcat
                else:
                    gpu = pu.fparent
                    if pucat in ('+','P','S','H') or any(sib.ftag in ('H','L') for sib in gpu.children):
                        assert not any(sib.ftag=='N' for sib in gpu.children),(str(gpu),sent['text'])
                        newcat = 'H'    # scene linkage
                    else:
                        newcat = 'C'    # conjoined non-scenes
                    pu.remove(u)
                    gpu.add(newcat, u)
                # TODO: mixed coordination, e.g. money "for antibiotics and a visit to the vet"
                # (should we add an implicit scene unit for the non-scene conjunct?)

        KEYWORDS = ('mircles','ufc fighter','kitchen and wait staff','For example','would have been distorted','surgery','eather','feed my cat','of the ants','patrons willing','were back quickly','worst Verizon store','as proper as')
        if any(kwd in sent['text'] for kwd in KEYWORDS):
            print(l1.root)



        """
        # move node to another parent with same category
        print(dummyroot)
        u = dummyroot.children[0]
        cat = u.incoming[0].tag
        print('child', u, cat)
        #assert False,repr(u)
        dummyroot.remove(u)
        print(l1.all[0])
        dummyroot.children[1].add(cat, u)
        assert False,l1.all[0]
        """



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

        - 'Huge ammount of time wasted time': first 'time' is case

        - when + clause: when often advmod when it should be mark

        - preposition stranding not consistent: ADP <case@L _ vs. ADP <obl _
            govobj.py workaround
        - ADV <advmod (NOUN|PRON !>cop _)   -- advmod attachments to nouns. many look weird

        - typos: !L=there&!L=it <expl _

        FIXED
        - amod(PRON,ADJ) should be xcomp(VERB,ADJ): "saw it *riddled* with..."
        - typo expl THERE
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
