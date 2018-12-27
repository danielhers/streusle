#!/usr/bin/env python3
"""
Given a .conllulex file, convert each sentence to a UCCA Passage object.
If the script is called directly, outputs the data as XML, Pickle or JSON files.

@since: 2018-12-27
@author: Daniel Hershcovich
@requires: ucca=1.0.129 semstr==1.1 tqdm
"""

import argparse
import os

from semstr.convert import iter_files, write_passage
from tqdm import tqdm
from ucca.core import Passage

from conllulex2json import load_sents

STREUSLE_SENT_ID = "streusle_sent_id"


def convert(sent: dict) -> Passage:
    return Passage(ID=sent[STREUSLE_SENT_ID])


class ConcatenatedFiles:
    def __init__(self, filenames):
        self.lines = []
        self.name = None
        for filename in iter_files(filenames):
            with open(filename, encoding="utf-8") as f:
                self.lines += list(f)
            self.name = filename

    def __iter__(self):
        return iter(self.lines)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    sents = list(load_sents(ConcatenatedFiles(args.filenames)))
    t = tqdm(sents, unit=" files", desc="Converting")
    for sent in t:
        t.set_postfix({STREUSLE_SENT_ID: sent[STREUSLE_SENT_ID]})
        passage = convert(sent)
        if args.write:
            write_passage(passage, out_dir=args.out_dir, output_format=args.format, binary=args.format == "pickle",
                          verbose=args.verbose)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Convert .conllulex files to UCCA")
    argparser.add_argument("filenames", nargs="+", help=".conllulex file name(s) to convert")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
    argparser.add_argument("-f", "--format", choices=("xml", "pickle", "json"), default="xml", help="output format")
    argparser.add_argument("-v", "--verbose", action="store_true", help="extra information")
    argparser.add_argument("-n", "--no-write", action="store_false", dest="write", help="do not write files")
    main(argparser.parse_args())
