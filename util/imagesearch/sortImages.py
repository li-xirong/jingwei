import sys
import os

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile, printStatus
from searchengine import getSearchEngine, submit

DEFAULT_TPP = 'lemm'

def process(options, collection, annotationName, engine, engineparams):
    rootpath = options.rootpath
    overwrite = options.overwrite
    tpp = options.tpp

    engineclass = getSearchEngine(engine)
    
    if engine == 'detector':
        searcher = engineclass(collection, collection, engineparams, rootpath)
    elif engine == 'rawtagnum':
        searcher = engineclass(collection, collection, tpp=tpp, rootpath=rootpath)
    else:
        searcher = engineclass(collection, collection, engineparams, tpp=tpp, rootpath=rootpath)
  
    submit([searcher], collection, annotationName, rootpath=rootpath, overwrite=overwrite)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection annotationName engine engineparams""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--tpp", default=DEFAULT_TPP, type="string", help="tag preprocess, can be raw, stem, or lemm (default: %s)" % DEFAULT_TPP)
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    sys.exit(main())  
