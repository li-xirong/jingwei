import sys
import os

from basic.constant import ROOT_PATH, DEFAULT_NEG_FILTER
from basic.common import checkToSkip,printStatus,makedirsforfile
from basic.util import readLabeledImageSet
from basic.annotationtable import readConcepts, writeConceptsTo, writeAnnotations
from negativeengine import STRING_TO_NEGATIVE_ENGINE 

INFO = __file__


def process(options, collection, annotationName):
    rootpath = options.rootpath
    overwrite = options.overwrite
    neg_filter = options.neg_filter
    
    concepts = readConcepts(collection, annotationName, rootpath)
    newAnnotationName = annotationName[:-4] + 'social.txt'
    ne = STRING_TO_NEGATIVE_ENGINE[neg_filter](collection, rootpath)

    newConcepts = []
    for concept in concepts:
        resultfile = os.path.join(rootpath, collection, 'Annotations', 'Image', newAnnotationName, '%s.txt'%concept)
        if checkToSkip(resultfile, overwrite):
            newConcepts.append(concept)
            continue

        try:
            pos_set = readLabeledImageSet(collection, concept, tpp='lemm', rootpath=rootpath)
        except:
            pos_set = None 
        if not pos_set:
            printStatus(INFO, '*** %s has not labeled examples, will be ignored ***' % concept)
            continue
        neg_set = ne.sample(concept, int(1e8))
        assert(len(set(pos_set).intersection(set(neg_set))) == 0)
        newlabels = [1] * len(pos_set) + [-1] * len(neg_set)
        newnames = pos_set + neg_set
        printStatus(INFO, "anno(%s) %d pos %d neg -> %s" % (concept,len(pos_set),len(neg_set),resultfile))
        writeAnnotations(newnames, newlabels, resultfile)
        newConcepts.append(concept)

    writeConceptsTo(newConcepts, collection, newAnnotationName, rootpath)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection annotationName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--neg_filter", default=DEFAULT_NEG_FILTER, type="string", help="filter for removing false negatives (default: %s)" % DEFAULT_NEG_FILTER)

    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    assert(options.neg_filter in str.split('wn co'))
    return process(options, args[0], args[1])

if __name__ == "__main__":
    sys.exit(main())

