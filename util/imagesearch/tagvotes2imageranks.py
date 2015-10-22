import sys, os
from optparse import OptionParser


from basic.constant import ROOT_PATH
from basic.common import writeRankingResults,checkToSkip
from basic.annotationtable import readConcepts
from basic.util import readLabeledImageSet


def generate_result_dir(options, testCollection, tagvotefile):
    rootpath = options.rootpath
    items = tagvotefile.split('/')
    start = len(items) - items[::-1].index(testCollection)
    modelName = '/'.join(items[start:-1])
    if options.tagged:
        resultdir = os.path.join(rootpath, testCollection, 'SimilarityIndex', testCollection, 'tagged,%s' % options.tpp, modelName)
    else:
        resultdir = os.path.join(rootpath, testCollection, 'SimilarityIndex', testCollection, modelName)
    return resultdir




def process(options, testCollection, annotationName, tagvotefile):
    rootpath = options.rootpath
    tpp = options.tpp
    tagged = options.tagged
    overwrite = options.overwrite

    resultdir = generate_result_dir(options, testCollection, tagvotefile)
    
    concepts = readConcepts(testCollection, annotationName, rootpath)
    todo = []
    for concept in concepts:
        resfile = os.path.join(resultdir, '%s.txt'%concept)
        if checkToSkip(resfile, overwrite):
            continue
        todo.append(concept)

    if not todo:
        print ('nothing to do')
        return 0

    nr_of_concepts = len(todo)
    labeled_set = [None] * nr_of_concepts
    if tagged:
        for i in range(nr_of_concepts):
            labeled_set[i] = set(readLabeledImageSet(testCollection, todo[i], tpp, rootpath))
        
    concept2index = dict(zip(todo, range(nr_of_concepts)))
    ranklists = [[] for i in range(nr_of_concepts)]

    for line in open(tagvotefile):
        elems = line.strip().split()
        imageid = elems[0]
        del elems[0]
        assert(len(elems)%2==0)

        for i in range(0, len(elems), 2):
            tag = elems[i]
            c = concept2index.get(tag, -1)
            if c >= 0:
                if tagged and imageid not in labeled_set[c]:
                    continue
                score = float(elems[i+1])
                ranklists[c].append((imageid,score))

    for i in range(nr_of_concepts):
        concept = todo[i]
        resfile = os.path.join(resultdir, '%s.txt'%concept)
        ranklist = sorted(ranklists[i], key=lambda v:(v[1], v[0]), reverse=True)
        print ('%s %d -> %s' % (concept, len(ranklist), resfile))
        writeRankingResults(ranklist, resfile)



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = OptionParser(usage="""usage: %prog [options] testCollection annotationName tagvotefile""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--tpp", default="lemm", type="string", help="tag preprocess, can be raw, stem, or lemm (default: lemm)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--tagged", default=0, type="int", help="consider only tagged images (default: 0)")
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2])

if __name__ == "__main__":
    sys.exit(main())


            
