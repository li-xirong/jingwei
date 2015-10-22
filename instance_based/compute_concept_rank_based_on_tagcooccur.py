import sys, os
import numpy as np

from basic.constant import ROOT_PATH
from basic.common import checkToSkip, makedirsforfile, printStatus
from basic.annotationtable import readConcepts
from tagdb import TagCooccurBase

INFO = __file__


def process(options, trainCollection, annotationName):
    rootpath = options.rootpath
    overwrite = options.overwrite
    
    resultfile = os.path.join(rootpath, trainCollection, 'TextData', 'tag.concept-rank.%s.pkl' % annotationName)
    if checkToSkip(resultfile, overwrite):
        return 0
        
    concepts = readConcepts(trainCollection, annotationName, rootpath)
    concept_num = len(concepts)
    concept2index = dict(zip(concepts, range(concept_num)))
    tcb = TagCooccurBase(trainCollection, rootpath=rootpath)
    tag_num = tcb.tag_num()
    DEFAULT_RANK = tag_num
    rank_matrix = np.zeros((tag_num, concept_num), dtype=np.int) + DEFAULT_RANK
    tag_list = []
    
    for i,u in enumerate(tcb.vob):
        ranklist = tcb.top_cooccur(u,-1)
        concept2rank = {}
        rank = [DEFAULT_RANK] * concept_num
        
        hit = 0
        for j,x in enumerate(ranklist):
            idx = concept2index.get(x[0], -1)
            if idx>=0:
                rank_matrix[i,idx] = j+1
                hit += 1
                if hit == concept_num:
                    break
        tag_list.append(u)
        
        if (i+1) % 1e4 == 0:
            printStatus(INFO, '%d done' % (i+1) )
    
    assert(len(tag_list) == tag_num)
    
    import cPickle as pickle
    makedirsforfile(resultfile)
    output = open(resultfile, 'wb')
    pickle.dump({'tags':tag_list, 'concepts':concepts, 'rank_matrix':rank_matrix}, output, -1)
    output.close()
    printStatus(INFO, '%dx%d dumped to %s' % (tag_num, concept_num, resultfile))

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection annotationName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])
    
if __name__ == "__main__":
    sys.exit(main())    

    
