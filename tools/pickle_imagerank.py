import sys, os
import cPickle as pickle
import numpy as np

from basic.constant import ROOT_PATH
from basic.common import checkToSkip, makedirsforfile, readRankingResults
from basic.annotationtable import readConcepts
from basic.util import readImageSet


def process(options, collection, annotationName, simdir, resultfile):
    rootpath = options.rootpath
    
    if checkToSkip(resultfile, options.overwrite):
        return 0
    
    concepts = readConcepts(collection, annotationName, rootpath=rootpath)
    concept_num = len(concepts)

    id_images = readImageSet(collection, collection, rootpath)
    image_num = len(id_images)
    im2index = dict(zip(id_images, range(image_num)))
    print ('%d instances, %d concepts to dump -> %s' % (image_num, concept_num, resultfile))
    
    scores = np.zeros((image_num, concept_num)) - 1e4
    
    for c_id,concept in enumerate(concepts):
        simfile = os.path.join(simdir, '%s.txt' % concept)
        ranklist = readRankingResults(simfile)
        for im,score in ranklist:
            idx = im2index[im]
            scores[idx,c_id] = score

    makedirsforfile(resultfile)
    output = open(resultfile, 'wb')
    pickle.dump({'concepts':concepts, 'id_images':map(int,id_images), 'scores':scores}, output, -1)
    output.close()
    
        
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection annotationName simdir resultfile""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2], args[3])
   
if __name__ == "__main__":
    sys.exit(main())
