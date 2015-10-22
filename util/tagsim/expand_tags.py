import sys
import os

from basic.constant import ROOT_PATH
from basic.common import writeRankingResults,checkToSkip,printStatus
from basic.annotationtable import readConcepts
from flickr_similarity import FlickrContextSim


INFO = __file__

def process(options, collection, annotationName):
    rootpath = options.rootpath
    overwrite = options.overwrite

    concepts = readConcepts(collection,annotationName,rootpath)
    resultdir = os.path.join(rootpath, collection, "SimilarityIndex", "ngd")

    todo = [x for x in concepts if not os.path.exists(os.path.join(resultdir,x+'.txt')) or overwrite]
    if not todo:
        printStatus(INFO, 'nothing to do')
        return

    fcs = FlickrContextSim(collection, rootpath=rootpath)
    vob = fcs.vob
    resultdir = os.path.join(rootpath, collection, "SimilarityIndex", "ngd")
    printStatus(INFO, 'expanding tags for %s-%s -> %s' % (collection, annotationName, resultdir))
    
    for concept in todo:
        resultfile = os.path.join(resultdir, concept + '.txt')
            
        vals = []
        for tag in vob:
            dist = fcs.computeNGD(concept, tag, img=1)
            if dist < 10:
                vals.append((tag,dist))
        vals.sort(key=lambda v:v[1])
        printStatus(INFO, '%s -> %s' % (concept, ' '.join([x[0] for x in vals[:3]])))
        writeRankingResults(vals, resultfile)
                
            
            
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection annotationName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    return process(options, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())
      
    
