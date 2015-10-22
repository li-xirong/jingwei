import sys
import os
import numpy as np

from basic.constant import ROOT_PATH
from basic.common import readRankingResults,writeRankingResults,checkToSkip
from basic.annotationtable import readConcepts
from basic.util import readImageScoreTable


              
def process(options, collection, annotationName, runfile, newRunName):
    rootpath = options.rootpath
    overwrite = options.overwrite

    dataset = options.testset if options.testset else collection
    
    concepts = readConcepts(collection, annotationName, rootpath)
    simdir = os.path.join(rootpath, collection, "SimilarityIndex", dataset)

    data = [x.strip() for x in open(runfile).readlines() if x.strip() and not x.strip().startswith("#")]
    models = []
    for line in data:
        weight,run = str.split(line)
        models.append((run, float(weight), 1))
    
    for concept in concepts:
        resultfile = os.path.join(simdir, newRunName, concept + ".txt")
        if checkToSkip(resultfile, overwrite):
            continue

        scorefile = os.path.join(simdir, models[0][0], concept + ".txt")
        if not os.path.exists(scorefile):
            print ("%s does not exist. skip" % scorefile)
            continue

        ranklist = readRankingResults(scorefile)
        names = sorted([x[0] for x in ranklist])

        nr_of_images = len(names)
        name2index = dict(zip(names, range(nr_of_images)))
   
        print ('%s %d' % (concept, nr_of_images))
        
        scoreTable = readImageScoreTable(concept, name2index, simdir, models, torank=options.torank)
        assert(scoreTable.shape[1] == nr_of_images)

        weights = [model[1] for model in models]

        scores = np.matrix(weights) * scoreTable
        scores = [float(scores[0,k]) for k in range(nr_of_images)]
  
        newranklist = [(names[i], scores[i]) for i in range(nr_of_images)]
        newranklist.sort(key=lambda v:(v[1],v[0]), reverse=True)
     
        writeRankingResults(newranklist, resultfile)
               

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection annotationName runfile newRunName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--testset", default=None, type="string", help="subset to process. Default is None")
    parser.add_option("--torank", default=1, type=int, help="rank-based score normalization (default: 1)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    sys.exit(main())
 
