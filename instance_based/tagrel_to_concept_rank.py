
import sys
import numpy as np
from basic.common import checkToSkip, printStatus, makedirsforfile

INFO = __file__


def process(options, inputfile, resultfile):
    assert(inputfile.endswith('.pkl'))
    #resultfile = inputfile[:-4] + '_rank.pkl'
    
    if checkToSkip(resultfile, options.overwrite):
        return 0
    
    import cPickle as pickle
    data = pickle.load(open(inputfile,'rb'))
    scores = data['scores']
    id_images = data['id_images']
    concepts = data['concepts']
    nr_of_images = len(id_images)
    nr_of_concepts = len(concepts)
    
    assert(scores.shape[0] == nr_of_images)
    assert(scores.shape[1] == nr_of_concepts)
 
    DEFAULT_RANK = nr_of_concepts 
    rank_matrix = np.zeros((nr_of_images, nr_of_concepts), dtype=np.int) + DEFAULT_RANK
    
    for i in xrange(nr_of_images):
        sorted_index = np.argsort(scores[i,:]) # in ascending order
        for j in range(nr_of_concepts):
            c_idx = sorted_index[j]
            rank = nr_of_concepts - j
            rank_matrix[i, c_idx] = rank
    
        if (i+1) % 1e5 == 0:
            printStatus(INFO, '%d done' % (i+1) )
    
    makedirsforfile(resultfile)
    output = open(resultfile, 'wb')
    pickle.dump({'id_images':id_images, 'concepts':concepts, 'rank_matrix':rank_matrix}, output, -1)
    output.close()
    printStatus(INFO, '%dx%d dumped to %s' % (nr_of_images, nr_of_concepts, resultfile))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] inputfile resultfile""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])
    
if __name__ == "__main__":
    sys.exit(main())    

