import sys, os
import cPickle as pickle
import numpy as np

from basic.common import checkToSkip, makedirsforfile

def process(options, conceptfile, tagvotesfile, resultfile):
    if checkToSkip(resultfile, options.overwrite):
        return 0
    
    concepts = map(str.strip, open(conceptfile).readlines())
    concept2index = dict(zip(concepts,range(len(concepts))))
    
    data = open(tagvotesfile).readlines()
    print ('%d instances to dump' % len(data))
    
    concept_num = len(concepts)
    image_num = len(data)
    scores = np.zeros((image_num, concept_num)) - 1e4
    id_images = [None] * image_num
    
    for i in xrange(image_num):
        elems = str.split(data[i])
        id_images[i] = int(elems[0])
        del elems[0]
        for k in range(0, len(elems), 2):
            tag = elems[k]
            score = float(elems[k+1])
            j = concept2index.get(tag, -1)
            if j >= 0:
                scores[i,j] = score
    
    makedirsforfile(resultfile)
    output = open(resultfile, 'wb')
    pickle.dump({'concepts':concepts, 'id_images':id_images, 'scores':scores}, output, -1)
    output.close()
    
        
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] conceptfile tagvotesfile resultfile""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2])
   
if __name__ == "__main__":
    sys.exit(main())
