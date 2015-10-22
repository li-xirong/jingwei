import sys,os
import numpy as np
from basic.constant import ROOT_PATH
from basic.common import makedirsforfile,checkToSkip,niceNumber,printStatus, printError
from basic.annotationtable import readConcepts

INFO = __file__


def process(options, testCollection, trainCollection, annotationName, tagrelMethod, tagfeature):
    rootpath = options.rootpath
    overwrite = options.overwrite

    concepts = readConcepts(trainCollection, annotationName, rootpath)
    nr_of_concepts = len(concepts)
    mapping = dict(zip(concepts,range(nr_of_concepts)))
    
    feat_dir = os.path.join(rootpath, testCollection, 'FeatureData', tagfeature)
    binary_file = os.path.join(feat_dir, 'feature.bin')
    id_file = os.path.join(feat_dir, 'id.txt')
    shape_file = os.path.join(feat_dir,'shape.txt')

    if checkToSkip(binary_file, overwrite):
        sys.exit(0)

    inputfile = os.path.join(rootpath, testCollection, 'autotagging', testCollection, trainCollection, tagrelMethod, 'id.tagvotes.txt')
    if not os.path.exists(inputfile):
        printError(INFO, '%s does not exist' % inputfile)
        sys.exit(0)

    makedirsforfile(binary_file)
    fw = open(binary_file, 'wb')
    processed = set()
    imset = []
    count_line = 0

    for line in open(inputfile):
        count_line += 1
        elems = str.split(line.strip())
        name = elems[0]

        if name in processed:
            continue
        processed.add(name)

        del elems[0]
        assert(len(elems) == 2 * nr_of_concepts)
        vec = [0] * nr_of_concepts

        for i in range(0, len(elems), 2):
            tag = elems[i]
            idx = mapping[tag]
            score = float(elems[i+1])
            vec[idx] = score

        s = float(sum(vec)) # l_1 normalized
        vec = np.array([x/s for x in vec], dtype=np.float32)
        vec.tofile(fw)
        imset.append(name)

    fw.close()

    fw = open(id_file, 'w')
    fw.write(' '.join(imset))
    fw.close()

    fw = open(shape_file, 'w')
    fw.write('%d %d' % (len(imset), nr_of_concepts))
    fw.close()
    print ('%d lines parsed, %d ids ->  %d unique ids' % (count_line, len(processed), len(imset)))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection trainCollection annotationName tagrelMethod tagfeature""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 5:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2], args[3], args[4])


if __name__ == "__main__":
    sys.exit(main())

