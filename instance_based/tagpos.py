import sys
import os

from basic.constant import ROOT_PATH
from basic.common import checkToSkip,printStatus,makedirsforfile
from basic.util import readImageSet
from util.imagesearch.datareader import TagReader
    

DEFAULT_TPP = 'lemm'
INFO = __file__


def process(options, collection):
    rootpath = options.rootpath
    tpp = options.tpp
    overwrite = options.overwrite

    
    resultfile = os.path.join(rootpath, collection, "tagrel", collection, 'tagpos,%s'%tpp, 'id.tagvotes.txt')
    if checkToSkip(resultfile, overwrite):
        sys.exit(0)    

    imset = readImageSet(collection, collection, rootpath)
    printStatus(INFO, 'working on %d test images -> %s' % (len(imset),resultfile))
    
    reader = TagReader(collection,tpp=tpp,rootpath=rootpath)   
    
    makedirsforfile(resultfile)
    fw = open(resultfile, "w")
    output = []
    done = 0
    
    for im in imset:
        tags = reader.get(im)
        tagSet = set()
        tagSeq = []
        for tag in str.split(tags):
            if tag not in tagSet:
                tagSeq.append(tag)
                tagSet.add(tag)
        assert(len(tagSeq) == len(tagSet))
        
        nr_tags = len(tagSeq)
        tagvotes = [(tagSeq[i], 1.0-float(i)/nr_tags) for i in range(nr_tags)]
        newline = "%s %s" % (im, " ".join(["%s %g" % (x[0],x[1]) for x in tagvotes]))
        output.append(newline + "\n")
        done += 1
        
        if len(output)%1e4 == 0:
            printStatus(INFO, '%d %s %s' % (done,im,' '.join(['%s:%g' % (x[0],x[1]) for x in tagvotes[:3]] )))
            fw.write("".join(output))
            fw.flush()
            output = []
        
    if output:
        fw.write("".join(output))
    fw.close()
    printStatus(INFO, 'done')
    

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--tpp", default=DEFAULT_TPP, type="string", help="tag preprocess, can be raw, stem, or lemm (default: %s)" % DEFAULT_TPP)

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())

