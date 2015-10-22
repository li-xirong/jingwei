import sys, os, random

from basic.constant import ROOT_PATH, DEFAULT_POS_NR
from basic.common import checkToSkip, readRankingResults, printStatus, makedirsforfile
from basic.annotationtable import readConcepts, writeConceptsTo, writeAnnotations, readAnnotationsFrom

INFO = __file__

if __name__ == '__main__':
    argv = sys.argv[1:]
    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection annotationName rankMethod posName""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--pos_nr", default=DEFAULT_POS_NR, type="int",  help="negative-positive ratio (default=DEFAULT_POS_NR)")
    parser.add_option("--neg_pos_ratio", default=1, type="int",  help="negative-positive ratio (default=1)")
    parser.add_option("--nr_neg_bags", default=1, type="int", help="nr of negative bags (default=1)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        sys.exit(0)

    rootpath = options.rootpath
    nr_pos = options.pos_nr
    collection = argv[0] #'train1m'
    annotationName = argv[1] # 'conceptsmir14social.txt'
    rankMethod = argv[2] #'train1m/fcs-wn_color64+dsift_borda'
    posName = argv[3] #'fcstagrelbc'
    neg_pos_ratio = options.neg_pos_ratio
    nr_neg = neg_pos_ratio * nr_pos
    nr_neg_bags = options.nr_neg_bags # 10
    overwrite = options.overwrite

    assert( annotationName.endswith('social.txt') )
    assert( rankMethod.startswith('tagged,lemm/%s'%collection) )

    newAnnotationTemplate = annotationName[:-4] + '.' + posName + str(nr_pos) + ('.random%d'%nr_neg) + '.%d.txt'
    concepts = readConcepts(collection, annotationName, rootpath)    
    simdir = os.path.join(rootpath, collection, 'SimilarityIndex', collection, rankMethod)

    scriptfile = os.path.join(rootpath,collection,'annotationfiles', annotationName[:-4] + '.' + posName + str(nr_pos) + ('.random%d'%nr_neg) + '.0-%d.txt'%(nr_neg_bags-1))
    makedirsforfile(scriptfile)
    fout = open(scriptfile,'w')
    fout.write('\n'.join([newAnnotationTemplate%t for t in range(nr_neg_bags)]) + '\n')
    fout.close()


    for concept in concepts:
        simfile = os.path.join(simdir, '%s.txt' % concept)
        ranklist = readRankingResults(simfile)
        pos_bag = [x[0] for x in ranklist[:nr_pos]]
        names, labels = readAnnotationsFrom(collection, annotationName, concept, skip_0=True, rootpath=rootpath)
        negativePool = [x[0] for x in zip(names,labels) if x[1] < 0]

        for t in range(nr_neg_bags):
            newAnnotationName = newAnnotationTemplate % t
            resultfile = os.path.join(rootpath, collection, 'Annotations', 'Image', newAnnotationName, '%s.txt'%concept)
            if checkToSkip(resultfile, overwrite):
                continue
            true_nr_neg = max(500, len(pos_bag)*neg_pos_ratio)
            neg_bag = random.sample(negativePool, true_nr_neg) #len(pos_bag)*neg_pos_ratio)
            assert(len(set(pos_bag).intersection(set(neg_bag))) == 0)
            printStatus(INFO, "anno(%s,%d) %d pos %d neg -> %s" % (concept,t,len(pos_bag),len(neg_bag),resultfile))
            writeAnnotations(pos_bag + neg_bag, [1]*len(pos_bag) + [-1]*len(neg_bag), resultfile)

    for t in range(nr_neg_bags):
        newAnnotationName = newAnnotationTemplate % t
        writeConceptsTo(concepts, collection, newAnnotationName)





