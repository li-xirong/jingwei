import sys, os, random

from basic.constant import ROOT_PATH
from basic.common import checkToSkip, makedirsforfile, printStatus
from basic.annotationtable import readConcepts, writeConcepts, readAnnotationsFrom, writeAnnotations

OVERWRITE=0
NEG_POS_RATIO=1
POS_BAG_NUM=1
NEG_BAG_NUM=10

INFO = os.path.basename(__file__)

def process(options, collection, annotationName, pos_num):
    assert(annotationName.endswith('.txt'))
    rootpath = options.rootpath
    pos_bag_num = options.pos_bag_num
    neg_bag_num = options.neg_bag_num
    neg_pos_ratio = options.neg_pos_ratio

    annotationNameStr = annotationName[:-4] + ('.random%d' % pos_num) + '.%d' + ('.npr%d' % neg_pos_ratio) + '.%d.txt'

    concepts = readConcepts(collection, annotationName, rootpath=rootpath)
    
    skip = 0
    newAnnotationNames = [None] * (pos_bag_num * neg_bag_num)

    for idxp in range(pos_bag_num):
        for idxn in range(neg_bag_num):
            anno_idx = idxp * neg_bag_num + idxn
            newAnnotationNames[anno_idx] = annotationNameStr % (idxp, idxn)
            resultfile = os.path.join(rootpath,collection,'Annotations',newAnnotationNames[anno_idx])
            if checkToSkip(resultfile, options.overwrite):
                skip += 1
                continue
            writeConcepts(concepts,resultfile)

    first,second,last = annotationNameStr.split('%d')
    scriptfile = os.path.join(rootpath,collection,'annotationfiles',first + '0-%d'%(pos_bag_num-1) + second + '0-%d'%(neg_bag_num-1) + last)

    makedirsforfile(scriptfile)
    fout = open(scriptfile,'w')
    fout.write('\n'.join(newAnnotationNames) + '\n')
    fout.close()

    if len(newAnnotationNames) == skip:
        return 0
        
    for concept in concepts:
        names,labels = readAnnotationsFrom(collection, annotationName, concept, skip_0=True, rootpath=rootpath)
        positivePool = [x[0] for x in zip(names,labels) if x[1]>0]
        negativePool = [x[0] for x in zip(names,labels) if x[1]<0]
        
        for idxp in range(pos_bag_num):
            if len(positivePool) > pos_num:
                positiveBag = random.sample(positivePool, pos_num)
            else:
                positiveBag = positivePool
            for idxn in range(neg_bag_num):
                anno_idx = idxp * neg_bag_num + idxn
                newAnnotationName = newAnnotationNames[anno_idx]
                resultfile = os.path.join(rootpath,collection,'Annotations','Image',newAnnotationName,'%s.txt'%concept)
                if checkToSkip(resultfile, options.overwrite):
                    continue
                real_neg_num = max(len(positiveBag) * neg_pos_ratio, 1000)
                real_neg_num = min(len(negativePool), real_neg_num)
                negativeBag = random.sample(negativePool, real_neg_num)

                assert(len(set(positiveBag).intersection(set(negativeBag))) == 0)
                printStatus(INFO, "anno(%s,%d) %d pos %d neg -> %s" % (concept,anno_idx,len(positiveBag),len(negativeBag),resultfile))
                writeAnnotations(positiveBag + negativeBag, [1]*len(positiveBag) + [-1]*len(negativeBag), resultfile)

                
def main(argv=None):
    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection annotationName pos_num""")
    parser.add_option("--overwrite", default=OVERWRITE, type="int", help="overwrite existing file (default: %d)" % OVERWRITE)
    parser.add_option("--neg_pos_ratio", default=NEG_POS_RATIO, type="int", help="negative/positive ratio (default: %d)" % NEG_POS_RATIO)
    parser.add_option("--pos_bag_num", default=POS_BAG_NUM, type="int", help="number of positive bags (default: %d)" % POS_BAG_NUM)
    parser.add_option("--neg_bag_num", default=NEG_BAG_NUM, type="int", help="number of negative bags (default: %d)" % NEG_BAG_NUM)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath where the train and test collections are stored (default: %s)" % ROOT_PATH)
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], int(args[2]))
 

if __name__ == "__main__":
    sys.exit(main())
    
    
