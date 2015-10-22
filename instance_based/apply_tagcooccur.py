import sys, os

from basic.constant import ROOT_PATH
from basic.common import checkToSkip, niceNumber, makedirsforfile
from tagcooccur import *

def process(options, trainCollection, annotationName, testCollection):
    rootpath = options.rootpath
    m = options.m
    k_r = options.kr
    k_d = options.kd
    k_s = options.ks
    k_c = options.kc
    feature = options.feature
    add_bonus = options.bonus
    overwrite = options.overwrite
    
    #outputName = 'cotag,m%d,kr%d,kd%d,ks%d,kc%d,bonus%d'%(m,k_r,k_d,k_s,k_c,add_bonus)
    outputName = 'cotag' # simplify the outputName to reduce the length of the result filename
    outputName = os.path.join(outputName, feature) if (k_c>1e-6) else outputName
    resultfile = os.path.join(rootpath, testCollection, 'autotagging', testCollection, trainCollection, annotationName, outputName, 'id.tagvotes.txt')

    if checkToSkip(resultfile, overwrite):
        sys.exit(0)
     
    testImageSet = readImageSet(testCollection, testCollection, rootpath=rootpath)
    test_tag_reader = TagReader(testCollection, rootpath=rootpath)
    
    if k_c < 1e-6:
        tagger = TagCooccurTagger(testCollection, trainCollection, annotationName, rootpath=rootpath)
    else:
        tagger = TagCooccurPlusTagger(testCollection, trainCollection, annotationName, feature=feature, rootpath=rootpath)
    tagger.m = m
    tagger.k_r = k_r
    tagger.k_d = k_d
    tagger.k_s = k_s
    tagger.k_c = k_c
    tagger.add_bonus = add_bonus
    
    makedirsforfile(resultfile)
    
    fw = open(resultfile, 'w')
    
    output = []
    done = 0
    for im in testImageSet:
        user_tags = test_tag_reader.get(im)
        tagvotes = tagger.predict(content=im, context=user_tags)
        newline = '%s %s' % (im, ' '.join(['%s %s'%(x[0], niceNumber(x[1],6)) for x in tagvotes]))
        output.append(newline)
        done += 1
        if len(output) % 1e4 == 0:
            fw.write('\n'.join(output) + '\n')
            output=[]
            printStatus(INFO, '%d done' % done)
    if output:
        fw.write('\n'.join(output) + '\n')
    fw.close()
    printStatus(INFO, '%d done' % done)
    

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection annotationName testCollection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--bonus", default=DEFAULT_BONUS, type="int", help="overwrite existing file (default: %d)" % DEFAULT_BONUS)
    parser.add_option("--m", default=DEFAULT_M, type="int", help="m (default: %d)" % DEFAULT_M)
    parser.add_option("--kr", default=DEFAULT_KR, type="int", help="k_r for rank_promotion (default: %d)" % DEFAULT_KR)
    parser.add_option("--kd", default=DEFAULT_KD, type="int", help="k_d for descriptiveness (default: %d)" % DEFAULT_KD)
    parser.add_option("--ks", default=DEFAULT_KS, type="int", help="k_s for stability(default: %d)" % DEFAULT_KS)
    parser.add_option("--kc", default=DEFAULT_KC, type="int", help="k_c for tagrel(default: %d)" % DEFAULT_KC)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--feature", default=DEFAULT_FEAT, type="string", help="rootpath (default: %s)" % DEFAULT_FEAT)
   
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2])
   
if __name__ == "__main__":
    sys.exit(main())    

