import sys, os
from optparse import OptionParser

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile, checkToSkip, printStatus

INFO = __file__


def buildHitlists(collection,concepts,tpp,rootpath=ROOT_PATH):
    conceptSet = []
    for concept in concepts:
        conceptSet += concept.split('-')
    conceptSet = set(conceptSet)

    tagfile = os.path.join(rootpath,collection,'TextData', 'id.userid.%stags.txt'%tpp)
    hitlists = dict([(x,[]) for x in conceptSet])
    
    for line in open(tagfile):
        elems = str.split(line.strip())
        photoid = elems[0]
        tagset = set([w.lower() for w in elems[2:]])
        tagset = tagset.intersection(conceptSet)
        
        for w in tagset:
            hitlists[w].append(photoid)
            
    return hitlists


def process(options, collection, conceptfile):
    rootpath = options.rootpath
    tpp = options.tpp
    overwrite = options.overwrite

    concepts = [x.strip() for x in open(conceptfile).readlines() if x.strip() and not x.strip().startswith('#')]
    resultdir = os.path.join(rootpath, collection, 'tagged,%s'%tpp)

    todo = []
    for concept in concepts:
        resultfile = os.path.join(resultdir, '%s.txt'%concept)
        if checkToSkip(resultfile, overwrite):
            continue
        todo.append(concept)

    if not todo:
        printStatus(INFO, 'nothing to do')
        return 0

    try:
        holdoutfile = os.path.join(rootpath,collection,'ImageSets','holdout.txt')
        holdoutSet = set(map(str.strip,open(holdoutfile).readlines()))
    except:
        holdoutSet = set()

    hitlists = buildHitlists(collection, todo, tpp, rootpath)
    min_hit = 1e6
    max_hit = 0

    for concept in todo:
        resultfile = os.path.join(resultdir, '%s.txt' % concept)
        if checkToSkip(resultfile,overwrite):
            continue
        subconcepts = concept.split('-')
        labeledSet = set(hitlists[subconcepts[0]])
        for i in range(1,len(subconcepts)):
            labeledSet = labeledSet.intersection(hitlists[subconcepts[i]])
        labeledSet = labeledSet.difference(holdoutSet)
        if len(labeledSet) == 0:
            printStatus(INFO, '%s has ZERO hit' % concept)
        else:
            printStatus(INFO, '%s, %d hits -> %s' %(concept, len(labeledSet), resultfile))
            makedirsforfile(resultfile)
            fw = open(resultfile, 'w')
            fw.write('\n'.join(labeledSet) + '\n')
            fw.close()
        if len(labeledSet) > max_hit:
            max_hit = len(labeledSet)
        if len(labeledSet) < min_hit:
            min_hit = len(labeledSet)
            
    printStatus(INFO, 'max hits: %d, min hits: %d' % (max_hit, min_hit))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = OptionParser(usage="""usage: %prog [options] collection conceptfile""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--tpp", default="lemm", type="string", help="tag preprocess, can be raw, stem, or lemm (default: lemm)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])

if __name__ == "__main__":
    sys.exit(main())


             
