import sys
import os


from basic.constant import ROOT_PATH
from basic.common import checkToSkip, printStatus

INFO = __file__

def process(options, collection):
    rootpath = options.rootpath
    tpp = options.tpp

    tagfile = os.path.join(rootpath, collection, "TextData", "id.userid.%stags.txt" % tpp)
    resultfile = os.path.join(rootpath, collection, "TextData", "%stag.userfreq.imagefreq.txt" % tpp)
    if checkToSkip(resultfile, options.overwrite):
        return 0
        
    printStatus(INFO, "parsing " + tagfile)
       
    tag2imfreq = {}
    tag2users = {}

    for line in open(tagfile):
        elems = str.split(line.strip())
        photoid = elems[0]
        userid = elems[1]
        tagset = set(elems[2:])
            
        for tag in tagset:
            tag2imfreq[tag] = tag2imfreq.get(tag, 0) + 1
            tag2users.setdefault(tag,[]).append(userid)
            
    printStatus(INFO, "collecting user-freq and image-freq")
    results = []
    for tag,users in tag2users.iteritems():
        userfreq = len(set(users))
        imfreq = tag2imfreq[tag]
        results.append((tag, userfreq, imfreq))
    
    printStatus(INFO, "sorting in descending order (user-freq as primary key)")
    results.sort(key=lambda v:(v[1],v[2]), reverse=True)
    printStatus(INFO, "-> %s" % resultfile)

    with open(resultfile, 'w') as fw:
        fw.write(''.join(['%s %d %d\n' % (tag, userfreq, imfreq) for (tag, userfreq, imfreq) in results]))
        fw.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--tpp", default='lemm', type="string", help="tag preprocess (default: lemm)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath where the train and test collections are stored (default: %s)" % ROOT_PATH)
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())    

