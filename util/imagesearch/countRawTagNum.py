import sys
import os

from basic.constant import ROOT_PATH
from basic.common import checkToSkip,makedirsforfile

if __name__ == "__main__":
    overwrite = 0
    collection = sys.argv[1]
    
    resultfile = os.path.join(ROOT_PATH, collection, "MetaData", "id.rawtagnum.txt")
    if checkToSkip(resultfile, overwrite):
        sys.exit(0)
        
    tagfile = os.path.join(ROOT_PATH, collection, "TextData", "id.userid.rawtags.txt")
    results = []
    
    for line in open(tagfile):
        elems = str.split(line.strip())
        name = elems[0]
        rawtagnum = len(elems)-2
        assert(rawtagnum>0)
        results.append((name,rawtagnum))
        
    results.sort(key=lambda v:(v[1],v[1]))    
    makedirsforfile(resultfile)    
    fout = open(resultfile, "w")
    fout.write("".join(["%s %s\n" % (x[0], x[1]) for x in results]))
    fout.close()
        
    

