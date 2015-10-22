import sys, os, math

from basic.constant import ROOT_PATH
from basic.common import printStatus

def generate_key(tag1, tag2):
    if tag1<tag2:
        return '%s,%s' % (tag1,tag2)
    return '%s,%s' % (tag2,tag1)
    
class TagBase:

    def __init__(self, collection, tpp='lemm', rootpath=ROOT_PATH):
        freqfile = os.path.join(rootpath, collection, 'TextData', '%stag.userfreq.imagefreq.txt'%tpp)
        self.tag2freq = {}
        for line in open(freqfile).readlines():
            elems = line.split()
            self.tag2freq[elems[0]] = int(elems[2])
        
        vobfile = os.path.join(rootpath, collection, 'TextData', 'wn.%s.txt' % collection)
        self.vob = set(map(str.strip, open(vobfile).readlines()))
        self.vob = self.vob.intersection(set(self.tag2freq.keys()))
        print '[%s] vob %d' % (self.__class__.__name__, len(self.vob))
        
        

    def freq(self, tag):
        return self.tag2freq.get(tag, 0)
        
    # Implement Equation (6) in www08-borkur
    def stability(self, tag, k_s=9):
        c = max(self.freq(tag), 1)
        return k_s / (k_s + abs(k_s - math.log(c, 2)))
        
    # Implement Equation (7) in www08-borkur
    def descriptiveness(self, tag, k_d=11):
        c = max(self.freq(tag), 1)
        return k_d / (k_d + abs(k_d - math.log(c, 2)))

    def contain(self, tag):
        return tag in self.tag2freq
        
    def tag_num(self):
        return len(self.vob)


class TagCooccurBase (TagBase):
    def __init__(self, collection, tpp='lemm', rootpath=ROOT_PATH):
        TagBase.__init__(self, collection, tpp, rootpath)
        jointfreqfile =  os.path.join(rootpath, collection, "TextData", "ucij.uuij.icij.iuij.txt")
        self.jointfreq = {}
       
        for line in open(jointfreqfile).readlines():
            t1, t2, userjointfreq, userunionfreq, imagejointfreq, imageunionfreq = str.split(line)
            key = generate_key(t1, t2)
            self.jointfreq[key] = int(imagejointfreq)
       
    def jointFreq(self, tagx, tagy):
        key = generate_key(tagx, tagy)
        return self.jointfreq.get(key, 0)

    def top_cooccur(self, tag, m):
        taglist = [(x,self.jointFreq(x,tag)) for x in self.vob]
        taglist = [(tag, self.freq(tag))] + [x for x in taglist if x[1] > 0]
        taglist.sort(key=lambda v:v[1], reverse=True)
        return taglist[:m]


class TagReader:
    def __init__(self,collection,tpp='lemm', rootpath=ROOT_PATH):
        self.name = '%s(%s,%s)' % (self.__class__.__name__,collection,tpp)
        self.photoid2tags = {}
        datafile = os.path.join(rootpath,collection,"TextData", "id.userid.%stags.txt" % tpp)
        self.vob = []

        with open(datafile) as fin:
            for line in fin:
                [photoid, userid, tags] = line.split("\t")
                self.photoid2tags[photoid] = tags
                self.vob += str.split(tags)
        self.vob = set(self.vob) 
        printStatus(self.name, "%d images, %d unique tags" % (len(self.photoid2tags), len(self.vob)))
        

    def get(self, photoid):
        return self.photoid2tags.get(photoid, "")
        
if __name__ == '__main__':
    collection = 'train10k'
    tb = TagBase(collection)
    tcb = TagCooccurBase(collection)
 
    tr = TagReader(collection)
   
    for tag in str.split('dog cat car'):
        print tag, tb.freq(tag), tcb.top_cooccur(tag,3)

