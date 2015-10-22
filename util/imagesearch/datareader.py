import sys
import os

from basic.constant import ROOT_PATH


class AttributeReader:

    def __init__(self,collection,attr,rootpath=ROOT_PATH):
        self.name = '%s(%s,%s)' % (self.__class__.__name__,collection,attr)
        self.photoid2val = {}
        datafile = os.path.join(rootpath,collection,'MetaData','id.%s.txt'%attr)

        with open(datafile) as fin:
            for line in fin:
                photoid, val = str.split(line)
                self.photoid2val[photoid] = float(val)

        self.printInfo('%d images' % len(self.photoid2val))

    def get(self, photoid):
        return self.photoid2val.get(photoid, 0)

    def printInfo(self, info):
        print ('[%s] %s' % (self.name, info))


class TagReader (AttributeReader):

    def __init__(self,collection,tpp='lemm', rootpath=ROOT_PATH):
        self.name = '%s(%s,%s)' % (self.__class__.__name__,collection,tpp)
        self.photoid2tags = {}
        datafile = os.path.join(rootpath,collection,"TextData", "id.userid.%stags.txt" % tpp)

        with open(datafile) as fin:
            for line in fin:
                [photoid, userid, tags] = line.split("\t")
                self.photoid2tags[photoid] = tags

        self.printInfo("%d images" % len(self.photoid2tags))

    def get(self, photoid):
        return self.photoid2tags.get(photoid, "")


class TagrelReader (AttributeReader):
    
    def __init__(self, collection, dataset, tagrelMethod, nonnegative=1, rootpath=ROOT_PATH):
        self.name = '%s(%s,%s,%s)' % (self.__class__.__name__, collection, dataset, tagrelMethod)
        self.nonnegative = nonnegative 
        datafile = os.path.join(rootpath, collection, "tagrel", dataset, tagrelMethod, 'id.tagvotes.txt')
        self.load(datafile)
        self.printInfo("%d images" % len(self.photoid2tagrel))


    def load(self, datafile):
        self.photoid2tagrel = {}
        

        for line in open(datafile).readlines():
            elems = str.split(line.strip())
            photoid = elems[0]
            numtags = (len(elems)-1)/2

            tagrel = [(elems[1+2*i], float(elems[2+2*i])) for i in range(numtags)]
            #tagrel = [(tag, max(0, score)) for (tag, score) in tagrel]
            self.photoid2tagrel[photoid] = dict(tagrel)            

        
    def get(self, photoid, tag):
        if tag.find('-')>0: # bi-concepts
            score = min([self.photoid2tagrel.get(photoid,{}).get(x,0) for x in tag.split('-')])
        else:
            score = self.photoid2tagrel.get(photoid, {}).get(tag, 0)
        if self.nonnegative:
            return max(score, 0)
        return score


class AutotagReader (TagrelReader):
    def __init__(self,collection,dataset,autotagMethod,rootpath=ROOT_PATH):
        self.name = '%s(%s,%s,%s)' % (self.__class__.__name__,collection,dataset,autotagMethod)
        self.nonnegative = 0
        datafile = os.path.join(rootpath,collection,'autotagging',dataset,autotagMethod,'id.tagvotes.txt')
        self.load(datafile)
        self.printInfo('%d images' % len(self.photoid2tagrel))



class TagrankReader (AttributeReader):
    def __init__(self, collection, dataset, tagrelMethod, rootpath=ROOT_PATH):
        self.name = '%s(%s,%s,%s)' % (self.__class__.__name__, collection, dataset, tagrelMethod)
        datafile = os.path.join(rootpath, collection, "tagrel", dataset, tagrelMethod, 'id.tagvotes.txt')
        self.load(datafile)
        self.printInfo("%d images" % len(self.photoid2tagrank))

    def load(self, datafile):
        self.photoid2tagrank = {}
        
        for line in open(datafile).readlines():
            elems = str.split(line.strip())
            photoid = elems[0]
            numtags = (len(elems)-1)/2
            tagrank = [(elems[1+2*i], i+1) for i in range(numtags)]
            self.photoid2tagrank[photoid] = dict(tagrank)  

    def get(self, photoid, tag):
        return self.photoid2tagrank.get(photoid, {}).get(tag, 1e6)


if __name__ == "__main__":
    collection = 'flickr81'
    dataset = collection
    
    attrReader = AttributeReader(collection,attr='rawtagnum')
    tagReader = TagReader(collection)
    autotagReader = AutotagReader(collection,dataset,autotagMethod='train10k/concepts81.txt/preknn/color64+dsift,l1knn,1000')
    imset = str.split('1000048060 1000024674')

    for tagrelMethod in str.split('tagpos,lemm'):
        reader = TagrelReader(collection, dataset, tagrelMethod)
        tagrankReader = TagrankReader(collection,dataset,tagrelMethod)
        for im in imset:
            tags = str.split(tagReader.get(im))
            for tag in tags:
                print im,tag,reader.get(im,tag),tagrankReader.get(im,tag),autotagReader.get(im,tag)
                
    
