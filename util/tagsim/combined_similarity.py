
from basic.constant import ROOT_PATH
from flickr_similarity import FlickrContextSim
from wordnet_similarity import WordnetSim

class AvgCombinedSim:
    
    def __init__(self, collection, rootpath=ROOT_PATH):
        self.wnsim = WordnetSim("wup")
        self.fcsim = FlickrContextSim(collection, rootpath)
        
    def compute(self, tagx, tagy, gamma=None, img=1):
        fcs = self.fcsim.compute(tagx, tagy, gamma, img)
        wns = self.wnsim.compute(tagx, tagy)
        return (fcs + wns) * 0.5

class MulCombinedSim (AvgCombinedSim):
    def compute(self, tagx, tagy, gamma=None, img=1):
        fcs = self.fcsim.compute(tagx, tagy, gamma, img)
        wns = self.wnsim.compute(tagx, tagy)
        return fcs * wns



if __name__ == "__main__":
    
    collection = 'train10k'
    fcs = FlickrContextSim(collection)
    avgcos = AvgCombinedSim(collection)
    mulcos = MulCombinedSim(collection)
    wns = WordnetSim('wup')

    tags = str.split('nature waterfall mountain 2012 bmw airshow jet airport beach food dog car cat animal street')
    for tagx in tags:
        for simclass in [wns, fcs, avgcos, mulcos]:
            taglist = [(tagy,simclass.compute(tagx,tagy)) for tagy in tags] # fcs.vob]
            taglist.sort(key=lambda v:v[1], reverse=True)
            print '%s(%s)'%(simclass.__class__.__name__, tagx), ' '.join(['%s %g' % (x[0],x[1]) for x in taglist[:10]])
            print ''



        
        

