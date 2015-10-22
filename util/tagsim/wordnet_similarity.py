from nltk.corpus import wordnet as wn

def wup_similarity(tagx, tagy):
    scores = []
    for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADJ_SAT, wn.ADV]:
        try:
            synsetx = wn.synset('%s.%s.01' % (tagx,pos))
            synsety = wn.synset('%s.%s.01' % (tagy,pos))
            score = synsetx.wup_similarity(synsety)
            if score is None:
                score = 0
        except Exception, e:
            score = 0
        scores.append(score)
    return max(scores)


WN_SIMILARITY = {"wup":wup_similarity}
    
class WordnetSim:
    def __init__(self, sim="wup"):
        self.simfunc = WN_SIMILARITY[sim]
        print ("[tagsim.wordnet_similarity.%s] %s similarity" % (self.__class__.__name__, sim))
        
    def compute(self, tagx, tagy):
        return self.simfunc(tagx, tagy)
        
        

if __name__ == '__main__':
    wns = WordnetSim('wup')
    tags = str.split('dog cat car x1 beach animal river lake vehicle boat bicycle')
    for tagx in tags:
        ranklist = [(tagy,wns.compute(tagx,tagy)) for tagy in tags]
        ranklist.sort(key=lambda v:v[1], reverse=True)
        print tagx, ranklist
