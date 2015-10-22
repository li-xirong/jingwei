import math

VALID_LABEL_SET_AP = set([-1, 0, 1, 2, 3])
VALID_LABEL_SET_NDCG2 = set([0, 1, 2, 3])

class MetricScorer:

    def __init__(self, k=0):
        self.k = k

    def score(self, sorted_labels):
        return 0.0

    def getLength(self, sorted_labels):
        length = self.k
        if length>len(sorted_labels) or length<=0:
            length = len(sorted_labels)
        return length    

    def name(self):
        if self.k > 0:
            return "%s@%d" % (self.__class__.__name__.replace("Scorer",""), self.k)
        return self.__class__.__name__.replace("Scorer","")


class APScorer (MetricScorer):
 
    def __init__(self, k):
        MetricScorer.__init__(self, k)
        

    def score(self, sorted_labels):
        nr_relevant = len([x for x in sorted_labels if x > 0])
        if nr_relevant == 0:
            return 0.0
            
        length = self.getLength(sorted_labels)
        ap = 0.0
        rel = 0
        
        for i in range(length):
            lab = sorted_labels[i]
            assert(lab in VALID_LABEL_SET_AP)
            if lab >= 1:
                rel += 1
                ap += float(rel) / (i+1.0)
        ap /= nr_relevant
        return ap

# reciprocal rank
class RRScorer (MetricScorer):

    def score(self, sorted_labels):
        for i in range(len(sorted_labels)):
            if 1 <= sorted_labels[i]:
                return 1.0/(i+1)
        return 0.0


class PrecisionScorer (MetricScorer):

    def score(self, sorted_labels):
        length = self.getLength(sorted_labels)

        rel = 0
        for i in range(length):
            if sorted_labels[i] >= 1:
                rel += 1

        return float(rel)/length

    
class NDCGScorer (PrecisionScorer):


    def score(self, sorted_labels):
        d = self.getDCG(sorted_labels)
        d2 = self.getIdealDCG(sorted_labels) 
        #print '\n', d, d2
        return d/d2
        
    def getDCG(self, sorted_labels):
        length = self.getLength(sorted_labels)

        dcg = max(sorted_labels[0], 0)
        #print dcg
        for i in range(1, length):
            rel = max(sorted_labels[i], 0)
            dcg += float(rel)/math.log(i+1, 2)
            #print i, sorted_labels[i], math.log(i+1,2), float(sorted_labels[i])/math.log(i+1, 2)
        return dcg

    def getIdealDCG(self, sorted_labels):
        ideal_labels = sorted(sorted_labels, reverse=True)
        assert(ideal_labels[0] > 0), len(ideal_labels)
        return self.getDCG(ideal_labels)


class NDCG2Scorer (NDCGScorer):

   def getDCG(self, sorted_labels):
        length = self.getLength(sorted_labels)
        dcg = 0
        for i in range(0, length):
            rel_i = max(sorted_labels[i], 0) 
            #assert(rel_i in VALID_LABEL_SET_NDCG2)
            dcg += (math.pow(2,rel_i) - 1) / math.log(i+2, 2)
        return dcg
        
        
def getScorer(name):
    mapping = {"P":PrecisionScorer, "AP":APScorer, "RR":RRScorer, "NDCG":NDCGScorer, "NDCG2":NDCG2Scorer}
    elems = name.split("@")
    if len(elems) == 2:
        k = int(elems[1])
    else:
        k = 0
    return mapping[elems[0]](k)
   

if __name__ == "__main__":
    sorted_labels = [1, 1, 0, 0, 0]
    sorted_labels = [3, 2, 3, -1, 1, 2]
    nr_relevant = len([x for x in sorted_labels if x > 0])
        
    for scorer in [APScorer(0), APScorer(1), APScorer(2), APScorer(3), PrecisionScorer(1), PrecisionScorer(2), PrecisionScorer(10), NDCGScorer(10), RRScorer(0)]:
        print scorer.name(), scorer.score(sorted_labels)
 
    
    sorted_labels = [3, 2, 3, 0, 1, 2]
    
    for k in range(1, 11):
        scorer1 = getScorer('NDCG@%d'%k)
        scorer2 = getScorer('NDCG2@%d'%k)
        print k, scorer1.score(sorted_labels), scorer2.score(sorted_labels)


