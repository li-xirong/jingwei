import sys, os, math
import cPickle as pickle

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile,checkToSkip, niceNumber, printStatus
from basic.util import readImageSet
from basic.annotationtable import readConcepts, readAnnotationsFrom
from tagdb import TagBase, TagReader

DEFAULT_BONUS = 0   
DEFAULT_M = 25
DEFAULT_KR = 4
DEFAULT_KS = 9
DEFAULT_KD = 11
DEFAULT_KC = 1
DEFAULT_FEAT = 'vgg-verydeep-16-fc7relul2'

INFO = __file__

class ConceptRankBase:
    def __init__(self, datafile):
        data = pickle.load(open(datafile,'rb'))
        self.vob = data['tags']
        self.rank_matrix = data['rank_matrix']
        self.concepts = data['concepts']
        
        self.tag_num = len(self.vob)
        self.con_num = len(self.concepts)
        assert(self.rank_matrix.shape[0] == self.tag_num)
        assert(self.rank_matrix.shape[1] == self.con_num)
        
        self.tag2idx = dict(zip(self.vob, xrange(self.tag_num)))
        self.con2idx = dict(zip(self.concepts, xrange(self.con_num)))
        
    def get_rank(self, tag, concept):
        t_idx = self.tag2idx.get(tag,-1)
        c_idx = self.con2idx.get(concept, -1)
        #print tag, concept, t_idx, c_idx
        if t_idx < 0: # or c_idx < 0:
            return self.tag_num
        return self.rank_matrix[t_idx,c_idx]
    
    def get_ranks(self, tag):
        t_idx = self.tag2idx[tag]
        return self.rank_matrix[t_idx,:]
    
    def contain(self, tag):
        return tag in self.tag2idx
        
'''
Concept rank per image determined by TagVote
'''
class VisualConceptRankBase (ConceptRankBase):
    def __init__(self, datafile):
        data = pickle.load(open(datafile,'rb'))
        self.vob = map(str, data['id_images']) # in order to be consistent with ConceptRankBase, image is treated as tag.
        self.rank_matrix = data['rank_matrix']
        self.concepts = data['concepts']
        
        self.tag_num = len(self.vob)
        self.con_num = len(self.concepts)
        assert(self.rank_matrix.shape[0] == self.tag_num)
        assert(self.rank_matrix.shape[1] == self.con_num)
        
        self.tag2idx = dict(zip(self.vob, xrange(self.tag_num)))
        self.con2idx = dict(zip(self.concepts, xrange(self.con_num)))

    def get_rank(self, tag, concept):
        t_idx = self.tag2idx[tag]
        c_idx = self.con2idx[concept]
        #print tag, concept, t_idx, c_idx
        return self.rank_matrix[t_idx,c_idx]
    
        
def rank_promotion(r, k_r=4):
    return float(k_r)/(k_r + r-1)
    
    
class TagCooccurTagger:

    def __init__(self, testCollection, trainCollection, annotationName, rootpath=ROOT_PATH):
        self.name = '%s-%s-%s' % (self.__class__.__name__, trainCollection, annotationName)
        self.concepts = readConcepts(trainCollection, annotationName, rootpath)
        self.concept_num = len(self.concepts)
        self.concept2index = dict(zip(self.concepts, range(self.concept_num)))
        self.tbase = TagBase(trainCollection, tpp='lemm', rootpath=rootpath)
        self.rbase = ConceptRankBase(os.path.join(rootpath,trainCollection,'TextData', 'tag.concept-rank.%s.pkl' % annotationName))
        self.DEFAULT_RANK = self.tbase.tag_num()
        self.m = DEFAULT_M
        self.k_r = DEFAULT_KR
        self.k_s = DEFAULT_KS
        self.k_d = DEFAULT_KD
        self.normalize = True
        self.add_bonus = False
        
    def _compute_relevance_score(self, conceptIndex, content, context):
        score = 0.0
        
        for u,ranks in context:
            r = ranks[conceptIndex]
            vote = int(r <= self.m)
            if vote == 0: # the vote counts ONLY WHEN this concept $conceptIndex is within the top-m ranked concepts with respect to the user tag $u
                continue
            promotion = rank_promotion(r, self.k_r) * self.tbase.stability(u, self.k_s) * self.tbase.descriptiveness(self.concepts[conceptIndex], self.k_d)
            score += promotion
            #print self.concepts[conceptIndex], rank_promotion(rank, self.k_r), self.tbase.stability(u, self.k_s), self.tbase.descriptiveness(c, self.k_d)
        
        if self.normalize:
            score /= max(1, len(context))
        
        return score
        
    '''
    context is user tags assigned to a specific image
    '''
    def _compute(self, content, context):
        user_tags = []
        for tag in context.split():
            if tag not in user_tags and self.rbase.contain(tag):
                user_tags.append(tag)
        
        new_context = [(u,self.rbase.get_ranks(u)) for u in user_tags]
        scores = [self._compute_relevance_score(i, content, new_context) for i in range(self.concept_num)]
        
        # use bonus to rank the original user tags at the top
        if self.add_bonus:
            bonus = [0] * self.concept_num
            init_score = 1.0
            STEP = 0.1
            for tag in user_tags:
                c_idx = self.concept2index.get(tag, -1)
                if c_idx >= 0:
                    bonus[c_idx] = init_score
                    init_score -= STEP
            scores = [scores[i]+bonus[i] for i in range(self.concept_num) ]
        
        return scores

    def predict(self, content, context):
        scores = self._compute(content, context)
        return sorted(zip(self.concepts, scores), key=lambda v:v[1], reverse=True)

 
class TagCooccurPlusTagger (TagCooccurTagger):

    def __init__(self, testCollection, trainCollection, annotationName, feature="vgg-verydeep-16-fc7relul2", rootpath=ROOT_PATH):
        TagCooccurTagger.__init__(self, testCollection, trainCollection, annotationName, rootpath)
        datafile = os.path.join(rootpath, testCollection, 'autotagging', '%s_%s_tagvote,%s_%s_rank.pkl'% (trainCollection, testCollection, feature, annotationName[:-4]))
        self.visual_rbase = VisualConceptRankBase(datafile)
        self.k_c = DEFAULT_KC

    def _compute_relevance_score(self, conceptIndex, content, context):
        score = 0.0
        voted = 0
        
        for u,ranks in context:
            r = ranks[conceptIndex]
            vote = int(r <= self.m)
            if vote == 0:
                continue
            promotion = rank_promotion(r, self.k_r) * self.tbase.stability(u, self.k_s) * self.tbase.descriptiveness(self.concepts[conceptIndex], self.k_d)
            r_c = self.visual_rbase.get_rank(content, self.concepts[conceptIndex])
            promotion *= rank_promotion(r_c, self.k_c)
            score += promotion
            voted += 1
        
        if self.normalize:
            #score /= max(1, voted)
            score /= max(1, len(context)) # len(context) is better as a normalization factor than voted
        
        return score
    
if __name__ == '__main__':
    rootpath = ROOT_PATH
    trainCollection = 'train10k'
    testCollection = 'mirflickr08'
    annotationName = 'conceptsmir14.txt'
    
    t1 = TagCooccurTagger(testCollection, trainCollection, annotationName, rootpath=rootpath)
    t2 = TagCooccurPlusTagger(testCollection, trainCollection, annotationName, rootpath=rootpath)
    for tagger in [t1, t2]:
        print tagger.predict('1', 'dog pet animal')
    

