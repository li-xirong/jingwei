
import sys
import os

from nltk.corpus import wordnet as wn


def getChildList(synset):
    names = []
    lemma_names = []
    try:
        children = synset.hyponyms()

        for child in children:
            if type(child.lemma_names) is list:
                lemma_names = child.lemma_names
            else:
                lemma_names += child.lemma_names()
            names += lemma_names
            deeper = getChildList(child)
            names += deeper
    except:
        traceback.print_exc(file=sys.stdout)

    return names


def wn_expand(tag):
    names = []
    synsets = wn.synsets(tag)
    for synset in synsets:
        if type(synset.lemma_names) is list:
            lemma_names = synset.lemma_names
        else:
            lemma_names = synset.lemma_names()
        names += lemma_names + getChildList(synset)
    # make the list unique
    temp = set()
    names = [x.lower() for x in names if x.lower() not in temp and not temp.add(x.lower())]
    return names



if __name__ == '__main__':
    for tag in str.split('2012 aerial aeroplane airplane dog cat portrait jaguar'):
        newtags = wn_expand(tag)
        print tag, len(newtags), newtags[:10]
        print 'jaguar' in newtags
            
        
            
