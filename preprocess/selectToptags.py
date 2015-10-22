import sys
import os
import re


from basic.constant import ROOT_PATH
from basic.data import CAMERA_BRAND_SET,PHOTO_TAG_SET
from basic.common import checkToSkip
from basic.annotationtable import writeConceptsTo
from nltk.corpus import wordnet as wn


_digits = re.compile('\d')


if __name__ == '__main__':
    rootpath = ROOT_PATH
    overwrite = 0
    collection = 'train1m'
    collection = sys.argv[1]
    N = int(sys.argv[2])
    tagfreqfile = os.path.join(rootpath,collection,'TextData','lemmtag.userfreq.imagefreq.txt')
    annotationName = 'concepts%stop%d.txt' % (collection,N)
    
    conceptfile = os.path.join(rootpath,collection,'Annotations',annotationName)
    if checkToSkip(conceptfile,overwrite):
        sys.exit(0)

    concepts = []
    
    nr_short = 0
    nr_camera = 0
    nr_photo = 0
    nr_digit = 0
    nr_nonwn = 0
    
    tag2freq = {}
    
    for line in open(tagfreqfile).readlines():
        tag,userfreq,imfreq = str.split(line.strip())
        if int(userfreq) < 100:
            continue

        tag2freq[tag] = int(imfreq)
        if len(tag)<3:
            print 'too short:', tag
            nr_short += 1
            continue
        if tag in CAMERA_BRAND_SET:
            print 'camera:', tag
            nr_camera += 1
            continue
        if tag in PHOTO_TAG_SET:
            print 'photo:', tag
            nr_photo += 1
            continue
        if bool(_digits.search(tag)):
            print 'digit:', tag
            nr_digit += 1
            continue
        try:
            if wn.synsets(tag):
                concepts.append(tag)
            else:
                print 'non wordnet:', tag
                nr_nonwn += 1
        except:
            print 'non wordnet:', tag
            nr_nonwn += 1
            continue
    
        if len(concepts)>=N:
            break
    
    for concept in concepts:
        print concept, tag2freq[concept]
    print '-'*50    
    writeConceptsTo(concepts,collection,annotationName,rootpath=rootpath)
    print 'short tags', nr_short
    print 'camera', nr_camera
    print 'photo', nr_photo
    print 'digit', nr_digit
    print 'non wordnet', nr_nonwn
    print 'nr of concepts:', len(concepts)
    
    
        
