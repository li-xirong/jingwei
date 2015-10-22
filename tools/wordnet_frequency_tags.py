import os, sys
from nltk.corpus import wordnet as wn
import numpy as np
from bisect import bisect_left
import h5py

from basic.constant import ROOT_PATH
from basic.util import readImageSet, bisect_index

_min_freq = {'train10k': 50, 'train100k': 250, 'train1m': 750, 'mirflickr08': 50, 'flickr81test': 50, 'flickr81': 50,
             'flickr51': 50, 'imagenet166': 10}

def validateAnnotation(ann_syns, other_syns):
    for syn in other_syns[:]:
        if syn in ann_syns:
            return True
        if syn == syn.root_hypernyms()[0]:
            other_syns.remove(syn)

    if len(other_syns) == 0:
        return False

    res = False
    for syn in other_syns:
        res |= validateAnnotation(ann_syns, syn.hypernyms())
    return res


def _build_dict(tags_file, name):
    tags = []
    count_tags = {}
    with open(tags_file, 'r') as f:
        for line in f:
            line = line[:-2]
            data = line.split("\t")
            tags += [x.lower() for x in data[2].split(" ")]

    for t in tags:
        n = count_tags.get(t, 0)
        count_tags[t] = n + 1

    print "N tags init: ", len(count_tags)

    for t in count_tags.copy():
        if count_tags[t] < _min_freq[name]:
            if t in ['airplane', 'clouds', 'sky', 'sunset', 'night', 'food', 'people', 'person', "airport",
                     "animal", "beach", "baby", "bird", "car", "cloud", "dog", "flower", "girl", "man", "night",
                     "people", "portrait", "river", "sea", "tree",
                     "bear", "birds", "bird", "boats", "boat", "book", "bridge", "buildings", "building", "cars",
                     "car", "castle", "cat", "cityscape", "clouds", "cloud", "computer", "coral", "cow", "dancing",
                     "dog", "earthquake", "elk", "fire", "fish", "flags", "flag", "flowers", "flower", "food",
                     "fox", "frost", "garden", "glacier", "grass", "harbor", "horses", "horse", "house", "lake",
                     "leaf", "map", "military", "moon", "mountain", "nighttime", "ocean", "person", "plane",
                     "plants", "plant", "police", "protest", "railroad", "rainbow", "reflection", "road", "rocks",
                     "rock", "running", "sand", "sign", "sky", "snow", "soccer", "sports", "sport", "statue",
                     "street", "sun", "sunset", "surf", "swimmers", "swimmer", "tattoo", "temple", "tiger", "tower",
                     "town", "toy", "train", "tree", "valley", "vehicle", "water", "waterfall", "wedding", "whales",
                     "whale", "window", "zebra", "airshow", "apple", "aquarium", "basin", "beach", "bird", "bmw",
                     "car", "chicken", "chopper", "cow", "decoration", "dog", "dolphin", "eagle", "fighter",
                     "flame", "flower", "forest", "fruit", "furniture", "glacier", "hairstyle", "hockey", "horse",
                     "jaguar", "jellyfish", "lion", "matrix", "motocrycle", "olympics", "owl", "palace", "panda",
                     "penguin", "rabbit", "rainbow", "rice", "sailboat", "seagull", "shark", "snowman", "spider",
                     "sport", "starfish", "statue", "swimmer", "telephone", "triumphal", "turtle", "watch",
                     "waterfall", "weapon", "wildlife", "wolf"]:
                print "Warning: tag '%s' is retained but it has only %d frequency ( < %d )." % (t, count_tags[t],_min_freq[name])
            else:
                del count_tags[t]

    print "N tags post sample insuff: ", len(count_tags)

    # filter tags
    thing_categories = ["physical entity", "color", "thing", "artifact", "organism", "natural", 'clouds', 'sky',
                        'sunset', 'night', 'food', 'people', 'person', "airport", "animal", "beach", "bear",
                        "birds", "bird", "boats", "boat", "book", "bridge", "buildings", "building", "cars", "car",
                        "castle", "cat", "cityscape", "clouds", "cloud", "computer", "coral", "cow", "dancing",
                        "dog", "earthquake", "elk", "fire", "fish", "flags", "flag", "flowers", "flower", "food",
                        "fox", "frost", "garden", "glacier", "grass", "harbor", "horses", "horse", "house", "lake",
                        "leaf", "map", "military", "moon", "mountain", "nighttime", "ocean", "person", "plane",
                        "plants", "plant", "police", "protest", "railroad", "rainbow", "reflection", "road",
                        "rocks", "rock", "running", "sand", "sign", "sky", "snow", "soccer", "sports", "sport",
                        "statue", "street", "sun", "sunset", "surf", "swimmers", "swimmer", "tattoo", "temple",
                        "tiger", "tower", "town", "toy", "train", "tree", "valley", "vehicle", "water", "waterfall",
                        "wedding", "whales", "whale", "window", "zebra", "airshow", "apple", "aquarium", "basin",
                        "beach", "bird", "bmw", "car", "chicken", "chopper", "cow", "decoration", "dog", "dolphin",
                        "eagle", "fighter", "flame", "flower", "forest", "fruit", "furniture", "glacier",
                        "hairstyle", "hockey", "horse", "jaguar", "jellyfish", "lion", "matrix", "motocrycle",
                        "olympics", "owl", "palace", "panda", "penguin", "rabbit", "rainbow", "rice", "sailboat",
                        "seagull", "shark", "snowman", "spider", "sport", "starfish", "statue", "swimmer",
                        "telephone", "triumphal", "turtle", "watch", "waterfall", "weapon", "wildlife", "wolf"]
    thing_synsets = []
    for t in thing_categories:
        thing_synsets.extend(wn.synsets(t))

    for t in count_tags.copy():
        if wn.morphy(t) is None or len(t) < 3 or not validateAnnotation(thing_synsets, wn.synsets(
                t)): #or not t in vocabulary_50: #
            del count_tags[t]

    print "N tags post wordnet filter: ", len(count_tags)

    vocab = count_tags.keys()
    #print count_tags

    return vocab, count_tags


##############
workingSet = os.path.split(os.path.realpath(os.path.curdir))[1]
id_images = readImageSet(workingSet, workingSet, ROOT_PATH)
id_images.sort()
#id_images = map(int, id_images)

resultfile = os.path.join('TextData', "lemm_wordnet_freq_tags.h5")
if os.path.exists(resultfile):
    print "File %s already exists. Aborting..." % resultfile
    sys.exit(1)

tags_file = os.path.join('TextData', "id.userid.lemmtags.txt")
if len(sys.argv) > 1:
    print "Getting vocabulary from %s" % sys.argv[1]
    otherCollection = h5py.File(sys.argv[1], 'r')
    vocab = list(otherCollection['vocab'])
    otherCollection.close()
else:
    # compute vocab
    print "Building vocabulary..."
    vocab, count_tags = _build_dict(tags_file, workingSet)
    vocab.sort()

N_tags = len(vocab)
print "N tags: ", N_tags

# load tags
id_tags = {}
with open(tags_file, 'r') as f:
    for line in f:
        line = line[:-2]
        data = line.split("\t")

        assert (len(data) == 3)
        id_image = data[0]
        tags = [x.lower() for x in data[2].split(" ")]

        final_tags = [t for t in tags if t in vocab]

        id_tags[id_image] = final_tags

N_images = len(id_tags)

print "N images: ", N_images

# build tag matrix
tagmatrix = np.zeros((N_images, N_tags), dtype=np.int8)

for i, id_im in enumerate(id_images):
    tags = id_tags[id_im]
    if len(tags) > 0:
        idx = [bisect_index(vocab, t) for t in tags]
        tagmatrix[i, idx] = True

# save output
fout = h5py.File(resultfile, 'w')
fout['tagmatrix'] = tagmatrix
fout['vocab'] = vocab
fout['id_images'] = id_images
fout.close()
