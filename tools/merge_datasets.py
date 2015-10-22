import os
import sys
import shutil
from basic.constant import ROOT_PATH
from basic.common import makedirsforfile
from basic.util import readImageSet

codepath = "/home/urix/shared/tagrelcodebase"
datapath = ROOT_PATH

if len(sys.argv) < 4:
	print "Usage: merge_datasets.py trainCollection testCollection feature"
	sys.exit(1)

coll1 = sys.argv[1]
coll2 = sys.argv[2]
feature = sys.argv[3]

coll1_features_file = "%s/%s/FeatureData/%s/feature.bin" % (datapath, coll1, feature)
coll2_features_file = "%s/%s/FeatureData/%s/feature.bin" % (datapath, coll2, feature)
new_features_file = "%s/%s+%s/FeatureData/%s/feature.bin" % (datapath, coll1, coll2, feature)
makedirsforfile(new_features_file)

coll1_shape_file = "%s/%s/FeatureData/%s/shape.txt" % (datapath, coll1, feature)
coll2_shape_file = "%s/%s/FeatureData/%s/shape.txt" % (datapath, coll2, feature)
new_shape_file = "%s/%s+%s/FeatureData/%s/shape.txt" % (datapath, coll1, coll2, feature)
makedirsforfile(new_shape_file)

# shape file
with open(new_shape_file, 'w') as fout:
	imA, featA = open(coll1_shape_file).read().strip().split(" ")
	imB, featB = open(coll2_shape_file).read().strip().split(" ")
	assert featA == featB

	fout.write('%d %d' % (int(imA) + int(imB), int(featA)))

# copy and concatenate features
file(new_features_file,'wb').write(file(coll1_features_file,'rb').read() + file(coll2_features_file,'rb').read())

# copy Annotations
shutil.copytree("%s/%s/Annotations" % (datapath, coll1), "%s/%s+%s/Annotations" % (datapath, coll1, coll2))

# read ids
testset_id_images = readImageSet(coll2, coll2, datapath)
testset_id_images = set(map(int, testset_id_images))

train_id_images = readImageSet(coll1, coll1, datapath)
train_id_images = set(map(int, train_id_images))

base_new_id = max(testset_id_images.union(train_id_images)) + 1
duplicates = testset_id_images.intersection(train_id_images)
duplicates = dict([(x, x+base_new_id) for x in duplicates])

print "Found %d duplicates." % len(duplicates)

# read id.txt
coll1_featid_file = "%s/%s/FeatureData/%s/id.txt" % (datapath, coll1, feature)
coll2_featid_file = "%s/%s/FeatureData/%s/id.txt" % (datapath, coll2, feature)
new_featid_file = "%s/%s+%s/FeatureData/%s/id.txt" % (datapath, coll1, coll2, feature)
makedirsforfile(new_featid_file)

coll1_ids = map(int, open(coll1_featid_file, 'r').read().strip().split())
coll2_ids = map(int, open(coll2_featid_file, 'r').read().strip().split())
#duplicates = set(coll1_ids).intersection(set(coll2_ids))
#duplicates = dict([(x, x+base_new_id) for x in duplicates])

coll1_ids = [str(duplicates.get(x, x)) for x in coll1_ids]
coll2_ids = map(str, coll2_ids)
with open(new_featid_file, 'w') as fout:
	fout.write(" ".join(coll1_ids + coll2_ids))

# read coll.txt
coll1_featid_file = "%s/%s/ImageSets/%s.txt" % (datapath, coll1, coll1)
coll2_featid_file = "%s/%s/ImageSets/%s.txt" % (datapath, coll2, coll2)
new_featid_file = "%s/%s+%s/ImageSets/%s+%s.txt" % (datapath, coll1, coll2, coll1, coll2)
makedirsforfile(new_featid_file)

coll1_ids = map(int, open(coll1_featid_file, 'r').read().strip().split('\n'))
coll2_ids = map(int, open(coll2_featid_file, 'r').read().strip().split('\n'))
#duplicates = set(coll1_ids).intersection(set(coll2_ids))
#duplicates = dict([(x, x+base_new_id) for x in duplicates])

coll1_ids = [str(duplicates.get(x, x)) for x in coll1_ids]
coll2_ids = map(str, coll2_ids)
with open(new_featid_file, 'w') as fout:
	fout.write("\r\n".join(coll1_ids + coll2_ids))

# read tags
coll1_tags_file = "%s/%s/TextData/id.userid.lemmtags.txt" % (datapath, coll1)
coll2_tags_file = "%s/%s/TextData/id.userid.lemmtags.txt" % (datapath, coll2)
new_tags_file = "%s/%s+%s/TextData/id.userid.lemmtags.txt" % (datapath, coll1, coll2)
makedirsforfile(new_tags_file)

# copy lemm tags file if exists
# if os.path.exists("%s/%s/TextData/lemm_wordnet_freq_tags.h5" % (datapath, coll1)):
# 	file("%s/%s+%s/TextData/lemm_wordnet_freq_tags.h5" % (datapath, coll1, coll2),'wb').write(file("%s/%s/TextData/lemm_wordnet_freq_tags.h5" % (datapath, coll1),'rb').read())

coll1 = [x.split('\t') for x in open(coll1_tags_file, 'r')]
coll1 = ["%d\t%s\t%s" % (duplicates.get(int(x[0]), int(x[0])), x[1], x[2]) for x in coll1]
coll2 = open(coll2_tags_file, 'r').read().strip()

with open(new_tags_file, 'w') as fout:
	fout.write("".join(coll1))
	fout.write("".join(coll2))
