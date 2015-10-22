import os, sys, array, shutil
import numpy as np
import glob

concepts = open('Annotations/' + sys.argv[1]).read().strip().split('\r\n')
print concepts
os.mkdir('TextData/tagged,lemm')

h_files = dict([(C, open('TextData/tagged,lemm/' + C + ".txt", 'w')) for C in concepts])
cnt = 0
with open('TextData/id.userid.lemmtags.txt') as f:
	for line in f:
		cnt += 1
		if len(line) < 1:
			continue
		id_im,uid,tags = line.split('\t')
		tags = tags.strip().split()
	
		for C in concepts:
			if C in tags:
				h_files[C].write("%s\n" % id_im)
	
for h in h_files:
	h_files[h].close()
	
print "Processed %d lines, %d concepts" % (cnt, len(concepts))
