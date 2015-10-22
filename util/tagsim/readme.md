

# tagsim


Semantic similarity between two tags, measured either in terms of tag co-occurrence or by WordNet, is used in tag based methods such as [SemanticField](../tag_based/semantictagrel.py) and tag + image based methods such as [TagRanking](../instance_based/tagranking.py) and [RobustPCA](../robustpca/robustpca.py). This folder provides implementations for the following tag similarities:

1.  [Flickr Context Similarity](flickr_similarity.py) 
2.  [WUP Similarity](wordnet_similarity.py)
3.  [Average/Multiplication combination](combined_similarity.py) of the Flickr similarity and the WUP similarity

## Dependencies

+  WUP and the combined similarity: [NLTK](www.nltk.org) 
+  Flickr: Precomputed tag occurrence (`train10k/TextData/lemmtag.userfreq.imagefreq.txt`) and co-occurrence (`train10k/TextData/ucij.uuij.icij.iuij.txt`)


## API

Initialize the Flickr similarity based on tag statistics computed from train10k:
```Python
from tagsim.flickr_similarity import FlickrContextSim

collection = 'train10k'
fcs = FlickrContextSim(collection)
```

Compute the Flickr similarity between two tags:
```Python
fcs.compute('dog', 'pet')
```


Initialize the WUP similarity:
```python
from tagsim.wordnet_similarity import WordnetSim

wnsim = WordnetSim("wup")
```

Compute the WUP similarity between two tags:
```Python
wnsim.compute('dog', 'pet')
```



