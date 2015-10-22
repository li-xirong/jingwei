import web
import os, sys, random
import json

from basic.common import readRankingResults
from basic.annotationtable import readAnnotationsFrom
from basic.util import readLabeledImageSet, readImageSet
from basic.metric import getScorer

from images import images, bigimages


urls = (
    '/', 'index',
    '/search', 'ImageSearch',
    '/images/(.*)', 'images',
    '/img/(.*)', 'images',
    '/images2/(.*)', 'bigimages'
)
       
render = web.template.render('templates/')

pwd = os.path.dirname(os.path.realpath(__file__))
config = json.load(open(os.path.join(pwd,'config.json')))

max_hits = config['max_hits']
rootpath = config['rootpath']
collection = config['collection']
rankMethod = config['rankMethod']
annotationName = config['annotationName']
metric = config['metric']
scorer = getScorer(metric)

simdir = os.path.join(rootpath, collection, 'SimilarityIndex', collection, rankMethod)
imset = readImageSet(collection, collection, rootpath)



class index:
    
    def GET(self):
        input = web.input(query=None)
        resp = {'status':0, 'hits':0, 'random':[], 'tagrel':[], 'metric':metric, 'perf':0}

        if input.query:
            resp['status'] = 1
            resp['query'] = input.query
            query = input.query.lower()

            if query.isdigit(): # request to view a specific image
                resp['hits'] = 1
                resp['tagrel'] = [{'id':query}]
                return  render.index(resp)
            
            try:
                names,labels = readAnnotationsFrom(collection, annotationName, query)
                name2label = dict(zip(names,labels))
            except Exception, e:
                name2label = {}

            content = []
            try:
                if input.tagrel == '0':
                    labeled = readLabeledImageSet(collection, query, rootpath=rootpath)
                    ranklist = [(x,0) for x in labeled]
                else:
                    simfile = os.path.join(simdir, '%s.txt' % query)
                    ranklist = readRankingResults(simfile)
                resp['hits'] = len(ranklist)
                for name,score in ranklist:
                    color = 'Chartreuse' if name2label.get(name,0)>0 else 'red'
                    color = 'white' if name not in name2label else color
                    res = {'id':name, 'color':color}
                    content.append(res)
                resp['perf'] = 0 if not name2label else scorer.score([name2label[x[0]] for x in ranklist if x[0] in name2label])
                resp['tagrel'] = content[:max_hits]
            except:
                None
        else:
            selected = random.sample(imset, max_hits)
            resp['random'] = [{'id':x, 'color':'white'} for x in selected] 
        return render.index(resp)

class ImageSearch:
    def POST(self):
        input = web.input()
        raise web.seeother('/?query=%s&tagrel=1' % input.tags)


        
if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
