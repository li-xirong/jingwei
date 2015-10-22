import web
import os
import json

pwd = os.path.dirname(os.path.realpath(__file__))
config = json.load(open(os.path.join(pwd,'config.json')))

imagedata_path = config['imagedata_path']
collection = config['collection']

COLLECTION_RENAMED = {} #{'flickr81':'nuswide'}

cType = {"png":"images/png",
         "jpg":"images/jpeg",
         "jpeg":"images/jpeg",
         "gif":"images/gif",
         "ico":"images/x-icon"}

def im2path(name, big=False):
    img_folder = 'ImageData' if big else 'ImageData128x128'
    img_dir = os.path.join(imagedata_path, COLLECTION_RENAMED.get(collection, collection), img_folder)
    subfolder = name[-1] if collection == 'mirflickr08' else name[-3:]
    ext = '.jpg'
    local_file_name = os.path.join(img_dir, subfolder, name + ext)
    return local_file_name


class images:
    def get_local(self, name):
        return im2path(name, False)    

    def GET(self,name):
        ext = name.split(".")[-1] # Gather extension
        if name.find('.')<0:
            ext = 'jpg'
        imfile = self.get_local( os.path.splitext(name)[0] )
        #print imfile
        try:
            web.header("Content-Type", cType[ext]) # Set the Header
            return open(imfile,"rb").read() # Notice 'rb' for reading images
        except:
            raise web.notfound()

class bigimages (images):
    def get_local(self, name):
        return im2path(name, True)

            
if __name__ == '__main__':
    im = bigimages()
    im.GET('4362444639.jpg')

