

FEATURE_TO_DIM = {"color64":64, "ehdl2":80, "pix32x32l2":1024, "cslbp80":80, "gist":960, "dsift":1024, 'color64+dsift':1088,
                  "color64pca38":38, "cslbp80pca12":12, "gistpca136":136, "dsiftpca225":225, "rgbsift":1000,
                  'getlf':256, 'colorhist':576, 'gist480':480,
                  'opponentsift2048':2048, 'opponentsift512':512,
                  'sift':1000, 'siftsp':5000, 'csiftsp':5000, 'rgbsiftsp':5000, 'opponentsiftsp':5000,
                  'bow400':400, 'bow400pca205':205, 
                  'tag400_train10k_fastlinear':400, 'tag400_train100k_fastlinear':400, 'tag400_train1m_fastlinear':400,
                  'tag400_train10k_fik50':400, 'tag400_train100k_fik50':400, 'tag400_train1m_fik50':400,
                  'tag400_train10k_fastlinear+color64+dsift':1488, 'tag400_train100k_fastlinear+color64+dsift':1488, 'tag400_train1m_fastlinear+color64+dsift':1488,
                  'tag400_train10k_fik50+color64+dsift':1488, 'tag400_train100k_fik50+color64+dsift':1488, 'tag400_train1m_fik50+color64+dsift':1488,
                  'd2sift1024':4096, 'd2sift7668webupv':7668, 'color64+d2sift7668webupv':7732,
                  'd2pcasift1929sp1113':7716, 'lbp':18, 'lbp1x3':54, 'color64l1':64, 'color64l1+lbp1x3':118, 'color64l1+lbp+lbp1x3':136,
                  'dlbp':1017, 'd2sift2048-1x1-1x3':7668, 'dsift-l1-1000':1634, 'color64-l1-1000':1634, 'dascaffeprob':1000, 'caffefc7':4096, 'caffefc7l1':4096,'dsiftl2':1024,
                  }

FEATURE_TO_MEAN   = {"color64":0.9291, "cslbp80":0.3343, "gist":1.7053, "dsift":0.0771}
FEATURE_TO_MEDIAN = {"color64":0.9140, "cslbp80":0.2930, "gist":1.6535, "dsift":0.0656}
FEATURE_TO_STD    = {"color64":0.1865, "cslbp80":0.1709, "gist":0.3836, "dsift":0.0404}
FEATURE_TO_L1MAX =  {"color64":12.5, "cslbp80":9.12, "gist":170.64, "dsift":2.0}

COLLECTION_TO_SIZE = {"flickr800k":815320, "flickr1m":1198818, "geoflickr1m":964849, "web13train":250000, 'flickr10m':int(1e7),
                      'tentagv10dev':1382290, 'flickr20m':int(2e7),'flickr20':19971,'msr2013train':int(1e6),
                      'mirflickr08dev':15000, 'mirflickr08test':10000,
                      'flickr81train':155545, 'train10k':int(1e4), 'train100k':int(1e5), 'train1m':1198818,
                      'social800k':815320, 'flickr81':259233, 'flickr55':81541}
                        
COLLECTION_TO_USERNUM = {'social800k':177871, 'tentagv10dev':42206, 'flickr81train':40202, 'flickr1m':347369, 'train1m':347369, 'flickr10m':941202, 'train10k':9249, 'train100k':68215}

COLLECTION_TO_CONCEPTSET = {'flickr20':'concepts20','flickr55':'concepts55', 'flickr81':'concepts81', 'flickr81train':'concepts81train', 'flickr81test':'concepts81test', 
'tentagv10dev':'conceptstentagv10dev', 'mirflickr08test':'conceptsmir18test','msr2013dev0':'conceptsweb15dev', 'ucsd18test':'conceptsucsd18test', 
'clef2010train':'conceptsclef2010train', 'clef2010val':'conceptsclef2010val', 'mirflickr08dev':'conceptsmir18dev','voc2008train':'conceptsvoc2008train',
'voc2008val':'conceptsvoc2008val', 'msr2013dev0':'conceptsmsr2013dev0', 'msr2013dev1':'conceptsmsr2013dev1', 'flickr81traintagged':'concepts81train', 'flickr81testtagged':'concepts81test',  'mirflickr08':'conceptsmir18', 'flickr81testtagged':'concepts81test'}

CAMERA_BRAND_SET = set(str.split('canon nikon sony samsung panasonic olympus fujifilm htc nokia kodak casio epson'))
PHOTO_TAG_SET = set(str.split('photography hdr photo photoshop aplusphoto picture platinumphoto diamondclassphotographer lens exposure flickrdiamond goldstaraward'))


