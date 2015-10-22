

The ``visualize`` folder contains 1) [pickle\_to_imagerank.py](pickle_to_imagerank.py) to convert tag relevance learning results stored in pickle files to image ranks per test tag, and 2) [webdemo](webdemo) to show the image ranks in html format.  

# Dependencies 

+ [web.py](http://webpy.org/): sudo easy_install web.py


# Get started 


1.  see [example-scripts.sh](example-scripts.sh) for generating image ranks given a list of pickle files
2.  go into [webdemo](webdemo), modify [config.json](webdemo/config.json) accordingly
3.  start a web server by ```python main.py 9090```, which listens to port 9090.

For a given test tag, say `rainbow`, 

+ Images ranked in terms of tag relevance: [http://localhost:9090/?query=rainbow&tagrel=1](http://localhost:9090/?query=rainbow&tagrel=1)
+ Images labeled with the test tag: [http://localhost:9090/?query=rainbow&tagrel=0](http://localhost:9090/?query=rainbow&tagrel=0)

