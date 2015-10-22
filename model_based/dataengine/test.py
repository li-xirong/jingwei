

from positiveengine import PositiveEngine
from negativeengine import NegativeEngine, WnNegativeEngine, CoNegativeEngine

if __name__ == "__main__":
    collection = "geoflickr1m"
    collection = 'tentagv10dev'
    
    rwpe = PositiveEngine(collection)
    rwne = NegativeEngine(collection)
    wnne = WnNegativeEngine(collection)
    cone = CoNegativeEngine(collection)
    
    for concept in str.split('2012 beach food london'): #str.split("car horse car-horse street animal "):
        print concept
        for engine in [rwne, wnne, cone]:
            print engine.getCount(concept)            
        print ''
            
            

