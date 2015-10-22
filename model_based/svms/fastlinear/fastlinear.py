#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
import sys
import os
import platform

LIB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')

if sys.platform.startswith('linux'):
    filename = os.path.join(LIB_PATH, 'linux', 'libfastlinear.so.1')
    libfastlinear = CDLL(filename)
elif sys.platform.startswith('darwin'):
    filename = os.path.join(LIB_PATH, 'mac', 'libfastlinear.so.1')
    libfastlinear = CDLL(filename)
else:
    libdir = 'win32' if platform.architecture()[0] == '32bit' else 'win64'
    filename = os.path.join(LIB_PATH, libdir,'libfastlinear.dll')
    libfastlinear = cdll.LoadLibrary(filename)

from liblinear193.python.liblinear import model as liblinear_model,toPyModel,fillprototype


class fastlinear_model(Structure):
    def __init__(self):
        self.__createfrom__ = 'python'

    def __del__(self):
        # free memory created by C to avoid memory leak
        if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
             libfastlinear.fastlinear_free_and_destroy_model(pointer(self))

    def get_feat_dim(self):
        return libfastlinear.fastlinear_get_feat_dim(self)

    def predict(self, x):
        assert(len(x) == self.get_feat_dim()), ('input %d, required %d' % (len(x), self.get_feat_dim()))
        c_x = (c_double * len(x))()
        for dim in range(len(x)):
            c_x[dim] = x[dim]

        return libfastlinear.fastlinear_predict(self, c_x)  


    def predict_probability(self, x):
        assert(len(x) == self.get_feat_dim()), ('input %d, required %d' % (len(x), self.get_feat_dim()))
        c_x = (c_double * len(x))()
        for dim in range(len(x)):
            c_x[dim] = x[dim]

        return libfastlinear.fastlinear_predict_probability(self, c_x)

        
    def get_probAB(self):
        return (libfastlinear.fastlinear_get_probA(self), libfastlinear.fastlinear_get_probB(self))
   
    def set_probAB(self, probA, probB):
        libfastlinear.fastlinear_set_probAB(self, probA, probB)

    def get_w(self):
        return libfastlinear.fastlinear_get_w(self)


    def add_rawsvm(self, new_model, w1, w2):
        assert(w1 >= -1e-8 and w1 <= 1+1e-8)
        assert(w2 >= -1e-8 and w2 <= 1+1e-8)
        return libfastlinear.add_new_liblinear(self, pointer(new_model), w1, w2)

    def add_fastsvm(self, new_model, w1, w2):
        assert(w1 >= -1e-8 and w1 <= 1+1e-8)
        assert(w2 >= -1e-8 and w2 <= 1+1e-8)
        return libfastlinear.add_new_fastlinear(self, pointer(new_model), w1, w2)



fillprototype(libfastlinear.create_fastlinear_model, POINTER(fastlinear_model), [POINTER(POINTER(liblinear_model)), POINTER(c_double), c_int, c_int])
fillprototype(libfastlinear.add_new_liblinear, c_int, [POINTER(fastlinear_model), POINTER(liblinear_model), c_double, c_double])
fillprototype(libfastlinear.add_new_fastlinear, c_int, [POINTER(fastlinear_model), POINTER(fastlinear_model), c_double, c_double])


fillprototype(libfastlinear.fastlinear_get_feat_dim, c_int, [POINTER(fastlinear_model)])
fillprototype(libfastlinear.fastlinear_predict, c_double, [POINTER(fastlinear_model), POINTER(c_double)])
fillprototype(libfastlinear.fastlinear_predict_probability, c_double, [POINTER(fastlinear_model), POINTER(c_double)])
fillprototype(libfastlinear.fastlinear_free_and_destroy_model, None, [POINTER(POINTER(fastlinear_model))])

fillprototype(libfastlinear.fastlinear_save_model, c_int, [c_char_p, POINTER(fastlinear_model)])
fillprototype(libfastlinear.fastlinear_load_model, POINTER(fastlinear_model), [c_char_p])

fillprototype(libfastlinear.fastlinear_get_probA, c_double, [POINTER(fastlinear_model)])
fillprototype(libfastlinear.fastlinear_get_probB, c_double, [POINTER(fastlinear_model)])
fillprototype(libfastlinear.fastlinear_set_probAB, None, [POINTER(fastlinear_model), c_double, c_double])
fillprototype(libfastlinear.fastlinear_get_w, c_double, [POINTER(fastlinear_model)])


def liblinear_to_fastlinear(liblinear_models, weights, feat_dim, params=None):
    num_models = len(weights)
    c_weights = (c_double * num_models)()
    model_ptr_ptr = (POINTER(liblinear_model) * num_models)()

    for t in range(num_models):
        c_weights[t] = c_double(weights[t])
        model_ptr_ptr[t] = pointer(liblinear_models[t])
    
    new_model = libfastlinear.create_fastlinear_model(model_ptr_ptr, c_weights, num_models, feat_dim)
    del model_ptr_ptr
    del c_weights

    if not new_model:
        print("can't do liblinear_to_fastlinear")
        return None

    return toPyModel(new_model)
    

def fastlinear_load_model(model_file_name):
    """
    fastlinear_load_model(model_file_name) -> model

    Load a fastlinear_model from model_file_name and return.
    """
    model = libfastlinear.fastlinear_load_model(model_file_name)

    if not model:
        print("can't open model file %s" % model_file_name)
        return None

    model = toPyModel(model)
    return model


def fastlinear_save_model(model_file_name, model):
    """
    fastlinear_save_model(model_file_name, model) -> None

    Save a fastlinear_model to the file model_file_name.
    """

    status = libfastlinear.fastlinear_save_model(model_file_name, model)

    return status


