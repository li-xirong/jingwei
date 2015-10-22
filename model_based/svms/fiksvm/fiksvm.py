#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
import sys
import os

import platform

LIB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')

if sys.platform.startswith('linux'):
    filename = os.path.join(LIB_PATH, 'linux', 'libfiksvm.so.1')
    libfiksvm = CDLL(filename)
elif sys.platform.startswith('darwin'):
    filename = os.path.join(LIB_PATH, 'mac', 'libfiksvm.so.1')
    libfiksvm = CDLL(filename)
else:
    libdir = 'win32' if platform.architecture()[0] == '32bit' else 'win64'
    filename = os.path.join(LIB_PATH, libdir,'libfiksvm.dll')
    libfiksvm = cdll.LoadLibrary(filename)
    

from svm import svm_model, toPyModel

def fillprototype(f, restype, argtypes): 
    f.restype = restype
    f.argtypes = argtypes


class fiksvm_approx_model(Structure):
    def __init__(self):
        self.__createfrom__ = 'python'

    def __del__(self):
        # free memory created by C to avoid memory leak
        if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
             libfiksvm.fiksvm_free_and_destroy_model(pointer(self))

    def get_nr_svs(self):
        return libfiksvm.fiksvm_get_nr_svs(self)

    def get_feat_dim(self):
        return libfiksvm.fiksvm_get_feat_dim(self)

    def predict(self, x):
        assert(len(x) == self.get_feat_dim())
        c_x = (c_double * len(x))()
        for dim in range(len(x)):
            c_x[dim] = x[dim]

        return libfiksvm.fiksvm_predict(self, c_x)  
    
    def predict_probability(self, x):
        assert(len(x) == self.get_feat_dim())
        c_x = (c_double * len(x))()
        for dim in range(len(x)):
            c_x[dim] = x[dim]

        return libfiksvm.fiksvm_predict_probability(self, c_x)  
        

    def get_probAB(self):
        return (libfiksvm.fiksvm_get_probA(self), libfiksvm.fiksvm_get_probB(self))
   
    def set_probAB(self, probA, probB):
        libfiksvm.fiksvm_set_probAB(self, probA, probB)


    def add_rawsvm(self, new_model, w1, w2):
        assert(w1 >= -1e-8 and w1 <= 1+1e-8)
        assert(w2 >= -1e-8 and w2 <= 1+1e-8)
        return libfiksvm.add_new_hikmodel(self, pointer(new_model), w1, w2)

    def add_fastsvm(self, new_model, w1, w2):
        assert(w1 >= -1e-8 and w1 <= 1+1e-8)
        assert(w2 >= -1e-8 and w2 <= 1+1e-8)
        return libfiksvm.add_new_fikmodel(self, pointer(new_model), w1, w2)


'''
def toPyModel(model_ptr):
	"""
	toPyModel(model_ptr) -> fiksvm_approx_model

	Convert a ctypes POINTER(fiksvm_approx_model) to a Python fiksvm_approx_model
	"""
	if bool(model_ptr) == False:
		raise ValueError("Null pointer")
	m = model_ptr.contents
	m.__createfrom__ = 'C'
	return m
'''

fillprototype(libfiksvm.create_fiksvm_approx_model, POINTER(fiksvm_approx_model), [POINTER(POINTER(svm_model)), c_int, POINTER(c_double), c_int, POINTER(c_double), POINTER(c_double), c_int])
fillprototype(libfiksvm.fiksvm_get_nr_svs, c_int, [POINTER(fiksvm_approx_model)])
fillprototype(libfiksvm.fiksvm_get_feat_dim, c_int, [POINTER(fiksvm_approx_model)])
fillprototype(libfiksvm.fiksvm_predict, c_double, [POINTER(fiksvm_approx_model), POINTER(c_double)])
fillprototype(libfiksvm.fiksvm_predict_probability, c_double, [POINTER(fiksvm_approx_model), POINTER(c_double)])
#fillprototype(libfiksvm.fiksvm_free_model_content, None, [POINTER(fiksvm_approx_model)])
fillprototype(libfiksvm.fiksvm_free_and_destroy_model, None, [POINTER(POINTER(fiksvm_approx_model))])

fillprototype(libfiksvm.fiksvm_save_model, c_int, [c_char_p, POINTER(fiksvm_approx_model)])
fillprototype(libfiksvm.fiksvm_load_model, POINTER(fiksvm_approx_model), [c_char_p])

fillprototype(libfiksvm.fiksvm_get_probA, c_double, [POINTER(fiksvm_approx_model)])
fillprototype(libfiksvm.fiksvm_get_probB, c_double, [POINTER(fiksvm_approx_model)])
fillprototype(libfiksvm.fiksvm_set_probAB, None, [POINTER(fiksvm_approx_model), c_double, c_double])

fillprototype(libfiksvm.add_new_hikmodel, c_int, [POINTER(fiksvm_approx_model), POINTER(svm_model), c_double, c_double])
fillprototype(libfiksvm.add_new_fikmodel, c_int, [POINTER(fiksvm_approx_model), POINTER(fiksvm_approx_model), c_double, c_double])


def svm_to_fiksvm(svm_models, weights, feat_dim, params):
    num_models = len(weights)
    min_vals = params['min_vals']
    max_vals = params['max_vals']
    nr_bins = params['nr_bins']

    c_weights = (c_double * num_models)()
    model_ptr_ptr = (POINTER(svm_model) * num_models)()

    for t in range(num_models):
        #c_weights[t] = c_double(1.0/num_models)
        c_weights[t] = c_double(weights[t])
        model_ptr_ptr[t] = pointer(svm_models[t])
        #print t, svm_models[t].get_nr_class()

    c_min_vals = (c_double * feat_dim)()
    c_max_vals = (c_double * feat_dim)()
    for i in range(feat_dim):
        c_min_vals[i] = c_double(min_vals[i])
        c_max_vals[i] = c_double(max_vals[i])
       
    new_model = libfiksvm.create_fiksvm_approx_model(model_ptr_ptr, num_models, c_weights, feat_dim, c_min_vals, c_max_vals, nr_bins)
    del model_ptr_ptr
    del c_weights
    del c_min_vals
    del c_max_vals

    if not new_model:
        print("can't do svm_to_fiksvm")
        return None

    return toPyModel(new_model)
    

def fiksvm_load_model(model_file_name):
    """
    fiksvm_load_model(model_file_name) -> model

    Load a fik_approx_model from model_file_name and return.
    """
    model = libfiksvm.fiksvm_load_model(model_file_name)

    if not model:
        print("can't open model file %s" % model_file_name)
        return None

    model = toPyModel(model)
    return model


def fiksvm_save_model(model_file_name, model):
    """
    fiksvm_save_model(model_file_name, model) -> None

    Save a fik_approx_model to the file model_file_name.
    """

    status = libfiksvm.fiksvm_save_model(model_file_name, model)

    return status


