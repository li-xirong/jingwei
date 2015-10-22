import os, sys
import subprocess

testCollection=sys.argv[1]
concepts=sys.argv[2]
resultpath="/home/urix/surveydbdas"

possible_settings = itertools.product([2**-6, 2**-4, 2**-2, 2**0, 2**1, 2**2, 2**4], [2**-10, 2**-8, 2**-6, 2**-4, 2**-2, 2**0])

for lambda1, lambda2 in possible_settings:
    result = subprocess.call("python robustpca/robustpca.py --lambda1 %f --lambda2 %f --outputonlytest 1" % (lambda1, lambda2), shell=True)