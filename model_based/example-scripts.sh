overwrite=0
trainCollection=train1m
socialAnnotationName=conceptstoysocial.txt
startAnnotationName=conceptstoysocial.pqtagrel100.random100.0.txt
feature=dascaffefc7
modelName=fastlinear 
#modelName=fiksvm

python negbp.py $trainCollection $socialAnnotationName $startAnnotationName $feature $modelName --overwrite $overwrite

