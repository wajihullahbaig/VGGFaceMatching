import os
import skimage
import numpy as np
from scipy import spatial

featuresFolder = '../Outputs/'
savePath = '../EER-FVCProtocol/'
fileList = os.listdir(featuresFolder)
fileList.sort()
genuines = {}
# Following FVC fingerprint protocols      
for h in range(0,793,8):
    print '\n'
    fileA = fileList[h]
    for i in range (1,9):
        fileName, fileExtension = os.path.splitext(fileA)
        splits = fileName.rsplit('_',1) 
        fileNameA =  splits[0]+'_000'+str(i)+fileExtension                                 
        vectorA = np.load(featuresFolder +fileNameA)
        for j in range (i+1,9):    
            fileNameB = splits[0] + '_000' + str(j) + fileExtension 
            vectorB = np.load(featuresFolder + fileNameB)
            vA = np.asarray(vectorA.item()['fc7'])
            vB = np.asarray(vectorB.item()['fc7'])
            score = 1.0-spatial.distance.cosine(vA,vB)
            print 'Matching :' + fileNameA +' vs ' + fileNameB + 'score: ' + str(score)             
            genuines [fileNameA +' vs ' + fileNameB] = score
    
np.save(savePath+'genuines.npy', genuines)
                        