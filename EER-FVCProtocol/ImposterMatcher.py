import os
import skimage
import numpy as np
from scipy import spatial

featuresFolder = '../Outputs/'
savePath = '../EER-FVCProtocol/'
fileList = os.listdir(featuresFolder)
fileList.sort()
imposters = {}
# Following FVC fingerprint protocols      
for h in range(0,793,8):
    print '\n'                                   
    vectorA = np.load(featuresFolder +fileList[h])
    for i in range (h+8,793,8):
        vectorB = np.load(featuresFolder + fileList[i])
        vA = np.asarray(vectorA.item()['fc7'])
        vB = np.asarray(vectorB.item()['fc7'])
        score = 1.0-spatial.distance.cosine(vA,vB)
        print 'Matching :' + fileList[h] +' vs ' + fileList[i] + 'score: ' + str(score)             
        imposters [fileList[h] +' vs ' + fileList[i]] = score
    
np.save(savePath+'imposters.npy', imposters)
                        