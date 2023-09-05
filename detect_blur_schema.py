
from __future__ import division
import numpy as np
from pywt import dwt2
from PIL import Image
# import matplotlib.pyplot as plt
import os
import math

MINZERO = 0.0 # Should be set to zero
THRESHOLD = 28 # Adjust the threshold in accordance with the accuracy desired (28 original)
PATH = './FN/5a4ece2f8a8c4.jpg' #state the path to image location here

def HaarTL3(Image, threshold):# pass in the grayscale image
    '''
input:      Image: 2D numpy array
            threshold: set the threshold parameter to adjust the accuracy
returns:    two parameters: Per and BlurExtent
            In general, for a blurred image Per should be close to 0
            and BlurExtent quantifies blur content should be close to 1 for blurred image
            both parameters Per and Blur range frome 0.0 to 1.0
Summary:    The function employs Haar Wavelet Transform for an Image taken to 3 levels, and consequently analyses 
            presence of 4 types of edge points 
    '''
    
    cA, (cH, cV, cD) = dwt2(Image, 'haar')
    '''cA gives the Approximation coefficients at level 1 in accordance with Haar Wavelet Transform
        cH gives the  horizontal edge details 
        similarly cV gives the vertical edge details and cD edge details
    '''
    cA1, (cH1, cV1, cD1) = dwt2(cA, 'haar')
    '''
    The same transform used with input as the Approximations used from previous level to get corresponding edge 
    details: Horizontal, Vertical and Diagonal 
    '''
    cA2, (cH2, cV2, cD2) = dwt2(cA1, 'haar')
    '''
    The same transform used with input as the Approximations used from previous level to get corresponding edge 
    details: Horizontal, Vertical and Diagonal 
    
    ****IMPORTANT**** 
    Each Approximation Coefficient Matrix in a level has half the dimensions that of previous level
    Eg. If, cA.shape = 32 x 32 = cH.shape = cV.shape = cD.shape
        then, cA1.shape = cH1.shape = cV1.shape = cD1.shape = 16 x 16
    '''
    Emap = np.sqrt(cH*cH + cV*cV + cD*cD)
    '''
    gives the intensity of edge approximation in level 1
    '''

    Emax = pool(Emap, K = 8, L = 8)
    '''
    Partition the Emax matrix by 8 x 8 kernel to perform max pooling inorder to downsample the matrix
    '''
    d, r, c = Emax.shape
    if d == 1:
        # print(Emax.shape)
        Emax = Emax.reshape((r,c))
        # print(Emax.shape)
    else:
        print("Dimension ERror")
    Emap1 = np.sqrt(cH1*cH1 + cV1*cV1 + cD1*cD1)

    '''
    gives the intensity of edge approximation in level 2
    '''
    Emax1 = pool(Emap1, K = 4, L = 4)
    '''
    Partition the Emax matrix by 8 x 8 kernel to perform max pooling inorder to downsample the matrix
    '''
    d1, r1, c1 = Emax1.shape

    if d1 == 1:
        # print(Emax1.shape)
        Emax1 = Emax1.reshape((r1,c1))
        # print(Emax1.shape)
    else:
        print("Dimension ERror")


    Emap2 = np.sqrt(cH2*cH2 + cV2*cV2 + cD2*cD2)
    '''
    gives the intensity of edge approximation in level 3
    '''

    Emax2 = pool(Emap2, K = 2, L = 2)
    '''
    Partition the Emax matrix by 8 x 8 kernel to perform max pooling inorder to downsample the matrix
    '''
    d2, r2, c2 = Emax2.shape
    if d2 == 1:
        # print(Emax2.shape)
        Emax2 = Emax2.reshape((r2,c2))
        # print(Emax2.shape)
    else:
        print("Dimension ERror")
    
    assert( (Emax.shape == Emax1.shape) * (Emax1.shape == Emax2.shape) )
    # print Emax > threshold
    R1 = np.logical_or(Emax > threshold, Emax1 > threshold, Emax2 > threshold) # Finds all the edge points
    # print('R1: ',R1, 'R1.shape: ', R1.shape)
    R5 = np.logical_and(R1, Emax < threshold)
    
    Nbrg = np.sum(R5)
    Nedge = np.sum(R1)
    
    
    A = Emax[R1] > Emax1[R1]
    B = Emax1[R1] > Emax2[R1]
    # print('A: ', A, 'A.shape: ', A.shape)
    R2 = np.logical_and(A, B) #Finds all Dirac A step points
    # print('R2: ',R2, 'R2.shape: ', R2.shape)
    Nda  = np.sum(R2)
    R3 = np.logical_and(Emax[R1] < Emax1[R1], Emax1[R1] < Emax2[R1]) #, Emax[R1] < threshold) #Roof or Gstructure
    # print 'R3: ', R3, 'R3 shape: ',R3.shape
    R4 = np.logical_and(Emax[R1] < Emax1[R1], Emax1[R1] > Emax2[R1]) #, Emax[R1] < threshold)# Roof structure
    # print 'R4: ', R4, 'R4 shape: ',R4.shape
    Nrg  = np.sum(R3) + np.sum(R4)

    R31 =np.logical_or( np.logical_and(Emax[R1] < Emax1[R1], Emax1[R1] < Emax2[R1]), np.logical_and(Emax[R1] < Emax1[R1], Emax1[R1] > Emax2[R1]))
    # Nrg = np.sum(R31)
    if Nedge == 0:
        Per = 0
    else:
        Per = Nda/Nedge
    if Nrg == 0:
        BlurExtent = 1
    else:
        BlurExtent = Nbrg/Nrg

    return Per, BlurExtent

def pool(mat, K, L):
    '''
    input:    Mat on which Max pooling is to be performed
              (K,L) kernel dimensions for max pooling
    output:   returns max pooled matrix
    Summary:  Performs Max pooling on a matrix with a kernel size (K,L) and stride (K,L) along corresponding dimensions
    '''
    S = []
    M, N = mat.shape
    MK = M // K
    NL = N // L
    S.append(mat[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3)))
    return np.array(S)

def Classify(PATH, threshold):
    Ima = Image.open(PATH).convert('L')
    Ima = np.array(Ima)
    h, w = Ima.shape
    '''
    an approximation's been made here to throw away atmost 7 pixels along each dimensions 
    '''

    H, W = h//8 * 8, w//8 * 8
    Ima = Ima[ :H,:W ]
    Per, BlurEx = HaarTL3(Ima, 28)
    if Per == 0.0 and BlurEx > 0.75:
        return True
    else:
        return False

print(Classify(PATH, THRESHOLD))
fin.py
Displaying fin.py.
