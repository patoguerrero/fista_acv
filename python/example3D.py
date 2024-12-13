# example of fista_acv
# applied to few-view computed tomography
# Patricio Guerrero
# KU Leuven
# patricio.guerrero@kuleuven.be


import torch
import numpy
import matplotlib.pyplot as plt
import fista_acv
from time import time



# -------------
# 1. read data
# -------------



t0 = time()

sod =     # source-object distance
sdd =     # source-detecotr distance
rows =    # rows in detector
pixels =  # columns in detector
pixel =   # pixel size (in detector)
views =   # number of rotations 
rot_step =  # rotation step 
voxel = pixel * sod/sdd  # voxel size (in reconstructed volume)


print('magnification', sdd/sod)

# ground truth reconstruction 
GT =     # shape  : (slices, pixels, pixels)


# few-view CT data  
dataFV =   # shape : (columns, views, rows), for tomosipo
print('dataFV', dataFV.shape, dataFV.dtype)

print('time read', time()-t0)



# -----------------------
# 3. parameter estimation
# -----------------------


# 3.1 geometrical missalignments
# if not available, can be computed with github.com/patoguerrero/alignCT/

det_x = 0       # (horizontal shift, mm)
eta = 0         # (detector in-plane tilt, degrees)
print('x(mm), eta(deg) ', det_x, eta)

# 3.2 cone beam geometry

vectors = fista_acv.vectors_tomosipo(pixel, voxel, sod, sdd, rot_step, views, det_x, 0, eta)
_, views, _ = dataFV.shape

# 3.3 operator norm

tq = time()
slices = 16
fpq, bpq = fista_acv.ct_operators(pixels, pixels, slices, min(slices*2, rows), vectors, 0, 1)
dev = torch.device('cuda:0')
q = fista_acv.power_method(fpq, bpq, pixels, slices, dev)
print('operator norm', q, 'time', time() - tq)

# 3.4 NGD with FISTA-aCV

tNGD = time()
slices = 2
n_FI = 140        # fista iterations
GT = GT[50 - slices//2: 50 + slices//2]
lmbd = fista_acv.hyperparameter_NGD(dataFV, GT, vectors, slices, q, n_FI, sod, sdd, voxel)
print('lambda', lmbd)
print('time NGD', time() - tNGD)
#lmbd = 0.00035670243


# -------------------------------
# 4. TV reconstruction (FISTA-CV)
# -------------------------------



slices = 20
FOV = 1400

trec = time()
dev, dataFV  = fista_acv.prepare_data(dataFV, slices, pixels, FOV, 0, sod, voxel)
rec = fista_acv.fista_cv3D(dataFV, FOV, vectors, slices, lmbd, q, n_FI, 0, sdd/sod, dev)
trec = time() - trec

print('time fista-CV', trec)
    
plt.imshow(rec[:, :, slices//2])
plt.show()
