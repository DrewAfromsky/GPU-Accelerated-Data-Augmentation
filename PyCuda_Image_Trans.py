# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################


from __future__ import division
import numpy as np
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import matplotlib.image as mpimg
from pycuda import gpuarray, compiler, tools, cumath
from numpy import linalg as la
import math
from scipy import signal
import cv2
import os


class Transpose:
    def __init__(self):
        # kernel
        self.kernel_code = """
			#include <stdio.h>

			__global__ void img_trans(float *T, const float *mat, int P, int L)
			{
                // 2-D thread ID assuming more than one block will be executed
                int index_x = threadIdx.x + blockIdx.x * blockDim.x; // ROWS
                int index_y = threadIdx.y + blockIdx.y * blockDim.y; // COLUMNS

                // index of the input array; L=columns
                int el = index_x * L + index_y;

                // index of the output array (transposed); P = rows
                int out = index_y * P + index_x;
                if(index_x < P && index_y < L){
                    T[out] = mat[el];
                }

            }
		"""
        
        self.mod = compiler.SourceModule(self.kernel_code)

    def image_transpose(self, A):
        # # Device memory allocation
        self.A_d = gpuarray.to_gpu(A)
        self.output_d = gpuarray.empty((A.shape[0], A.shape[1]), np.float32)    

        # create CUDA Event to measure time
        start = cuda.Event()
        end = cuda.Event()

        # function call
        func = self.mod.get_function('img_trans')
        start.record()
        start_ = time.time()
        func(self.output_d, self.A_d, np.int32(A.shape[0]), np.int32(A.shape[1]), block=(32, 32, 1), grid = (np.int(np.ceil(float(A.shape[0])/32)), np.int(np.ceil(float(A.shape[1])/32)),1)) # In CUDA block=(x,y,z), grid=(x,y,z)
        end_ = time.time()
        end.record()

		# CUDA Event synchronize
        end.synchronize()

        output_gpu = self.output_d.get()
        kernel_time = end_-start_

        return output_gpu, kernel_time    

if __name__ == '__main__':
    import cv2
    import glob
    naive_times=[]

    # Iterate through all 50,000 images
    file_list = []
    sheep_list = []
    
    # for file in glob.glob("*.png"): # .png for CIFAR; .jpg for pathology
    for file in glob.glob("42259_horse.png"):
        file_list.append(file)
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        A=np.float32(gray)
        sheep_list.append(A.shape)
        module = Transpose()
        cu_output_naive, t = module.image_transpose(A)
        naive_times.append(t)


    print(sheep_list[0])
    print(len(naive_times))
    
    counter = 0
    for nt in range(len(naive_times)):
        counter += naive_times[nt]
    print(counter)      
    
# MAKE_PLOT = True
# if MAKE_PLOT:
#     plt.figure()
#     plt.imsave("/home/daa2162/CUDA_42257_frog_TRANS.png", cu_output_naive, cmap="gray")
    # plt.imsave("/home/daa2162/CUDA_Transposed_OG.png", A, cmap="gray")
    # plt.imsave("/home/daa2162/CUDA_35800_bird.png", cu_output_naive, cmap="gray")
    # plt.imsave("/home/daa2162/CUDA_35800_bird_OG.png", A, cmap="gray")    