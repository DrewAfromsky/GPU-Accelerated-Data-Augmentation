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
import matplotlib.image as mpimg
from numpy import linalg as la
import math
from scipy import signal
import cv2
import os
import pyopencl as cl
import pyopencl.array
from scipy import signal

class Transpose:
    def __init__(self):

        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # Set up a command queue:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)    

        self.kernel_code = """
			__kernel void img_trans(__global float *T, __global const float *mat, int P, int L)
			{
                // 2-D thread ID assuming more than one block will be executed
                int Row = get_global_id(0); 
                int Col = get_global_id(1); 

                // index of the input array; L=columns
                int el = Row * L + Col;

                // index of the output array (transposed)
                int out = Col * P + Row;
                if(Row < P && Col < L){
                    T[out] = mat[el];
                }

            }
		"""

        self.prg_naive = cl.Program(self.ctx, self.kernel_code).build()

    def image_trans(self, A):

        # Device memory allocation
        self.A_d = cl.array.to_device(self.queue,A)
        self.output_d = cl.array.empty(self.queue, (A.shape[0], A.shape[1]), np.float32)

        # Call kernel function
        func_naive = self.prg_naive.img_trans
        m_A = A.shape[0]


        # Measure time
        evt = func_naive(self.queue,(np.int(m_A),np.int(m_A)), None, 
        self.output_d.data, 
        self.A_d.data, 
        np.int32(A.shape[0]), 
        np.int32(A.shape[1]))
        evt.wait()
        time_ = 1e-9 * (evt.profile.end - evt.profile.start)

        output_gpu = self.output_d.get()
        kernel_time = time_

        # Return the result and the kernel execution time
        return output_gpu, kernel_time  

if __name__ == '__main__':
    import cv2
    import glob

    naive_times=[]

    # Iterate through all 50,000 images
    file_list = []
    sheep_list = []
    
    # for file in glob.glob("*.png"): # .png for CIFAR; .jpg for pathology
    for file in glob.glob("ZT76_39_A_4_3.jpg"):
        file_list.append(file)
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        A=np.float32(gray)
        sheep_list.append(A.shape)
        module = Transpose()
        cu_output_naive, t = module.image_trans(A)
        naive_times.append(t)
        
    print(sheep_list[0])
    print(len(naive_times))
    
    counter = 0
    for nt in range(len(naive_times)):
        counter += naive_times[nt]
    print(counter)      
    
MAKE_PLOT = True
if MAKE_PLOT:
    plt.figure()
    # plt.imsave("/home/daa2162/CL_42257_frog_TRANS.png", cu_output_naive, cmap="gray")
    # plt.imsave("/home/daa2162/OpenCL_Transposed_OG.jpg", A, cmap="gray")

