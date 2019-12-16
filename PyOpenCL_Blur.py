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

class Blur:
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


        # Write the kernel code
        self.kernel_code = """
        __kernel void image_blur(__global float * in, 
        __global float * out, 
        int w, 
        int h)
            {

            int Col = get_global_id(0); 
            int Row = get_global_id(1); 

            if (Col < w && Row < h) {
                int pixVal = 0;
                int pixels = 0;

                // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
                for (int blurRow = -%(BLUR_SIZE)s; blurRow < %(BLUR_SIZE)s + 1; ++blurRow) {
                    for (int blurCol = -%(BLUR_SIZE)s; blurCol < %(BLUR_SIZE)s + 1; ++blurCol)
                {
                        int curRow = Row + blurRow;
                        int curCol = Col + blurCol;
                        // Verifty we have a valid image pixel
                        if (curRow > -1 && curRow < h && curCol > -1 && curCol < w){
                            pixVal += in[curRow * w + curCol];
                            pixels++; // Keep track of # of pixels in the avg
                        }
                    }
                }
            // Write our new pixel value out
            out[Row * w + Col] = (unsigned char) (pixVal / pixels);
            }            
        }
        """
        # For a 3x3 convolutional kernel, the BLUR_SIZE = 1; 
        # value of BLUR_SIZE is set such that 2*BLUR_SIZE gives us 
        # the number of pixels on each side of the patch of the image
        self.kernel_code_blur = self.kernel_code % {
            'BLUR_SIZE': 5, # 5 for pathology; 1 for CIFAR
            }
        self.prg_naive = cl.Program(self.ctx, self.kernel_code_blur).build()

    def image_avg_blur(self, A):
        # INPUTS:
        # A --> matrix of the image    
        # kernel --> the filter
        # start1= time.time()
        # Transfer data to device  
        self.A_d = cl.array.to_device(self.queue,A)
        # end1=time.time()
        # k2 = end1-start1
        # print(k2)
        # self.kernel_d = gpuarray.to_gpu(kernel)
        # self.output_d = gpuarray.empty((A.shape[0], A.shape[1]), np.float32)    
        self.output_d = cl.array.empty(self.queue, (A.shape[0], A.shape[1]), np.float32)
        
        # Call kernel function
        func_naive = self.prg_naive.image_blur
        
        # if (A.shape[0] > A.shape[1]):
        m_A = A.shape[0]
        # else:
        #     m_A = A.shape[1]

        # Measure time
        evt = func_naive(self.queue,(np.int(m_A),np.int(m_A)), None, 
        self.A_d.data, 
        self.output_d.data, 
        np.int32(A.shape[1]), 
        np.int32(A.shape[0]))
        evt.wait()
        time_ = 1e-9 * (evt.profile.end - evt.profile.start) #this is the recommended way to record OpenCL running time

        output_gpu = self.output_d.get()
        kernel_time = time_

        # Return the result and the kernel execution time
        return output_gpu, kernel_time  

if __name__ == '__main__':
    import cv2
    import glob     

    print(os.getcwd())
    
    naive_times=[]

    # Iterate through all 50,000 images
    file_list = []

    sheep_list = []
    
    # for file in glob.glob("42257_frog.png"): # 42259_horse.png
    for file in glob.glob("42259_horse.png"): # 42259_horse.png

        file_list.append(file)
        img = cv2.imread(file)
    # for fl in range(len(file_list)):
        # img = cv2.imread(file_list[fl])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        A=np.float32(gray)

        sheep_list.append(A.shape)

    # Create instance for OpenCL
        module = Blur()

    # Record times
        cl_output_naive, t = module.image_avg_blur(A)
        # print(t)
        naive_times.append(t)
    
    # print(sheep_list[0])
    print(len(naive_times))
    
    counter = 0
    for nt in range(len(naive_times)):
        counter += naive_times[nt]
    print(counter)

        # Compute Tiled GPU Convolution
        # cu_output_tiled, t_tiled = module.tiled_conv2d_gpu(A, kernel_flip)
        # tiled_times.append(t_tiled)
        # Compute Serial Convolution
        # cpu_output, t_cpu = module.conv2d_cpu(A, kernel)
        # serial_times.append(t_cpu)
    # img2 = cv2.imread("29291_deer.png")
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # A2=np.float32(gray2)

 # Plot times
    # MAKE_PLOT = True
    # if MAKE_PLOT:
    #     plt.figure()
        # plt.imsave("/home/daa2162/Blurred_Image_CIFAR.jpg", cu_output_naive, cmap="gray")
        # plt.imsave("/home/daa2162/Blurred_Image_PATH_LAST_CL.jpg", cl_output_naive, cmap="gray")
       
        # plt.imsave("/home/daa2162/42259_horse_CL_BLUR.png", cl_output_naive, cmap="gray")
        # plt.imsave("/home/daa2162/LAST_CL.png", cl_output_naive, cmap="gray")

        # plt.imsave("/home/daa2162/42257_frog_OPENCL_BLUR.png", cl_output_naive, cmap="gray")
        # plt.imsave("/home/daa2162/42257_frog_OPENCL.png", A2, cmap="gray")



        # plt.gcf()
        # plt.plot(msize, py_times,'r', label="Python")
    # plt.plot(msize, cu_times_tiled,'b', label="CUDA TILED")
    # plt.plot(msize, cu_times_naive,'g', label="CUDA NAIVE")
    # plt.legend(loc='upper left')
    # plt.title('2D Convolution')
    # plt.xlabel('size of array')
    # plt.ylabel('output coding times(sec)')
    # plt.gca().set_xlim((min(msize), max(msize)))
    # plt.savefig('plots_pycuda_2D_conv_SMALL.png')
    

   
