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
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray, compiler, tools, cumath
from numpy import linalg as la
import math
from scipy import signal
import cv2
import os
# import glob

class Blur:
    def __init__(self):
    
        # Write the kernel code
        self.kernel_code = """
        __global__ void image_blur(float * in, float * out, int w, int h)
            {
            int Col = blockIdx.x * blockDim.x + threadIdx.x;
            int Row = blockIdx.y * blockDim.y + threadIdx.y;

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
            'BLUR_SIZE': 5, # 1 for CIFAR; 5 for pathology
            }
        self.mod = compiler.SourceModule(self.kernel_code_blur)

            # Write your tiled 2D Convolution kernel here
        self.kernel_code_tiled = """
            __global__ void blur_tiled(float * in, float * mask, float * out, const int mask_width, int w, int h) {
                
                int tx = threadIdx.x;
                int ty = threadIdx.y;

                int row_o = blockIdx.y*%(TILE_SIZE)s + ty;
                int col_o = blockIdx.x*%(TILE_SIZE)s + tx;

                int row_i = row_o - mask_width/2;
                int col_i = col_o - mask_width/2;

                __shared__ float in_s[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

                if ((row_i >= 0) && (row_i < h) && (col_i >= 0) && (col_i < w)) {
                    in_s[ty][tx] = in[row_i * w + col_i];    
                } else {
                    in_s[ty][tx] = 0.0f;
                }
        
                __syncthreads(); 
        
                
                if (ty < %(TILE_SIZE)s && tx < %(TILE_SIZE)s) {
                    float out_VAL = 0.0f;
                    for (int i = 0; i < mask_width; i++) {
                        for (int j = 0; j < mask_width; j++) {
                            out_VAL += mask[i * mask_width + j] * in_s[i + ty][j + tx];
                        }
                    }

                    __syncthreads();  

                    if (row_o < h && col_o < w) {
                        out[row_o * w + col_o] = out_VAL;
                    }
                }
            }

        """
        self.kernel_code_T = self.kernel_code_tiled % {
                'TILE_SIZE': 28, #28
                'BLOCK_SIZE': 32,}
        self.mod_tiled = compiler.SourceModule(self.kernel_code_T)

    def image_avg_blur(self, A):
        # INPUTS:
        # A --> matrix of the image    
        # kernel --> the filter
        self.A_d = gpuarray.to_gpu(A)
        # self.kernel_d = gpuarray.to_gpu(kernel)
        self.output_d = gpuarray.empty((A.shape[0], A.shape[1]), np.float32)    

        # Compute block and grid size
        grid_dim_x = np.ceil(np.float32(self.A_d.shape[0]/32))
        grid_dim_y = np.ceil(np.float32(self.A_d.shape[1]/32))

        # create CUDA Event to measure time
        start = cuda.Event()
        end = cuda.Event()

        # Call kernel function
        func_naive = self.mod.get_function('image_blur')

        # Measure time
        start.record()
        start_ = time.time()
        func_naive(self.A_d, self.output_d, np.int32(A.shape[1]), np.int32(A.shape[0]), 
        block = (32, 32, 1), grid = (np.int(grid_dim_y),np.int(grid_dim_x),1))
        end_ = time.time()
        end.record()

        # CUDA Event synchronize
        end.synchronize()

        output_gpu = self.output_d.get()
        kernel_time = end_-start_

        # Return the result and the kernel execution time
        return output_gpu, kernel_time   

    def image_avg_blur_tiled(self, A, kernel):
        # INPUTS:
        # A --> matrix of the image    
        # kernel --> the filter
        self.A_d = gpuarray.to_gpu(A)
        self.kernel_d = gpuarray.to_gpu(kernel)
        self.output_d = gpuarray.empty((A.shape[0], A.shape[1]), np.float32)    

        # Compute block and grid size
        grid_dim_x = np.ceil(np.float32(self.A_d.shape[0]/32))
        grid_dim_y = np.ceil(np.float32(self.A_d.shape[1]/32))

        # create CUDA Event to measure time
        start = cuda.Event()
        end = cuda.Event()

        # Call kernel function
        func_tiled = self.mod_tiled.get_function('blur_tiled')

        # Measure time
        start.record()
        start_ = time.time()
        func_tiled(self.A_d, self.kernel_d, self.output_d, np.int32(kernel.shape[1]), np.int32(A.shape[1]), np.int32(A.shape[0]), 
        block=(32, 32, 1), grid = (np.int(grid_dim_y),np.int(grid_dim_x),1))
        end_ = time.time()
        end.record()

        # CUDA Event synchronize
        end.synchronize()

        output_gpu = self.output_d.get()
        kernel_time = end_-start_

        # Return the result and the kernel execution time
        return output_gpu, kernel_time   
 

if __name__ == '__main__':
    import cv2
    import glob

    # def show(images):

    #     """
    #     Plot the top 16 images (index 0~15) for visualization.
    #     :param images: images to be shown
    #     """

    #     xshow = images[:16]
    #     fig = plt.figure(figsize=(6,6))
    #     fig.set_tight_layout(True)
        
    #     for i in range(16):
    #         ax = fig.add_subplot(4,4,i+1)
    #         # ax.imshow((xshow[i,:]*255).astype(np.uint8))
    #         ax.imshow((xshow[i,:]))
    #         ax.axis('off')         

    # Create kernel filter for tiled image blur
    # kernel = np.float32(np.random.randint(low=0, high=5, size=(5,5)))
    # kernel_flip = np.rot90(kernel, 2).astype(np.float32)
    
    # l = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],
    # [1,1,1],[1,1,1],[1,1,1],[1,1,1]] # 3x3 for CIFAR/11x11 for PATHOLOGY

    l2 = [[1,1,1],[1,1,1],[1,1,1]]

    kernel = np.asarray(l2, dtype=np.float32) # l1
    kernel_flip = np.rot90(kernel, 2).astype(np.float32)

    print(os.getcwd())

    # Create the input matrix
    # img = cv2.imread("IMG-1595.jpg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # A=np.float32(gray)
    # msize.append(A.shape[0] * A.shape[1])
    
    naive_times=[]
    tiled_times=[]

    # Iterate through all 50,000 images
    file_list = []
    sheep_list = []
    
    for file in glob.glob("*.png"):# 42259_horse.png
        file_list.append(file)
    # for file in glob.glob("*.png"):
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        A=np.float32(gray)

        sheep_list.append(A.shape)
    # print(len(file_list))

    # for fl in range(len(file_list)):
    #     img = cv2.imread(file_list[fl])
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     A=np.float32(gray)
    #     sheep_list.append(A.shape)

    # print(sheep_list[÷0])
    # Create kernel filter; 1's for box kernel aka avg blur
    # kernel = np.float32(np.array([[1,1,1], [1,1,1], [1,1,1]]))
    # kernel_flip = np.rot90(kernel, 2).astype(np.float32)

    # Create instance for CUDA
        module = Blur()

    # Record times
    # for e in range(3):
        # Compute Naive GPU image blur
        cu_output_naive, t = module.image_avg_blur(A)
        cu_output_tiled, t2 = module.image_avg_blur_tiled(A, kernel_flip)
        # print(t)
        naive_times.append(t)
        tiled_times.append(t2)

    # print(len(file_list))
    print(sheep_list[0])
    print(len(naive_times))
    print(len(tiled_times))

    
    counter = 0
    for nt in range(len(naive_times)):
        counter += naive_times[nt]
    print("NAIVE Execution Times: ",counter)

    counter2 = 0
    for nt2 in range(len(tiled_times)):
        counter2 += tiled_times[nt2]
    print("TILED Execution Times: ",counter2)

        # Compute Tiled GPU Convolution
        # cu_output_tiled, t_tiled = module.tiled_conv2d_gpu(A, kernel_flip)
        # tiled_times.append(t_tiled)
        # Compute Serial Convolution
        # cpu_output, t_cpu = module.conv2d_cpu(A, kernel)
        # serial_times.append(t_cpu)
    # img2 = cv2.imread("29291_deer.png")
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # A2=np.float32(gray2)

    # MAKE_PLOT = True
    # if MAKE_PLOT:
    #     plt.figure()
        # plt.imsave("/home/daa2162/CUDA_29291_deer.png", A2, cmap="gray")
     # Plot times
    # MAKE_PLOT = True
    # if MAKE_PLOT:
    #     plt.figure()
    #     # plt.imsave("/home/daa2162/Blurred_Image_CIFAR.jpg", cu_output_naive, cmap="gray")
        # plt.imsave("/home/daa2162/CUDA_42257_frog.png", cu_output_naive, cmap="gray")
        # plt.imsave("/home/daa2162/CUDA_TILED_PATH.png", cu_output_tiled, cmap="gray")

        # plt.imsave("/home/daa2162/CUDA_Blurred_Image_PATH_LAST.jpg", cu_output_naive, cmap="gray")
        # plt.imsave("/home/daa2162/CUDA_42257_frog_OG.png", A, cmap="gray")
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
    

   
