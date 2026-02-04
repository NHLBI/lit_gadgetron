#import warnings
#warnings.filterwarnings("ignore")
import numpy as np
from scipy import ndimage
import time
import cupy as cp
import cupyx.scipy.ndimage as cpsp
import sigpy
from sigpy.mri.app import EspiritCalib

import torch 
import cupy as cp
import gc

def get_GPU_most_free():
    numdevices =  torch.cuda.device_count()
    GPU_free=0
    dev_GPU=-1
    for devno in range(numdevices):
        with torch.cuda.device(devno):
            f,t = torch.cuda.mem_get_info() 
            if GPU_free < f:
                GPU_free = f
                dev_GPU=devno
        
    return GPU_free,dev_GPU
import copy
def espirit_csm_calculation(channels_images,calib_width,kernel_width,crop,thresh): #,calib_width=12,kernel_width=3,crop=0.5,thresh=0.02

    in_image = copy.deepcopy(channels_images)
    print("is it working")
    print(calib_width)
    print(kernel_width)
    print(crop)
    print(thresh)
    GPU_freeM,dev_num=get_GPU_most_free()
    print(f"Free GPU {GPU_freeM}")
    tr=np.arange(len(channels_images.shape))[::-1] #3D tr (3,2,1,0) 2D (2,1,0)
    fft_axes=-np.arange(1,len(channels_images.shape)) # 3D [-1,-2,-3] 2D (-1,-2)
    print(tr)
    print(fft_axes)
    
    if dev_num!=-1:
        with cp.cuda.Device(dev_num):
            images = cp.array(channels_images).transpose(tr)
            ksp = sigpy.fft(images,axes=fft_axes)
            csm_cp = EspiritCalib(ksp, calib_width=calib_width, kernel_width=kernel_width,crop=crop,thresh=thresh,show_pbar=False, device=cp.cuda.Device(dev_num)).run()
            csm = cp.asnumpy(csm_cp).tranpose(tr).astype(np.complex64)
    print(type(csm))
    
    return csm

def smooth(img, box=5, use_cpu=False):
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    '''
    if use_cpu:
        t_real = np.zeros(img.shape)
        t_imag = np.zeros(img.shape)

        ndimage.filters.uniform_filter(img.real,size=box,output=t_real)
        ndimage.filters.uniform_filter(img.imag,size=box,output=t_imag)
    else:
        img_cp = cp.array(img)
        t_real = cp.zeros(img.shape)
        t_imag = cp.zeros(img.shape)
        cpsp.uniform_filter(img_cp.real,size=box,output=t_real)
        cpsp.uniform_filter(img_cp.imag,size=box,output=t_imag)

    simg = t_real + 1j*t_imag
    
    return simg
    
def calculate_csm_walsh(img, smoothing=5, niter=1):
    '''Calculates the coil sensitivities for 2D data using an iterative version of the Walsh method

    :param img: Input images, ``[coil, y, x]``
    :param smoothing: Smoothing block size (default ``5``)
    :parma niter: Number of iterations for the eigenvector power method (default ``3``)

    :returns csm: Relative coil sensitivity maps, ``[coil, y, x]``
    :returns rho: Total power in the estimated coils maps, ``[y, x]``
    '''

    if img.ndim == 3:
        img = img[:,np.newaxis,:,:] # add a z dim
    #assert img.ndim == 3, "Coil sensitivity map must have exactly 3 dimensions"

    ncoils = img.shape[0]
    nz = img.shape[1]
    ny = img.shape[2]
    nx = img.shape[3]
    
    start_time = time.time()

    # Compute the sample covariance pointwise
    Rs = np.zeros((ncoils,ncoils,nz,ny,nx),dtype=img.dtype)
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:,:] = img[p,:,:,:] * np.conj(img[q,:,:,:])
            
    print("--- First for loop %s seconds ---" % (time.time() - start_time))

    # Smooth the covariance
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q] = smooth(Rs[p,q,:,:,:], smoothing, use_cpu=True)
    
    print("--- Second for loop %s seconds ---" % (time.time() - start_time))

    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    print("I got here")
    rho = np.zeros((nz, ny, nx))
    csm = np.zeros((ncoils, nz, ny, nx),dtype=img.dtype)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                R = Rs[:,:,z,y,x]
                v = np.sum(R,axis=0)
                lam = np.linalg.norm(v)
                v = v/lam

                for iter in range(niter):
                    v = np.dot(R,v)
                    lam = np.linalg.norm(v)
                    v = v/lam

                rho[z,y,x] = lam
                csm[:,z,y,x] = v
    print("--- Third for loop %s seconds ---" % (time.time() - start_time))

    return (csm, rho)

def calculate_csm_walsh_gpu(img, smoothing=5, niter=3):
    '''Calculates the coil sensitivities for 2D data using an iterative version of the Walsh method

    :param img: Input images, ``[coil, y, x]``
    :param smoothing: Smoothing block size (default ``5``)
    :parma niter: Number of iterations for the eigenvector power method (default ``3``)

    :returns csm: Relative coil sensitivity maps, ``[coil, y, x]``
    :returns rho: Total power in the estimated coils maps, ``[y, x]``
    '''
    odim = img.ndim
    if img.ndim == 3:
        img = img[:,np.newaxis,:,:] # add a z dim
    #assert img.ndim == 3, "Coil sensitivity map must have exactly 3 dimensions"
    img = cp.array(img)
    ncoils = img.shape[0]
    nz = img.shape[1]
    ny = img.shape[2]
    nx = img.shape[3]
    
    start_time = time.time()

    # Compute the sample covariance pointwise
    Rs = cp.zeros((ncoils,ncoils,nz,ny,nx),dtype=img.dtype)
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:,:] = img[p,:,:,:] * cp.conj(img[q,:,:,:])
            
    # Smooth the covariance
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q] = smooth(Rs[p,q,:,:,:], smoothing)
    
    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    rho = cp.zeros((nz, ny, nx))
    csm = cp.zeros((ncoils, nz, ny, nx),dtype=img.dtype)
    v = cp.sum(Rs,axis=0)
    lam = cp.linalg.norm(v,axis=0)
    vo = v/lam
    #for z in range(nz):
    if(odim==3):
        vo = vo.squeeze()
        Rs = Rs.squeeze()
        for x in range(nx):
            v = vo[:,:,x]
            
            for iter in range(niter):
                v = cp.transpose(cp.matmul(cp.transpose(Rs[:,:,:,x],[2,0,1]),v),[1,0,2])
                lam = cp.linalg.norm(v,axis=0)
                v = v/lam
                v = cp.diagonal(v,axis1=1,axis2=2)
            
            rho[:,:,x] = cp.diagonal(lam)
            csm[:,0,:,x] = v

    else:
        for y in range(ny):
            for x in range(nx):
                v = vo[:,:,y,x]
                for iter in range(niter):
                    v = cp.transpose(cp.matmul(cp.transpose(Rs[:,:,:,y,x],[2,0,1]),v),[1,0,2])
                    lam = cp.linalg.norm(v,axis=0)
                    v = v/lam
                    v = cp.diagonal(v,axis1=1,axis2=2)
                
                rho[:,y,x] = cp.diagonal(lam)
                csm[:,:,y,x] = v
            
    del v,lam,Rs
    
    return (cp.asnumpy(csm), cp.asnumpy(rho))
