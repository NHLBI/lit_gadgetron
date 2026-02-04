import warnings
warnings.filterwarnings("ignore")
import gadgetron
import numpy as np 
import ismrmrd as mrd
import registration_oflow3D as reg
import time
import sys
import os
import io
from contextlib import redirect_stderr, redirect_stdout
import torch
os.environ["NUMBA_CUDA_ENABLE_PYNVJITLINK"] = "1"

def get_GPU_most_free(gpu_list=[]):

    if len(gpu_list)==0:
        gpu_list=list(range(torch.cuda.device_count()))
    GPU_free=0
    dev_GPU=-1
    for devno in gpu_list:
        with torch.cuda.device(devno):
            f,t = torch.cuda.mem_get_info() 
            if GPU_free < f:
                GPU_free = f
                dev_GPU=devno
        
    return GPU_free,dev_GPU


def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def registration_one_image(mov_image_np, ref_image_np,gpu_list=[]):
    GPU_freeM,dev_num=get_GPU_most_free(gpu_list)
    print(f"Free Memory GPU {GPU_freeM} GPU num {dev_num}")
    print("Registration of image 1: ", mov_image_np.shape)
    deformation_fields = reg.register_one_image_only_deformation(mov_image_np,ref_image_np,gpu_id=dev_num).transpose(1,2,3,0)

    return deformation_fields.astype(np.float32)

def registration_images(images,ref_index=0,gpu_list=[]):
    
    eprint("Registration of images: ", images.shape,gpu_list)
    #images t,nx,ny,nz
    #[0,2,3,1]
    #images = images.transpose(3,0,1,2)
    #3 2 0 1
    images = images.transpose(3,0,1,2)
    ref_index=min(int(ref_index*images.shape[0]),images.shape[0]-1)
    st = time.time()   
    #fout = io.StringIO
    #ferr = io.StringIO
    #fnull = open(os.devnull, 'w')
    GPU_freeM,dev_num=get_GPU_most_free(gpu_list)
    eprint(f"Free Memory GPU {GPU_freeM} GPU num {dev_num}")
    eprint(f"----------------Running registration : Ref index {ref_index} Nbins {images.shape[0]}--------------")
    #with redirect_stdout(fnull) and redirect_stderr(fnull):
    deformation_fields = reg.register_images_only_deformation(images,ref_index,gpu_id=dev_num)
    print(deformation_fields.shape)
    deformation_fields = deformation_fields.transpose(2,3,4,1,0)
    np.nan_to_num(deformation_fields)
    eprint("Registration Time: ", time.time()-st)

    return deformation_fields.astype(np.float32)

        
def registration_images_old(images,bidirectional=False,ref_index=0):
    
    print("Registration of images: ", images.shape)
    #images t,nx,ny,nz
    #[0,2,3,1]
    #images = images.transpose(3,0,1,2)
    #3 2 0 1
    images = images.transpose(3,0,1,2)
    if(bidirectional):
        ref_index = images.shape[0]//2
    else:
        ref_index=ref_index
        
    st = time.time()   
    fout = io.StringIO
    ferr = io.StringIO
    fnull = open(os.devnull, 'w')
    eprint("----------------Running registration--------------")
    with redirect_stdout(fnull) and redirect_stderr(fnull):
        deformation_fields = reg.register_images_only_deformation(images,ref_index)
    print(deformation_fields.shape)
    deformation_fields = deformation_fields.transpose(2,3,4,1,0)
    np.nan_to_num(deformation_fields)
    eprint("Registration Time: ", time.time()-st)

    return deformation_fields.astype(np.float32)

def registration_images_back(images,bidirectional=False,ref_index=0):
    
    print("Registration of images: ", images.shape)
    #images t,nx,ny,nz
    #[0,2,3,1]
    #images = images.transpose(3,0,1,2)
    #3 2 0 1
    images = images.transpose(3,0,1,2)
    if(bidirectional):
        ref_index = images.shape[0]//2
    else:
        ref_index=ref_index
        
    st = time.time()   
    fout = io.StringIO
    ferr = io.StringIO
    fnull = open(os.devnull, 'w')
    eprint("----------------Running registration--------------")
    deformation_fields = np.zeros((images.shape[0],3,images.shape[1],images.shape[2],images.shape[3]))
    with redirect_stdout(fnull) and redirect_stderr(fnull):
        for idx in range(images.shape[0]):
            idxs = [ref_index,idx]
            deformation_fields[idx,...] = reg.register_images_only_deformation(images[idxs,...],1)[[0],...]
    print(deformation_fields.shape)
    deformation_fields = deformation_fields.transpose(2,3,4,1,0)
    np.nan_to_num(deformation_fields)
    eprint("Registration Time: ", time.time()-st)

    return deformation_fields.astype(np.float32)
      

