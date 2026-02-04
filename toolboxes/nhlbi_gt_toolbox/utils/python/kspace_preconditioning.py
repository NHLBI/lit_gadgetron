import gadgetron
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from utils_function import get_GPU_most_free
import sigpy      as sp
import sigpy.mri  as mr
from copy import deepcopy

def estimate_k_precond(traj,matrix_x,matrix_y,matrix_z,lamda,ovs):    
    print(f" me_x {matrix_x} me_y {matrix_y} me_z {matrix_z} l {lamda} ovs {ovs}")
    coord=deepcopy(traj).T
    coord[...,0]= coord[...,0]*matrix_x
    coord[...,1]= coord[...,1]*matrix_y
    coord[...,2]= coord[...,2]*matrix_z
    matrix_a= (matrix_x,matrix_y, matrix_z)
    mps_precond = np.ones((1,) + matrix_a) 
    mps_precond /= len(mps_precond)**0.5
    GPU_freeM,dev_num=get_GPU_most_free()
    if dev_num!=-1:
        devnum=1
        device = sp.Device(devnum)
        dcf_precond=mr.kspace_precond(mps_precond,coord=sp.to_device(coord, device),device=sp.Device(device),lamda=lamda, oversamp=ovs)
        dcf_prec=np.sqrt(dcf_precond.squeeze().get().flatten())
    #eprint(f"Free Memory GPU {GPU_freeM} GPU num {dev_num}")
    return dcf_prec.astype(np.float32)

    
      
