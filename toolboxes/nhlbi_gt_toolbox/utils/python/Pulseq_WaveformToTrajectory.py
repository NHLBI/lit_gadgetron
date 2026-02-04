import numpy as np 
import os.path as op
import ismrmrd as mrd
from utils_function import eprint, load_json, parse_params, read_params
from time import time 
import gadgetron
import glob
from utils_function import eprint, load_json, parse_params, read_params,create_ismrmrd_image_fast
import os 
from einops import rearrange
from copy import deepcopy
import typing
from scipy.fft import fft,ifft,fftshift,ifftshift
import cupy as cp
from cupyx.scipy.fft import fft as cufft
from cupyx.scipy.fft import ifft as cuifft
from cupyx.scipy.fft import fftshift as cufftshift
from cupyx.scipy.fft import ifftshift as cuifftshift
import torch
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

def read_n_to_last_line(filename:str, n:int = 1):
    """
    Returns the nth before last line of a file (n=1 gives last line) 

    Parameters
    ----------
    
    filename : str,
        file path
    
    n : int, (default n=1)
        nth before last line.

    Returns
    -------

    last_line : str,
       nth before last line.
       
    """

    num_newlines = 0
    with open(filename, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)    
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b'\n':
                    num_newlines += 1
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line

def siemens_calculate_transform_pcs_to_dcs(patient_position:str):
    """
       Build a rotation matrix to transform a position in patient coordinates to a position in physical coordinates.

    Parameters
    ----------
    
    patient_position : str, 
        Possible patient positions are:
            - HFP : Head first-prone
            - HFS : Head first-supine
            - HFDR : Head first-decubitus right
            - HFDL : Head first-decubitus left
            - FFP : Feet first-prone
            - FFS : Feet first-supine
            - FFDR :  Feet first-decubitus right
            - FFDL : Feet first-decubitus left

    Returns
    ------- 

    R_pcs2dcs : np.float32,
        rotation matrix

    """
    # Initialize the rotation matrix
    R_pcs2dcs = np.zeros((3, 3), dtype=float)
    
    # Define constants for patient positions
    HFP = 'HFP'  # Head first / prone
    HFS = 'HFS'  # Head first / supine
    HFDR = 'HFDR'  # Head first / decubitus right
    HFDL = 'HFDL'  # Head first / decubitus left
    FFP = 'FFP'  # Feet first / prone
    FFS = 'FFS'  # Feet first / supine
    FFDR = 'FFDR'  # Feet first / decubitus right
    FFDL = 'FFDL'  # Feet first / decubitus left
    
    # Calculate the rotation matrix based on patient position
    if patient_position == HFP:
        R_pcs2dcs = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
    elif patient_position == HFS:
        R_pcs2dcs = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
    elif patient_position == HFDR:
        R_pcs2dcs = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
    elif patient_position == HFDL:
        R_pcs2dcs = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ])
    elif patient_position == FFP:
        R_pcs2dcs = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif patient_position == FFS:
        R_pcs2dcs = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    elif patient_position == FFDR:
        R_pcs2dcs = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    elif patient_position == FFDL:
        R_pcs2dcs = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError('Unknown patient position: {}'.format(patient_position))
    
    return R_pcs2dcs

def readGIRFKernel(girf_path):
    """
    Reading GIRF text files and returning GIRF. 

    Parameters
    ---------- 
    girf_path : string, 
        path to girf text files 

    Returns
    -------

    GIRF : ndarray, shape(samples,3)

    dtGIRF : float, 
        dwell time of GIRF (s)
    """
    girf_path_x=f'{girf_path}x.txt'
    girf_path_y=f'{girf_path}y.txt'
    girf_path_z=f'{girf_path}z.txt'

    if not op.exists(girf_path_x):
        raise ValueError(f'This girf path does not exist :{girf_path_x}')
    if not op.exists(girf_path_y):
        raise ValueError(f'This girf path does not exist :{girf_path_y}')
    if not op.exists(girf_path_z):
        raise ValueError(f'This girf path does not exist :{girf_path_z}')
    
    with open(girf_path_x,"r") as girf_file_x:
        girfx_lines=girf_file_x.read().splitlines()
        girfx_header=np.float64(girfx_lines[0:4]) # Header 0: samples 2 : dt in ns
        dtGIRF=girfx_header[2]*1e-9 # from ns to s
        sGIRF =int(girfx_header[0])
        girfx_data=np.float64(girfx_lines[4:])
        girfx_data=np.reshape(girfx_data,[sGIRF,2])
        girfx=girfx_data[:,0]+1j*girfx_data[:,1]
    
    with open(girf_path_y,"r") as girf_file_y:
        girfy_lines=girf_file_y.read().splitlines()
        girfy_header=np.float64(girfy_lines[0:4]) # Header
        girfy_data=np.float64(girfy_lines[4:])
        girfy_data=np.reshape(girfy_data,[sGIRF,2])
        girfy=girfy_data[:,0]+1j*girfy_data[:,1]
    
    with open(girf_path_z,"r") as girf_file_z:
        girfz_lines=girf_file_z.read().splitlines()
        girfz_header=np.float64(girfz_lines[0:4]) # Header
        girfz_data=np.float64(girfz_lines[4:])
        girfz_data=np.reshape(girfz_data,[sGIRF,2])
        girfz=girfz_data[:,0]+1j*girfz_data[:,1]
    
    print('GIRF not swap')
    #GIRF=np.column_stack((girfy,girfx,girfz)) # Swap x,y Needs investigation
    GIRF=np.column_stack((girfx,girfy,girfz)) # Swap x,y Needs investigation
    return GIRF,dtGIRF

def hanningt(windowlength):
    """
    Returning the Hanning window. 
    It is equivalent to matlab function that does not include zeros extremities.

    Parameters
    ---------- 
    windowlength : int
        Number of points in the output window.

    Returns
    -------

    hfilter : ndarray, shape(M,)
        The window, with the maximum value normalized to one
    """
    
    hfilter=np.hanning(windowlength+2)
    hfilter=hfilter[1:-1]
    return hfilter

def user_opts(user_opts_in: dict, user_opts_default: dict) -> dict:
    """
    Modify default dictionnary with user dictionnary values

    Parameters
    ----------
    
    user_opts_in : dict,
        Dictionnary with user values 

    user_opts_default : dict.
        Dictionnary with default value

    Returns
    -------

    user_opts_default : dict,
        Updated default dictionnary with user values
    """
    input_keys = list(user_opts_in.keys())
    for key in input_keys:
        if key in user_opts_default.keys():
            user_opts_default[key] = user_opts_in[key]
        else:
            user_opts_default.update({key:user_opts_in[key]})
    
    return user_opts_default 
def apply_GIRF_plus_cp(gradients_nominal_input,dt,R_dict=typing.Dict,tRR=0,batchsize=32*5,Clock_shift=0.85e-6,dt_shift=0.5):
    """
    Applying GIRF on gradients waveforms. Optimized with CUDA implementation 

    Parameters
    ---------- 
    gradients_nominal : ndarray, shape(samples,interleaves,3)
        Gradients

    dt : float
        sample dwell time (in s)

    R_dict : dict
        Dictionnary that contains the following information :
            - R : ndarray, rotation matrix (default : np.eye(3))
            - T : float, Field strength (in Tesla) (default : 0.55T)
            - systemModel : strings, name of the system (default : '')
            - GIRF_folder : strings, path to the GIRF text files (default : '/opt/GIRF')

    tRR : float, 
        sub-dwell-time offset (default : 0)

    Returns
    -------

    kPred : ndarray, shape(samples,interleaves,3) 
        predicted kspace trajectories

    GPred : ndarray, shape(samples,interleaves,3) 
        predicted gradients
    """
    t1=time()
    
    default_R={
        "R": np.eye(3),
        "T" : 0.55,
        "systemModel":'',
        "GIRF_folder": '/opt/GIRF/GIRF_20250225'
    }
        
    user_opts(R_dict,default_R)

    if default_R["systemModel"]=="MAGNETOM eMeRge-XL":
        girf_path=op.join(default_R['GIRF_folder'],"GIRF_fmax_")
    else:
        girf_path=op.join(default_R['GIRF_folder'],"GIRF")
    
    [GIRF,dtGIRF]=readGIRFKernel(girf_path)

    #
    [samples, interleaves, ndim] = gradients_nominal_input.shape
    if ndim!=3:
        # 2D
        gradients_nominal=np.zeros([samples, interleaves,3])
        gradients_nominal[:,:,:ndim]=gradients_nominal_input
    else:
        gradients_nominal=gradients_nominal_input
    
    GPU_freeM,dev_num=get_GPU_most_free()
    with cp.cuda.Device(dev_num):
        R=cp.asarray(default_R['R'])
        batch=np.arange(0,interleaves-1,min(interleaves,batchsize)).tolist()
        batch.append(interleaves)
        dtype_complex=np.complex64 #np.cdouble
        # if readout is real long, need to pad the GIRF measurement
        s_GIRF=np.shape(GIRF)[0]
        if samples*dt > dtGIRF*s_GIRF:
            pad_factor = 1.5 * (samples * dt) / (dtGIRF * s_GIRF) # 1.5 factor to ease calculations below
            new_GIRF = np.zeros((round(s_GIRF * pad_factor), 3),dtype=dtype_complex)
            for i in range(3):
                fft_GIRF = fftshift(ifft(ifftshift(GIRF[:,i])))
                zeropad = round( abs((s_GIRF-new_GIRF.shape[0]) /2 ))
                temp = np.zeros((np.shape(new_GIRF)[0],1),dtype=dtype_complex)
                #Smoothing of padding
                H_size= 200
                hfilter=hanningt(H_size)
                fft_GIRF[:int(H_size/2)] = fft_GIRF[:int(H_size/2)]*np.reshape(hfilter[:int(H_size/2)],np.shape(fft_GIRF[:int(H_size/2)]))
                fft_GIRF[-int(H_size/2):] = fft_GIRF[-int(H_size/2):]*np.reshape(hfilter[-int(H_size/2):],np.shape(fft_GIRF[-int(H_size/2):]))
                temp[zeropad:zeropad + s_GIRF] = fft_GIRF
                new_GIRF[:,i] = fftshift(fft(fftshift(temp[:,0])))
                s_GIRF=np.shape(new_GIRF)[0]
                GIRF=new_GIRF


        # GIRF prediction
        ADCshift = Clock_shift + dt_shift * dt + tRR * dt #NCO Clock shift
        L = round(dtGIRF * s_GIRF / dt) # when waveform not at GRT
        dw = 1 / (L * dt) # frequency resolution [Hz]
        BW = 1 / dt
        # Scale k-space in units of 1/cm
        gamma = 2.67522212e+8   # gyromagnetic ratio for 1H [rad/sec/T]

        
        GPred = cp.zeros(gradients_nominal.shape,dtype=dtype_complex) # SAMPLES INTERLEAVES NDIM
        
        hannin=cp.asarray(hanningt(400))
        
        w = cp.arange(-np.floor(L/2),np.ceil(L/2))* dw # [Hz]
        # Differenet index ranges 
        index_range=(np.floor(L/2)+np.arange(-np.floor(samples/2),np.ceil(samples/2))).astype(np.int32)
        index_range_ter=(np.floor(L/2)+np.arange(-np.floor(samples/2),np.ceil(samples/2))).astype(np.int32)
        index_range_bis=(np.floor(L/2)+np.arange(-np.floor(s_GIRF/2),np.ceil(s_GIRF/2))).astype(np.int32)
        i_s=int(index_range[samples-1])

        for b_idx in range(len(batch)-1):
            Predicted = cp.zeros((batch[b_idx+1]-batch[b_idx],samples, 3),dtype=dtype_complex)
            G1=(R @ cp.asarray(gradients_nominal[:,batch[b_idx]:batch[b_idx+1],:]).transpose([1,2,0])).transpose([0,2,1]) #[INTERLEAVES L DIM]

            for ax in range(3):
                G_bis=cp.zeros((batch[b_idx+1]-batch[b_idx],L),dtype=np.double)
                G_bis[:,index_range]=G1[:,:,ax]
                H1=G_bis[:,i_s,None]*hannin[None,:]
                G_bis[:,i_s+1:int(i_s+1+H1.shape[1]*0.5)] = H1[:,int(H1.shape[1]*0.5):]
                V=cufftshift(cufft(cuifftshift(G_bis,axes=-1),axis=-1),axes=-1)
                GIRF2 = cp.zeros((L,1),dtype=dtype_complex)
                if dt > dtGIRF:
                    ##NOT tested
                    # RR crop
                    GIRF2[:] = GIRF[round(s_GIRF/2 - L/2):round(s_GIRF/2 + L/2),ax]
                    # RR .. padding
                    temp = hanningt(10)
                    GIRF2[0] = 0
                    GIRF2[-1] = 0
                    GIRF2[1:round(temp.shape[0]/2)] = GIRF2[1:round(temp.shape[0]/2)]*temp[:int(temp.shape[0]/2)]
                    GIRF2[-1-round(temp.shape[0]/2):-2] = GIRF2[-1-round(temp.shape[0]/2):-2]*temp[int(temp.shape[0]/2):]
                else :
                    #--------------------------------------------------------------
                    # Modified by NGL
                    # Usual operation.. (ACW)
                    #zeropad = round(abs((s_GIRF-L)/2)) #amount of zeropadding
                    #GIRF1((1+zeropad):(zeropad+s_GIRF))= GIRF(:,ax)
                    #--------------------------------------------------------------
                    
                    GIRF2[index_range_bis.astype(np.int32),0] = GIRF[:,ax]

                P2=V*GIRF2[None,:,0]*cp.exp(1j * ADCshift * 2 * np.pi * w[None,:])
                PredGrad = cufftshift(cuifft(cuifftshift(P2,axes=-1),axis=-1),axes=-1)
                PredGrad_tmp=np.abs(PredGrad)
                PredGrad_tmp[np.real(PredGrad)<0]*=-1 #Correct polarity of gradients
                Predicted[:,:,ax]=PredGrad_tmp[:,index_range_ter]

            GPred[:,batch[b_idx]:batch[b_idx+1],:] = (R.T  @ Predicted.transpose([0,2,1])).transpose([2,0,1]) #SAMPLES INTERLEAVES NDIM
        scale_factor=0.01*(gamma/(2*np.pi))*0.01*dt
        kPred = cp.cumsum(scale_factor*GPred,axis=0).get() # (kPred*0.01):: assuming gradients in are in G/cm!!!

    return  kPred,cp.asnumpy(GPred)



def Pulseq_WaveformtoTrajectoryGadget(connection):
    
    mrd_header=connection.header
    params={'traj_folder_or_file':'/opt/data/bstar_traj/',
            'GIRF_folder':'/opt/GIRF/GIRF_20250225',
            'applyGIRF':True,
            'split_echoes':True
            }


    boolean_keys=['applyGIRF','split_echoes']
    str_keys=['traj_folder_or_file','GIRF_folder']



    params_init = parse_params(connection.config)
    params=read_params(params_init,params_ref=params,boolean_keys=boolean_keys,str_keys=str_keys)

    reading_time=time()
    trj_file=params['traj_folder_or_file']
    print(trj_file)
    # Detect Traj_file.h5 based on the hash
    if op.isdir(trj_file):
        hash_traj=''
        for k in range(len(mrd_header.userParameters.userParameterString)):
                if mrd_header.userParameters.userParameterString[k].name =='tSequenceVariant':
                        hash_traj=mrd_header.userParameters.userParameterString[k].value
        print(hash_traj)
        if hash_traj:
                seq_files=glob.glob(op.join(trj_file,'*.seq'))
                seq_files.sort()
                for seq_file in seq_files:
                        hash_seq=read_n_to_last_line(seq_file).split(' ')[-1][:-1]
                        if hash_traj==hash_seq:
                            print(f"Seq file found : {op.basename(seq_file)[:-4]}")
                            break
                trj_file=glob.glob(op.join(trj_file,f"*{op.basename(seq_file)[:-4]}*.h5"))[0]

    with mrd.File(trj_file,'r') as mrd_file:
        traj_header=mrd_file['dataset'].header
        traj_acq = mrd_file['dataset'].acquisitions[:]
        

    reading_time_traj=time()
    print(' Reading traj running time %f s'%(reading_time_traj-reading_time))

    traj_unscaled = rearrange(np.array([tr.traj for tr in traj_acq if tr.flags==0]).squeeze(),'INT RO DIM -> RO INT DIM')
    ## ANGLES INFORMATION
    #phi_angles=np.array([float(tr.data.real[0,0]) for tr in traj_acq if tr.flags==0]).squeeze()
    #theta_angles=np.array([float(tr.data.real[0,1]) for tr in traj_acq if tr.flags==0]).squeeze()
    #nav_angles=np.concatenate([phi_angles[:,None],theta_angles[:,None]],axis=1) # INT 2

    ## userParameterLong keys (true_resolution_flag,fid_navigator_flag,self_navigation_flag,grad_samples,adc_samples)
    #self_navigation_flag= [uPL.value for uPL in traj_header.encoding[0].trajectoryDescription.userParameterLong if uPL.name=='self_navigation_flag'][0]
    #adc_samples= [uPL.value for uPL in traj_header.encoding[0].trajectoryDescription.userParameterLong if uPL.name=='adc_samples'][0]
    
    ## userParameterDoublekeys (grid_scale_factor,grad_raster_time,TE_nav,dwell_time_nav,real_dwell_time)
    #Do need real_dwell_time should be good for bstar to verify 
    
    #real_dwell_time= [uPD.value for uPD in traj_header.encoding[0].trajectoryDescription.userParameterDouble if uPD.name=='real_dwell_time'][0] #s
    
    
    # WARNING Need to be tested with self NAVIGATOR ON
    idx_ref=0
    while (not traj_acq[idx_ref].flags==0):
        idx_ref+=1
    discard_pre=traj_acq[idx_ref].discard_pre
    discard_post=traj_acq[idx_ref].discard_post
    real_dwell_time=traj_acq[idx_ref].sample_time_us*1e-6
    print(real_dwell_time)
    nr_readouts=traj_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum+1
    nr_interleaves=traj_header.encoding[0].encodingLimits.segment.maximum+1
    
    if params["applyGIRF"]:
        
        gamma_matlab=42.576e6 #matlab gamma
        gr_raster_time=1e-5
        gamma_GIRF=2.67522212e+8
        scale_factor_before_GIRF=1/(2*np.pi*gamma_matlab*1e-3)/gr_raster_time
        scale_factor_inside_GIRF=0.01*(gamma_GIRF/(2*np.pi))*0.01*real_dwell_time
        # Error in replication bstar (multiply by raster time instead of dwell_time)
        gradients_nominal=np.diff(traj_unscaled,axis=0)/(2*np.pi*gamma_matlab*1e-3)/gr_raster_time
        all_gradients=np.concatenate([gradients_nominal,np.zeros([2,traj_unscaled.shape[1],traj_unscaled.shape[2]])],axis=0) #all_gradients=np.concatenate([np.zeros([discard_pre+1,traj_unscaled.shape[1],traj_unscaled.shape[2]]),gradients_nominal,],axis=0)
        """
        #SAGITAL
        R_gcs2dcs_SAG = np.array([
                [0, 0, 1],  # [PE] maps to [Z]
                [-1, 0, 0],  # [RO] maps to [Y]
                [0, 1, 0]   # [SL] maps to [X]
            ])
        #CORONAL
        R_gcs2dcs_COR = np.array([
                [1, 0, 0],  # [PE] maps to [Z]
                [0, 0, -1],  # [RO] maps to [Y]
                [0, 1, 0]   # [SL] maps to [X]
            ])
        #TRANSVERSAL
        R_gcs2dcs_TRANS = np.array([
                [0, 1, 0],  # [PE] maps to [Z]
                [-1, 0, 0],  # [RO] maps to [Y]
                [0, 0, -1]   # [SL] maps to [X]
            ])
        """
    
    
    max_xyz=[np.abs(traj_unscaled[:,:,0]).max(),np.abs(traj_unscaled[:,:,1]).max(),np.abs(traj_unscaled[:,:,2]).max()]
    for ax in range(len(max_xyz)):
        traj_unscaled[...,ax] *= 0.5 / max_xyz[ax]

    eprint(f"shape traj {traj_unscaled.shape}")
    traj_pulseq=traj_unscaled

    FOV=traj_header.encoding[0].reconSpace.fieldOfView_mm
    matrixR=traj_header.encoding[0].reconSpace.matrixSize
    wsize_cuda=32
    connection.filter(lambda input: type(input) ==mrd.Acquisition)
    for acq in connection :
        # Modify header
        if acq.scan_counter==1 and params["applyGIRF"]:
            GIRF_time=time()
            rot_mat_prs     = np.transpose(np.array([np.array(acq.phase_dir),np.array(acq.read_dir),np.array(acq.slice_dir)]))
            sR = {"R" : rot_mat_prs,
            "T" : mrd_header.acquisitionSystemInformation.systemFieldStrength_T,
            "systemModel": mrd_header.acquisitionSystemInformation.systemModel,
            "GIRF_folder":params['GIRF_folder']
            }
            # Fix GIRF correction but it is incorrect
            trajectory_GIRF, grad_GIRF = apply_GIRF_plus_cp(all_gradients, real_dwell_time, sR,0,batchsize=32*5) #RO INT DIM #tRR=1.25
            GIRF_time_end=time()
            print(' GIRF time %f s'%(GIRF_time_end-GIRF_time))
            traj_c = trajectory_GIRF[:,:,:].real/scale_factor_before_GIRF/scale_factor_inside_GIRF 
            for ax in range(len(max_xyz)):
                traj_c[...,ax] *= 0.5 / max_xyz[ax]
            traj_echo1=traj_c[:int(traj_unscaled.shape[0]/2),...]
            traj_echo2=traj_c[-int(traj_unscaled.shape[0]/2):,...]
            traj_pulseq=np.concatenate([traj_echo1,traj_echo2],axis=0)

            eprint(params['split_echoes'])
        if params['split_echoes']:
            data_echo1=deepcopy(acq.data[:,discard_pre:discard_pre+traj_echo1.shape[0]])
            dcf_echo1=np.ones((traj_echo1.shape[0],1))
            traj_acq_echo1=np.concatenate([deepcopy(traj_echo1[:,int(acq.scan_counter-1),:]),dcf_echo1],axis=-1)

            data_echo2=deepcopy(acq.data[:,-(discard_post+traj_echo2.shape[0]):-discard_post])[:,::-1]
            dcf_echo2=np.ones((traj_echo2.shape[0],1))
            traj_acq_echo2=np.concatenate([deepcopy(traj_echo2[:,int(acq.scan_counter-1),:]),dcf_echo2],axis=-1)[::-1,:]

            acq.resize(trajectory_dimensions=4)
            acq.resize(data_echo1.shape[1],data_echo1.shape[0],trajectory_dimensions=4)
            acq.traj[:]=traj_acq_echo1
            acq.data[:]=data_echo1
            acq.discard_pre=0
            acq.discard_post=0
            acq.center_sample=0
            acq_2=mrd.Acquisition(head=acq.getHead())
            acq_2.resize(trajectory_dimensions=4)
            acq_2.resize(data_echo2.shape[1],data_echo2.shape[0],trajectory_dimensions=4)
            acq_2.traj[:]=traj_acq_echo2
            acq_2.data[:]=data_echo2
            acq_2.discard_pre=0
            acq_2.discard_post=0
            acq_2.idx.set =1 #Set second echo at set 1
            acq_2.center_sample=0
            connection.send(acq)
            connection.send(acq_2)
            
        else:
            data_acq=deepcopy(acq.data[:,discard_pre:-discard_post])
            dcf=np.ones((traj_unscaled.shape[0],1))
            traj_acq=np.concatenate([deepcopy(traj_pulseq[:,int(acq.scan_counter-1),:]),dcf],axis=-1)
            acq.resize(trajectory_dimensions=4)
            acq.resize(data_acq.shape[1],data_acq.shape[0],trajectory_dimensions=4)
            acq.traj[:]=traj_acq
            acq.data[:]=data_acq
            acq.discard_pre=0
            acq.discard_post=0
            acq.center_sample=0
            connection.send(acq)
        #acq.user_float[:3]=[FOV.x,FOV.y,FOV.z]
        #acq.user_int[:3]=[int(wsize_cuda*np.round(matrixR.x/wsize_cuda)),int(wsize_cuda*np.round(matrixR.y/wsize_cuda)),int(wsize_cuda*np.round(matrixR.z/wsize_cuda))]
        #acq.user_int[3]=traj_pulseq.shape[1] #Total number of scans 
            
        #acq.resize(number_of_samples=acq.number_of_samples,active_channels=acq.active_channels,trajectory_dimensions=4)
        #dcf=np.ones((traj_unscaled.shape[0],1))
        #traj_acq=np.concatenate([np.zeros([discard_pre,4]),np.concatenate([deepcopy(traj_pulseq[:,int(acq.scan_counter-1),:]),dcf],axis=-1),np.zeros([discard_post,4])],axis=0)
        #acq.traj[:]=traj_acq
        
        


if __name__ == '__main__':
    gadgetron.external.listen(2009,Pulseq_WaveformtoTrajectoryGadget)