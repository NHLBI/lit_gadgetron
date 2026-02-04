
import gadgetron
import ismrmrd as mrd

import numpy as np
import cupy as cp
import sys
import cmath
import math
import time
from scipy import signal

from scipy.fft import fft,ifft,fftshift,ifftshift
from cupyx.scipy.fft import fft as cufft
from cupyx.scipy.fft import ifft as cuifft
from cupyx.scipy.fft import fftshift as cufftshift
from cupyx.scipy.fft import ifftshift as cuifftshift
from cupyx.scipy import ndimage
from cupy.linalg import svd as cusvd

import kaiser_window
from utils_function import eprint, load_json, parse_params, read_params

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, cp.asnumpy(data))
    return y

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


def cufilterData(input, filter):

    
    cuin = cp.asarray(input) 
    if(len(cuin.shape)>2):
        cuin = cp.reshape(cuin,(cuin.shape[0]*cuin.shape[1],cuin.shape[2]))
    filter = cp.asarray(filter)
    
    concat_arr = cp.concatenate(((cuin[:,::-1]),cuin[:,:],(cuin[:,::-1])),axis=1).astype(cp.complex64)
    st = time.time()
    
    temp = cp.zeros([concat_arr.shape[0],max(concat_arr.shape[1],filter.shape[0])],cp.complex64)
    eprint("filter_shape:", filter.shape)
    eprint("concat_arr_shape:", concat_arr.shape)
    if(cuin.shape[1]>cuin.shape[0]):
        for ii in range(0,cuin.shape[0]):
            temp[ii,:] = cp.convolve(concat_arr[ii,:].squeeze(),cufftshift(cufft(filter)),'same')
    else:
        temp = ndimage.convolve1d(concat_arr,cufftshift(cufft(filter)),axis=1)
    

    cp.cuda.runtime.deviceSynchronize()
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (filtrations):', elapsed_time, 'seconds')
    out = temp[:,int(temp.shape[1]/2-cuin.shape[1]/2):int(temp.shape[1]/2+cuin.shape[1]/2)]

    if(len(input.shape)>2):
        out = cp.reshape(out,(input.shape[0],input.shape[1],input.shape[2]))

    
    
    return out

def filterData(input, filter):
    if(len(input.shape)>2):
            input = np.reshape(input,(input.shape[0]*input.shape[1],input.shape[2]))
    out = input
    if(filter.shape[0] > input.shape[1]):
        for ii in range(0,input.shape[0]):
                concat_arr = np.concatenate(((input[ii,::-1]),input[ii,:],(input[ii,::-1])))
                temp = np.real(np.convolve(concat_arr,fftshift(fft(filter)),'same'))
                out[ii,:] = temp[int(temp.shape[0]/2-input.shape[2]/2):int(temp.shape[0]/2+input.shape[2]/2)]
    else:
        temp = np.zeros((input.shape[0],3*input.shape[1]),dtype=complex)
        for ii in range(0,input.shape[0]):
                temp[ii,:] = (np.convolve(np.concatenate((np.flipud(input[ii,:]),input[ii,:],np.flipud(input[ii,:]))),fftshift(fft((filter))),'same'))
    
        out = temp[:,int(temp.shape[1]/2-input.shape[1]/2):int(temp.shape[1]/2+input.shape[1]/2)]
    return out

def correctTrajectoryFluctuations(nav_waveform,nav_angles,highpassfreq=0.01,highpassorder=5,bstar=False):
    """
    Correct Trajectory Fluctuations
   
    Parameters
    ----------
   
    nav_waveform: cp.ndarray,
        NAV/DC waveform signal [nav_samples_times_channels,samples]
 
    nav_angles : np.ndarray,
        DC  angle
 
    highpassfreq: float,
        High Pass cutoff frequency
 
    highpassorder : int,
        High pass filter order  
    Returns
    -------
 
    nav_waveform_filtered : np.ndarray,
        nav_waveform without trajectory fluctuation signal
    """
    [nav_samples_times_channels,samples]=nav_waveform.shape
    nav_waveform_filtered = cp.zeros((nav_waveform.shape))

    
    if bstar:
        print("BSTAR__ANGULAR CORRECTION TO VERIFY")
        interleaves=336
        numNavsPerStack = int(nav_angles.shape[0]/ interleaves)

        interleaves=nav_angles.shape[0]
        numNavsPerStack = nav_angles.shape[1]
        nav_angles_theta= nav_angles[...,1]
        nav_angles_phi= nav_angles[...,0]
        idx_per_interleaves_theta = cp.argsort(-1*nav_angles_theta,axis=1)         
        idx_per_interleaves_phi = cp.argsort(-1*nav_angles_phi,axis=1)     
        nav_w_reshape = cp.reshape(nav_waveform,(nav_samples_times_channels,interleaves,numNavsPerStack))

        sorted_signal=np.zeros(nav_w_reshape.shape)
        for n in range(sorted_signal.shape[0]):
            tmp_nav=nav_w_reshape[n,...]
            sorted_signal[n,...]=np.take_along_axis(cp.asnumpy(tmp_nav), idx_per_interleaves_theta.get(), axis=1)

        highpassorder=5
        sos = signal.butter(highpassorder, 5, btype='high', fs=numNavsPerStack, output='sos')
        filtered_signal = signal.sosfiltfilt(sos, sorted_signal)
        nav_waveform_filtered = np.zeros((filtered_signal.shape))
        for n in range(sorted_signal.shape[0]):
            tmp_nav_filtered=filtered_signal[n,...]
            nav_waveform_filtered_n=nav_waveform_filtered[n,...]
            np.put_along_axis(nav_waveform_filtered_n, idx_per_interleaves_theta.get(),filtered_signal[n,...], axis=1)
            nav_waveform_filtered[n,...]=nav_waveform_filtered_n
        nav_waveform_filtered=cp.asarray(np.reshape(nav_waveform_filtered,nav_waveform.shape))

    else:
        una = cp.unique(nav_angles)
        idx = cp.argsort(-1*nav_angles) #max to min
        interleaves = len(una)
        
        factor = (int(math.ceil(samples / interleaves)) % int(interleaves)) - round(samples/ interleaves)
        if (factor < 0):
            factor = 0
        
        numNavsPerStack = int(nav_angles.shape[0]/ interleaves)
        nav_angle_samplingTime=-float(cp.mean(cp.diff(una)))
        sorted_signal = cp.asnumpy(nav_waveform[:,idx])

        
        sorted_signal = cp.reshape(sorted_signal,(nav_samples_times_channels,interleaves,numNavsPerStack)).transpose(0,2,1)
        sos = signal.butter(highpassorder, 5, btype='high', fs=interleaves, output='sos')
        filtered_signal = signal.sosfiltfilt(sos, sorted_signal)
        filtered_signal = cp.reshape(filtered_signal.transpose(0,2,1),(nav_waveform.shape))

        nav_waveform_filtered[:,idx] =filtered_signal
    
    return nav_waveform_filtered

def estimateCardiacOrRespiratoryGatingSignal(nav_data,nav_tstamp,nav_angles,filter_freqs=[0.8,0.85,2.0,2.1],filter_errors=[0.01,0.01,0.01],afilter=[True,0.1,7],bstar=False,gaussian_flag=False):
    '''
    Estimating  cardiac or respiratory gating signal.

    Parameters
    ----------
    
    nav_data : np.ndarray, 
        NAV/DC signal [channels,samples,nav_sample]

    nav_tstamp : float, 
        Acquisition time of NAV/DC signal

    nav_angles : np.ndarray, 
        DC  angle

    afilter : List[bool,highpassfreq,highpassorder], 
        Flag for angular trajectory filtering with the cutoff frequency and filter

    filter_freqs : list, 
        Kaiser bandpass parameter [0.8,0.85,2.0,2.1]

    Returns
    ------- 

    yfilt1 : np.ndarray,
        Initial ecg frequency estimated 

    samplingTime: np.float 
        sampling time 
    '''
    [number_channels,samples,nav_sample]=nav_data.shape
    t1 = time.time()
    # Get signal waveform (nav_waveform) with the shape [number_channels,nav_sample samples]
    nav_waveform = cp.abs(cufft(nav_data,axis=2))
    nav_waveform= cp.transpose(nav_waveform,(0,2,1)) 
    nav_waveform= cp.reshape(nav_waveform,(number_channels*nav_sample,samples))
    t2 = time.time()
    #Angular Filtering 
    if afilter[0]:
        nav_waveform = correctTrajectoryFluctuations(nav_waveform,nav_angles,highpassfreq=afilter[1],highpassorder=afilter[2],bstar=bstar)

    
    diff_nav_tstamp=cp.diff(nav_tstamp.squeeze())
    mean_samplingTime=cp.mean(diff_nav_tstamp)
    max_samplingTime=cp.max(diff_nav_tstamp)
    eprint(f' max: {max_samplingTime}  {mean_samplingTime} nav sampling time')
    if gaussian_flag:
        samplingTime = float(2.5 *max_samplingTime)
    else:
        if (2*mean_samplingTime)>max_samplingTime:
            samplingTime = float(2.5 *max_samplingTime) #mean_samplingTime #max_samplingTime
        else:
            samplingTime = float(2.5 *max_samplingTime)
    eprint(filter_freqs)
    # Bandpass filterations 
    t3 = time.time()
    bpfilter = kaiser_window.kaiser_window_generate(filter_freqs,filter_errors,'bandpass',1/(samplingTime*1e-3),nav_waveform.shape[1]) #DP

    filtered_signal = cufilterData(nav_waveform,bpfilter)
    t4 = time.time()
    filtered_signal = cp.asarray(np.reshape(filtered_signal,(number_channels,nav_sample,samples)))
    compressed_signal = cp.zeros((filtered_signal.shape[0],filtered_signal.shape[2]),dtype=complex)


    temp = (filtered_signal.transpose((0,2,1))).astype(cp.csingle)
    
    [u,s,v] = cp.linalg.svd(temp,full_matrices=False)
    t5 = time.time()
    compressed_signal = u[:,:,0]

    threshold = 0.98    
    C=corr2_coeff(compressed_signal.real,compressed_signal.real)
    G=(cp.abs(C)>threshold).astype(np.complex64)
    t6 = time.time()
    [ug,sg,vg]=cp.linalg.svd(G,full_matrices=False)
    
    ind_dom_motion = cp.argwhere(cp.abs((ug[:,0]))>0.1)
    dominantM = C[ind_dom_motion,ind_dom_motion]
    negInd = cp.argwhere(dominantM[:,0]<0)
    yfilt1 = compressed_signal[ind_dom_motion,:]
    yfilt1[negInd,:]*=-1
    
    yfilt1 = cp.asnumpy(yfilt1.mean(0).real)
    t7 = time.time()
    eprint(f'Execution time 1: {t2-t1} 2: {t3-t2} 3: {t4-t3} 4: {t5-t4} 5: {t6-t5} 6: {t7-t6} 7: {t7-t1}')
    return yfilt1, samplingTime/2.5

def stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional):

    eprint('Stable Binning')
    st = time.time()   

    if(bidirectional):
        input_diff = np.diff(selectedSig,axis=1)
        input_diff = np.concatenate(([np.asarray(input_diff[0,0])],input_diff[0,:]),axis=0)
    else:
        input_diff = np.ones(selectedSig.shape)
        
    I = np.argsort(selectedSig*np.sign(input_diff)).squeeze()
    sig = np.sort(selectedSig*np.sign(input_diff))

    lengthData = int(np.floor(acceptancePercent/100 * sig.shape[1]))
    st = time.time()   

    cp.cuda.runtime.deviceSynchronize()

    
    scp = cp.asarray(sig).squeeze()
    cp.cuda.runtime.deviceSynchronize()


    numWindows = 100
    stride = int(np.floor((sig.shape[1]-lengthData)/numWindows))
    indexer = cp.asarray(stride*np.arange(numWindows)[None, :] + np.arange(lengthData)[:, None])

    ss = cp.squeeze(scp[indexer])


    b = cp.polyfit(cp.asarray(range(0,lengthData)), ss, deg=1)
    slope = cp.asnumpy(b[0,:].squeeze())
    
    slope[slope==0]=[]
    V=np.min(slope)

    I2=np.argmin(slope)*stride
    

    
    Smin   = np.min(sig[0,range(I2,I2+lengthData-1)])
    Smax   = np.max(sig[0,range(I2,I2+lengthData-1)])
    
    indices = np.flatnonzero( (sig[0,:].squeeze() < Smax) & (sig[0,:].squeeze() > Smin) )
    timestamp = np.array(timestamp)
    accepted_times = timestamp[I[indices]]
    

    #plt.plot(range(0,sig.shape[1]),sig.squeeze(),range(I,I+lengthData),sig[0,range(I,I+lengthData)].squeeze(),"r-")
    cp.cuda.runtime.deviceSynchronize()

    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (stable binning function):', elapsed_time, 'seconds')
             
    return Smin, Smax, [I[indices]], [accepted_times]   

def binning_div(selectedSig,timestamp,numBins,evenBins,bidirectional):
    
    eprint('Binning Division')
    st = time.time()   
    
    indices = []
    accepted_times = []
    if(bidirectional):
        input_diff = np.diff(selectedSig,axis=1)
        input_diff = np.concatenate(([np.asarray(input_diff[0,0])],input_diff[0,:]),axis=0)
    else:
        input_diff = np.ones(selectedSig.shape)
        
    I = np.argsort(selectedSig*np.sign(input_diff))
    sig = np.sort(selectedSig*np.sign(input_diff))
    
    n95 = np.percentile(sig,99)
    n05 = np.percentile(sig,1)
    
    
    if(evenBins):

        low_idx = np.argmin(np.abs(sig-n05))
        high_idx = np.argmin(np.abs(sig-n95))
        
        delta = np.floor((high_idx-low_idx)/(numBins))
        
        indices_sorted_min = np.floor(low_idx + delta*range(0,numBins))
        indices_sorted_max = np.floor(low_idx + delta*range(1,numBins+1))

    else:
        delta = (n95-n05)/numBins
        min_limits = n05 + delta*range(0,numBins)
        max_limits = n05 + delta*range(1,numBins+1)
        
        indices_sorted_min = []
        indices_sorted_max = []
        
        for ii in range(0,numBins):
            indices_sorted_min.append(max(np.flatnonzero(sig<min_limits[ii])))
            indices_sorted_max.append(max(np.flatnonzero(sig<max_limits[ii])))
    
    timestamp = np.array(timestamp)
    I = I.squeeze()
    for ii in range(0,numBins):
        #eprint(int(indices_sorted_max[ii]))
        indices.append(I[np.array(range(int(indices_sorted_min[ii]),int(indices_sorted_max[ii])))])
        accepted_times.append(timestamp[indices[ii]])

    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (Binning Div function):', elapsed_time, 'seconds')
    return indices, accepted_times
    
def binning(selectedSig, timestamp, acceptancePercent, bidirectional, do_stable_binning, evenBins, numBins):
    eprint("Binning")
    st = time.time()   

    selectedSig = selectedSig-np.min(selectedSig)
    selectedSig = selectedSig/np.percentile(np.abs(np.sort(selectedSig)),99)

    min_resp, max_resp, indices, accepted_times = stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional)
    # Flip sign if the most stable phase is >0.5 of the scaled signal
    selectedSig = selectedSig*np.power(-1,(max_resp+min_resp)/2 >0.5)+1*((max_resp+min_resp)/2 >0.5) 
    
    if(do_stable_binning):
        #% Do it again in case of a flip
        min_resp, max_resp, indices, atimes = stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional) 
    else:
        indices, atimes = binning_div(selectedSig,timestamp,numBins,evenBins,bidirectional)
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (binning function):', elapsed_time, 'seconds')
        
    return atimes,indices

def get_idx_to_send(data_timestamps,acceptedTimes,sampling_time):
    """
    data_timestamps: np.array
    acceptedTimes : list cupy.array
    """
    data_timestamps =data_timestamps.squeeze()[:,None]
    sampling_diff=sampling_time/2
    idx_to_send=[]
    maxSize=0
    for ii in range(len(acceptedTimes)):
        accepted_times=acceptedTimes[ii].squeeze()[None,:]
        try:
            eprint(f'GPU Binning idx:{ii}')
            diff_t=cp.abs(cp.asarray(data_timestamps)-cp.asarray(accepted_times))
            idxs_ii=((diff_t<=sampling_diff).sum(1).ravel().nonzero()[0]).tolist()
        except:
            eprint("GPUs couldnt do idx to send moving to cpu")
            diff_t=np.abs(data_timestamps-accepted_times)
            idxs_ii=((diff_t<=sampling_diff).sum(1).ravel().nonzero()[0]).tolist()
        idxs_ii.insert(0,len(idxs_ii))
        idx_to_send.append(np.array(idxs_ii))
        maxSize=max([len(idx_to_send[ii]),maxSize])
    return idx_to_send,maxSize

def create_ismrmrd_image(data, reference, field_of_view, index):
        return mrd.image.Image.from_array(
            data,
            acquisition=reference,
            image_index=index,
            image_type=mrd.IMTYPE_MAGNITUDE,
            field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z),
            transpose=False
        )
        
def BinningOnTheFlyGadget(connection):
    eprint("Ready to do some Binning in Python")

    #connection.filter(lambda acq: acq)
    connection.filter(mrd.Acquisition)
    
    params = parse_params(connection.config)

    params_init = parse_params(connection.config)
    params={'binningPercent':40,
            'numBins':6,
            'useDC':False,
            'stableBinning': False,
            'evenbins':True,
            'bidirectional':False,
            'warm_up': False,
            'phantom': False,
            'angular_filteration':False,
            "gaussian":False,
            "bstar":False,
            "samples":40,
            }

    BPfilter_freqs=   [0.08,0.1,0.45,0.50]
    boolean_keys=['useDC','stableBinning','evenbins','bidirectional','warm_up','phantom','angular_filteration','gaussian','bstar']
    str_keys=[]
    int_keys=['binningPercent','numBins','samples']
    params=read_params(params_init,params_ref=params,boolean_keys=boolean_keys,str_keys=str_keys,int_keys=int_keys)

    bidirectional = params['bidirectional']
    evenbins = params['evenbins']
    do_stable_binning = params['stableBinning']
    numBins = params['numBins']
    useDC = params['useDC']
    binningPercent = params['binningPercent']

    gaussian_flag=params['gaussian']
    bstar_flag=params['bstar']
    eprint("bidirectional: ", bidirectional)
    eprint("evenbins: ", evenbins)
    eprint("stableBinning: ", do_stable_binning)
    eprint("Numbins: ", numBins)
    eprint("useDC: ", useDC)
    eprint("Binning %: ", binningPercent)
    count = 0
    firstacq=0
    navangles = []
    acq_tstamp = []
    nav_data    = []
    nav_tstamp  = []
    nav_indices = []
    data_indices = []
    kencode_step = []
    mrd_header = connection.header
    
    eprint("zencode/2:", int(mrd_header.encoding[0].encodedSpace.matrixSize.z/2))
    
    field_of_view = mrd_header.encoding[0].reconSpace.fieldOfView_mm

    if params['warm_up']:
        st = time.time()
        dx = cp.random.rand(2,2).astype(cp.csingle)
        u,s,v = cp.linalg.svd(dx,full_matrices=False)
        
        input = cp.asarray(cp.random.rand(1,5500))
        filterIn = cp.asarray(cp.random.rand(5500))

        cp.convolve(input[0,:].squeeze(),cufftshift(cufft(filterIn)),'same')
        ndimage.convolve1d(input[0,:].squeeze(),cufftshift(cufft(filterIn)),axis=0)

        ss = cp.zeros((int(5500*0.4),5500-int(5500*0.4)))
        cp.polyfit(cp.asarray(range(0,int(5500*0.4))),ss , deg=1)  
        
        et = time.time()
        elapsed_time = et - st
        eprint('Execution time (warmups):', elapsed_time, 'seconds')
        del ss
        cp._default_memory_pool.free_all_blocks()
    st = time.time()

    acquisition_0 = []
    for acq in connection:
        # Navigator acquisition
        if (acq.isFlagSet(mrd.ACQ_IS_RTFEEDBACK_DATA or mrd.ACQ_IS_HPFEEDBACK_DATA or mrd.ACQ_IS_NAVIGATION_DATA)):
            # Use Navigator
            if not useDC:
                nav_data.append(np.array(acq.data))
                nav_tstamp.append(acq.acquisition_time_stamp)
                nav_indices.append(count)
        else :
            # Data acquisition
            if useDC:
                if(acq.idx.kspace_encode_step_2 == int(mrd_header.encoding[0].encodedSpace.matrixSize.z/2) or bstar_flag):
                    nav_data.append(np.array(acq.data[:,0:1]))
                    nav_tstamp.append(acq.acquisition_time_stamp)
                    nav_indices.append(count)
            if not bstar_flag:
                navangles.append(180*cmath.phase(complex(acq.traj[25,0],acq.traj[25,1]))/math.pi)
            acq_tstamp.append(acq.acquisition_time_stamp)
            data_indices.append(count)
            kencode_step.append(acq.idx.kspace_encode_step_1) 
            if(len(acquisition_0)<1):
                acquisition_0.append(acq)
            connection.send(acq)                
        count+=1
        # Buffer not working with VIBE
        if acq.user_int[0] or acq.isFlagSet(mrd.ACQ_LAST_IN_MEASUREMENT):
            print("LAST MEASUREMENT")
            samples=len(nav_indices)
            [number_channels,nav_sample]=nav_data[0].shape
            nav_data_copy=cp.concatenate(cp.asarray(nav_data),axis=1).reshape((number_channels,samples,nav_sample)).copy()
            nav_tstamp_copy=np.array(nav_tstamp).copy()
            acq_tstamp_copy = np.array(acq_tstamp).copy()
            nav_angles_copy=[]
            if gaussian_flag:
                print("3D cine: 1 sample every 3 samples")
                nav_data_copy=nav_data_copy[:,1::3,:]
                nav_tstamp_copy=nav_tstamp_copy[1::3]
            if bstar_flag:
                nav_data_copy=nav_data_copy[:,::params['samples'],:]
                nav_tstamp_copy=nav_tstamp_copy[::params['samples']]
            # Not working
            if params['angular_filteration']:
                traj_angles = cp.asarray(navangles)
                temp_indices = cp.asarray(nav_indices)
                temp_indices = temp_indices[1:None]-1
                kencode_step = cp.asarray(kencode_step)

                kstep_nav   = cp.concatenate((cp.asarray([kencode_step[0]]),kencode_step[temp_indices]),axis=0)
                angles_sorted = cp.zeros(cp.unique(traj_angles).shape)
                angles_sorted[kencode_step] = traj_angles
                navangles    = angles_sorted[kstep_nav]
                nav_angles_copy = navangles.copy()

            print(f'DATA {nav_data_copy.shape}')
            print(f'Tstamp {nav_tstamp_copy.shape}')
            st = time.time()   
            respiratory_waveform, samplingTime = estimateCardiacOrRespiratoryGatingSignal(nav_data_copy,nav_tstamp_copy,nav_angles_copy,filter_freqs=BPfilter_freqs,filter_errors=[0.001,0.001,0.001],afilter = [params["angular_filteration"],0.01,5],bstar=bstar_flag,gaussian_flag=gaussian_flag) 
            cp.cuda.runtime.deviceSynchronize()

            et = time.time()
            elapsed_time = et - st
            eprint('Execution time (GatingSignal):', elapsed_time, 'seconds')

            acceptedTimes,idx_acceptedTimes = binning(respiratory_waveform,nav_tstamp_copy,binningPercent,bidirectional, do_stable_binning, evenbins, numBins )
            if bstar_flag and params['samples']==1:
                idx_to_send=[]
                maxSize=0
                for idx_a in idx_acceptedTimes:
                    idx_a.sort()
                    idx_to_send.append(np.concatenate([[len(idx_a)],idx_a]))
                    maxSize=max([len(idx_to_send[-1]),maxSize])
                    eprint(idx_to_send[-1][:40])
            else:
                idx_to_send,maxSize=get_idx_to_send(acq_tstamp_copy,acceptedTimes, samplingTime)
            
            [Nidx,Nr,Nc,Nechoes]=[maxSize,len(acceptedTimes),1,1]
            data = np.zeros([Nidx,Nr,Nc,Nechoes])
            for ii in range(0,len(acceptedTimes)):
                data[range(0,len(idx_to_send[ii])),0,ii,0] = idx_to_send[ii].squeeze()
            
            image = create_ismrmrd_image(data, acquisition_0[0], field_of_view, 0)
            connection.send(image)

    eprint("Total Acq:",count)
    et = time.time()
    # # get the execution time
    elapsed_time = et - st
    eprint('Execution time:', elapsed_time, 'seconds')      
        
if __name__ == "__main__":
    gadgetron.external.listen(20000,BinningOnTheFlyGadget)
