
import gadgetron
import ismrmrd as mrd

import numpy as np
import cupy as cp
import sys
import cmath
import math
import time
import shelve
from scipy.signal import butter, filtfilt
from scipy import signal

# From PD's binning
from scipy.signal import find_peaks # type: ignore
from operator import itemgetter
from itertools import groupby
from typing import List, Dict, Tuple

import cupyx.scipy

from scipy.fft import fft,ifft,fftshift,ifftshift
from cupyx.scipy.fft import fft as cufft
from cupyx.scipy.fft import ifft as cuifft
from cupyx.scipy.fft import fftshift as cufftshift
from cupyx.scipy.fft import ifftshift as cuifftshift
from cupyx.scipy import ndimage
from cupy.linalg import svd as cusvd

import importlib
import inspect
#import matplotlib.pyplot as plt
import kaiser_window

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, cp.asnumpy(data))
    return y

def cufilterData(input, filter):

    
    cuin = cp.asarray(input) 
    if(len(cuin.shape)>2):
        cuin = cp.reshape(cuin,(cuin.shape[0]*cuin.shape[1],cuin.shape[2]))
    filter = cp.asarray(filter)
    
    concat_arr = cp.concatenate(((cuin[:,::-1]),cuin[:,:],(cuin[:,::-1])),axis=1).astype(cp.complex64)
    st = time.time()
    
    temp = cp.zeros([concat_arr.shape[0],max(concat_arr.shape[1],filter.shape[0])],cp.complex64)

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

def correctTrajectoryFluctuations(data_array,navangles):
    na  = cp.asnumpy(navangles)
    una = np.unique(na)
    idx = np.argsort(-1*na)
    interleaves = len(una)

    factor = (int(math.ceil(data_array.shape[1] / interleaves)) % int(interleaves)) - round(data_array.shape[1] / interleaves)
    # print(na[idx[0:20]])
    # print(len(na))
    # print(np.min(na))
    # print(np.max(na))
    #factor = int(np.prod(data_array.shape) / (data_array.shape[0] * (int(round(data_array.shape[1] / interleaves)) % int(interleaves))) - interleaves)
    if (factor < 0):
        factor = 0

    
    nav_samplingTime = 0
    numNavsPerStack = int(len(navangles) / interleaves)
    # print("numNavsPerStack:",numNavsPerStack)
    # print("len(navangles):",len(navangles))
    # print("interleaves:",interleaves)

    for  ii in range(numNavsPerStack + 1,len(na),numNavsPerStack): # this hardcoding is a potential bug
            nav_samplingTime += na[idx[ii - numNavsPerStack]] - na[idx[ii]] 
    
    # print("nav_samplingTime:",nav_samplingTime)
    # print("data_array:",data_array.shape)

    nav_samplingTime = nav_samplingTime/interleaves
    # print("nav_samplingTime:",nav_samplingTime)
   
    sorted_signal = data_array[:,idx]
    
   
    sorted_signal = cp.reshape(sorted_signal,(data_array.shape[0],int(data_array.shape[1]/interleaves),interleaves+factor))
    
    # print("sorted_signal.shape:",sorted_signal.shape)
    # plt.plot(cp.asnumpy(sorted_signal[0,0,:]).squeeze())
    # plt.show()
    # print("sorted_signal.shape:",sorted_signal.shape)
    #filter = kaiser_window.kaiser_window_generate(0.1,[0.00001,0.00001],'highpass',abs(1/nav_samplingTime),sorted_signal.shape[2])
    #filter = kaiser_window.kaiser_window_generate([0.0001,0.001,abs(1/(nav_samplingTime))-0.001*abs(1/(nav_samplingTime)),abs(1/(nav_samplingTime))],[0.01,0.01,0.01],'bandpass',abs(1/(nav_samplingTime)),sorted_signal.shape[2])
    
    filtered_signal = butter_highpass_filter(sorted_signal.squeeze(), 0.01, abs(1/(nav_samplingTime)), order=5)
    #filtered_signal = cufilterData(sorted_signal.squeeze(),filter)
    # plt.plot(cp.asnumpy(filtered_signal[0,:]).squeeze())
    # plt.show()
    # plt.plot(np.linspace(-1*abs(1/nav_samplingTime),abs(1/nav_samplingTime),math.floor(filtered_signal.shape[2])),ifftshift(ifft(ifftshift(cp.asnumpy(filtered_signal[0,0,:]).squeeze()))))
    # plt.show()
    #filtered_signal = np.transpose(filtered_signal,(1,2,0))

    
    filtered_signalX = cp.reshape(filtered_signal,(data_array.shape[0],data_array.shape[1]))
    filtered_signal = np.copy(filtered_signalX)
    filtered_signal[:,idx] = filtered_signalX
    
    return filtered_signal

def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def findNavAngle(navTimestamps, acqTimestamps, traj_angles):
    orderedInd = np.argsort(acqTimestamps)
    navAngles = []
    
    for jj in range(0,len(navTimestamps)):
        tstamp = navTimestamps[jj]
        size_nA = len(navAngles)

        for ii in range(0,len(acqTimestamps)):
            if (ii > 0 and (int(tstamp) - int(acqTimestamps[orderedInd[ii]])) <= 0 and (int(tstamp) - int(acqTimestamps[orderedInd[ii - 1]])) >= 0):
                navAngles.append(traj_angles[orderedInd[ii - 1]])
                break
            else: 
                if ((int(tstamp) - int(acqTimestamps[orderedInd[ii]])) <= 0 and ii == 0 and jj == 0):
                    navAngles.append(traj_angles[orderedInd[ii]])
                    break
                
    return navAngles

def findNavAngle_fast(traj_angles):
    orderedInd = np.argsort(acqTimestamps)
    navAngles = []
    
    for jj in range(0,len(navTimestamps)):
        tstamp = navTimestamps[jj]
        size_nA = len(navAngles)

        for ii in range(0,len(acqTimestamps)):
            if (ii > 0 and (int(tstamp) - int(acqTimestamps[orderedInd[ii]])) <= 0 and (int(tstamp) - int(acqTimestamps[orderedInd[ii - 1]])) >= 0):
                navAngles.append(traj_angles[orderedInd[ii - 1]])
                break
            else: 
                if ((int(tstamp) - int(acqTimestamps[orderedInd[ii]])) <= 0 and ii == 0 and jj == 0):
                    navAngles.append(traj_angles[orderedInd[ii]])
                    break
                
    return navAngles


def estimateGatingSignal(nav_data,nav_tstamp,navangles,acq_tstamp):

    # Gaussian sub-sample test (should be moved upstream..)
    eprint("Cropping nav_tstamp and nav_data every 3rd sample for gaussian dc signal")
    nav_tstamp=nav_tstamp[1::3]
    nav_data=nav_data[1::3]

    data_array = cp.asarray(np.concatenate(nav_data,axis=1))    
    data_array= cp.reshape(data_array,(nav_data[0].shape[0],len(nav_data),nav_data[0].shape[1]))

    number_channels = data_array.shape[0]
    data_array = cp.abs(cufft(data_array,axis=2))

    data_array= cp.transpose(data_array,(0,2,1))
    data_array= cp.reshape(data_array,(nav_data[0].shape[0]*nav_data[0].shape[1],len(nav_data)))
    #data_array= np.transpose(data_array,(1,0))

    # Trajectory Correction
    # data_array = correctTrajectoryFluctuations(data_array,navangles)

    # Bandpass filterations
    samplingTime = 0
    for ii in range(1,len(nav_tstamp)):
        samplingTime += float(nav_tstamp[ii] - nav_tstamp[ii - 1]) * 2.5
    samplingTime = samplingTime/len(nav_tstamp)
    
    max_sampTime = 0
    for ii in range(1,len(nav_tstamp)):
        if(max_sampTime < (nav_tstamp[ii]-nav_tstamp[ii - 1])*2.5):
            max_sampTime = (nav_tstamp[ii] - nav_tstamp[ii - 1]) * 2.5
    if(2*samplingTime > max_sampTime):
        samplingTime = max_sampTime
    eprint("EstimateGatingSignal: Sampling Time: ", samplingTime)

    # add flag for max sampling time? 
    samplingTime = max_sampTime # fixing for gaussian dist

    bpfilter = kaiser_window.kaiser_window_generate([0.08,0.1,0.45,0.50],[0.001,0.001,0.001],'bandpass',1/(samplingTime*1e-3),data_array.shape[1])

    filtered_signal = cufilterData(data_array,bpfilter)

    filtered_signal = cp.asarray(np.reshape(filtered_signal,(nav_data[0].shape[0],nav_data[0].shape[1],len(nav_data))))
    compressed_signal = cp.zeros((filtered_signal.shape[0],filtered_signal.shape[2]),dtype=complex)


    temp = (filtered_signal.transpose((0,2,1))).astype(cp.csingle)
    [u,s,v] = cp.linalg.svd(temp,full_matrices=False)
    compressed_signal = u[:,:,0]


    C=cp.zeros((compressed_signal.shape[0],compressed_signal.shape[0]),dtype=complex)
    G=cp.zeros((compressed_signal.shape[0],compressed_signal.shape[0]),dtype=complex)
    

    threshold = 0.98
    for ii in range(0,compressed_signal.shape[0]):
        for jj in range(0,compressed_signal.shape[0]):
            C[ii,jj]= corr2(cp.real(compressed_signal[ii,:]).squeeze(),cp.real(compressed_signal[jj,:]).squeeze())
            G[ii,jj]= (cp.abs(C[ii,jj])>threshold)
    

    [ug,sg,vg]=cp.linalg.svd(G,full_matrices=False)
    ind_dom_motion = cp.argwhere(cp.abs(cp.sum(ug[:,cp.argwhere(cp.max(cp.diag(sg))==cp.diag(sg))],axis=1))>0.1)
    ind_dom_motion = cp.argwhere(cp.abs((ug[:,0]))>0.1)


    dominantM = C[ind_dom_motion,ind_dom_motion]

    negInd = ind_dom_motion[cp.argwhere(dominantM[:,0]<0)]
    negInd = cp.argwhere(dominantM[:,0]<0)
    yfilt1 = compressed_signal[ind_dom_motion,:]
   

    for ii in range(0,negInd.shape[0]):
        maxC= np.max(yfilt1[negInd(ii),:])
        yfilt1[negInd(ii),:] = yfilt1[negInd(ii),:]*-1
    
    yfilt1 = cp.asnumpy(np.real(np.mean(yfilt1,axis=0)))
    
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
    #print(I.shape)
    accept_perc = acceptancePercent

    lengthData = int(np.floor(acceptancePercent/100 * sig.shape[1]))
    slope = np.zeros(sig.shape[1]-lengthData)

    st = time.time()   

    ss = cp.zeros((lengthData,sig.shape[1]-lengthData))

    cp.cuda.runtime.deviceSynchronize()

    
    scp = cp.asarray(sig).squeeze()
    cp.cuda.runtime.deviceSynchronize()


    numWindows = 100
    stride = int(np.floor((sig.shape[1]-lengthData)/numWindows))
    indexer = cp.asarray(stride*np.arange(numWindows)[None, :] + np.arange(lengthData)[:, None])

    ss = cp.squeeze(scp[indexer])


    b = cp.polyfit(cp.asarray(range(0,lengthData)), ss, deg=1)
        # for ii in range(0,sig.shape[1]-lengthData):
    #     b = cp.polyfit(cp.asarray(range(0,lengthData)), ss[0,range(ii,ii+lengthData)], deg=1)
    #     slope[ii] = b[0]
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
             
    return I, Smin, Smax, indices, accepted_times
    
    
    
def binning_div(selectedSig,timestamp,numCBins,evenBins,bidirectional):
    
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
        
        delta = np.floor((high_idx-low_idx)/(numCBins))
        
        indices_sorted_min = np.floor(low_idx + delta*range(0,numCBins))
        indices_sorted_max = np.floor(low_idx + delta*range(1,numCBins+1))

    else:
        delta = (n95-n05)/numCBins
        min_limits = n05 + delta*range(0,numCBins)
        max_limits = n05 + delta*range(1,numCBins+1)
        
        indices_sorted_min = []
        indices_sorted_max = []
        
        for ii in range(0,numCBins):
            indices_sorted_min.append(max(np.flatnonzero(sig<min_limits[ii])))
            indices_sorted_max.append(max(np.flatnonzero(sig<max_limits[ii])))
    
    timestamp = np.array(timestamp)
    I = I.squeeze()
    for ii in range(0,numCBins):
        #eprint(int(indices_sorted_max[ii]))
        indices.append(I[np.array(range(int(indices_sorted_min[ii]),int(indices_sorted_max[ii])))])
        accepted_times.append(timestamp[indices[ii]])

    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (Binning Div function):', elapsed_time, 'seconds')
    return indices, accepted_times
        
        

def cardiacbinning(selectedSig:np.ndarray,samplingTime:np.float64,numCBins:int,ecg_freq:np.float64,evenbins:bool=False,phantomflag:bool=False,arrythmia_detection:bool=False) -> Tuple[List[List],np.float64]:
    """
    Calculating cardiac bins based on the navigator/DC signal. 

    Parameters
    ----------
    
    selectedSig : np.ndarray, 
        NAV/DC signal 

    samplingTime : float, 
        sampling time 

    numCBins : int, 
        number of bins 

    ecg_freq : float, 
        initial ecg frequency estimated 

    evenbins : boolean, 
        Flag to obtain uniform bins (default: False)

    phantomflag : boolean, 
        Flag for phantom data (simplified binning: mod(sample index,numCBins)) (default: False)

    arrythmia_detection : boolean, 
        Remove RR vectors which are too long (>1.5(ecg_freq_initial) (default: False)   

    Returns
    -------

    bins : List[List], List of binned indexes : 
        - Length of the list is equal to the number of bins    

    final_ecg_freq : np.float64,
        ecg frequency estimated 
    """


    # Selected the maximum in frequency for having a initial guess of cardiac frequency   
    fs=1/float(2.5*samplingTime/1000)
    n_samples=cp.shape(selectedSig)[1]

    #Minimum distance betwwen peak 
    min_distance=round(0.75* (fs/ecg_freq))
    peaks, _ = find_peaks(selectedSig.flatten(), distance=min_distance)
    #binnings
    bins=[[] for n in range(numCBins)]
    final_ecg_freq=ecg_freq
    if phantomflag:
        RR_vec=np.arange(n_samples)
        bin_data_label=RR_vec % (numCBins)
        RR_seq=list(map(list, zip(RR_vec.tolist(), bin_data_label.tolist())))
        RR_seq.sort(key=itemgetter(1)) # Not necessary now 
        bins=[[x for x,y in data] for (key, data) in groupby(RR_seq, itemgetter(1))]

    else:
        if len(peaks)>=2:
            RR_ecg=list()
            peak_list=list(zip(peaks[:-1],peaks[1:],np.diff(peaks).tolist()))
            if arrythmia_detection:
                ln_p=len(peak_list)
                peak_list=[(peak_start,peak_stop,RR_vec) for peak_start,peak_stop,RR_vec in peak_list if RR_vec <1.5*(fs/ecg_freq)]
                eprint('Arrythmia reject (%)...')
                eprint(100*(ln_p-len(peak_list))/ln_p)
                #eprint(np.median(np.array(peak_list),axis=0)[2])
                #eprint((fs/ecg_freq))
            for peak_start,peak_stop,RR_vec in peak_list:
                bin_data_label=np.arange(RR_vec)//(RR_vec/numCBins)
                RR_seq=list(map(list, zip(np.arange(peak_start,peak_stop).tolist(), bin_data_label.tolist())))
                RR_seq.sort(key=itemgetter(1)) # Not necessary now 
                RR_peak_bins=[[x for x,y in data] for (key, data) in groupby(RR_seq, itemgetter(1))]
                bins=list(map(lambda x, y:x + y, bins, RR_peak_bins))
                RR_ecg.append(fs/RR_vec)
            final_ecg_freq=np.mean(np.array(RR_ecg))
    eprint('ECG FREQ INIT : {} ECG FREQ FINAL : {}'.format(ecg_freq,final_ecg_freq))
    if evenbins:
        len_bins=[len(bins[n]) for n in range(numCBins)]
        min_len=min(len_bins)
        bins=[bins[n][:min_len] for n in range(numCBins)]

    return bins,final_ecg_freq                                         
        
   
    
def binning(selectedSig, timestamp, acceptancePercent, bidirectional, do_stable_binning, evenBins, numCBins):
    eprint("Binning")
    st = time.time()   

    selectedSig = selectedSig-np.min(selectedSig)
    selectedSig = selectedSig/np.percentile(np.abs(np.sort(selectedSig)),99)


    if(do_stable_binning):
        I, Smin, Smax, indices, accepted_times = stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional)
        
        max_resp = Smax
        min_resp = Smin
        
        # Flip sign if the most stable phase is >0.5 of the scaled signal
        selectedSig = selectedSig*np.power(-1,(max_resp+min_resp)/2 >0.5)+1*((max_resp+min_resp)/2 >0.5) 
        
        #% Do it again in case of a flip
        I, Smin, Smax, indices, accepted_times = stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional) 
        
        max_resp = Smax
        min_resp = Smin
        
        atimes =[]
        atimes.append(accepted_times)
    else:
        cp.cuda.runtime.deviceSynchronize()

        I, min_resp, max_resp, ind, at = stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional)
        cp.cuda.runtime.deviceSynchronize()

        # Flip sign if the most stable phase is >0.5 of the scaled signal
        selectedSig = selectedSig*np.power(-1,(max_resp+min_resp)/2 > 0.5)+1*((max_resp+min_resp)/2 > 0.5) 
        
        
        indices, atimes = binning_div(selectedSig,timestamp,numCBins,evenBins,bidirectional)
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (binning function):', elapsed_time, 'seconds')
        
    return atimes
    
    
# idx_to_send_nav{1} = get_idx_to_send(header.dataacq_time_stamp,nav_gating.accepted_times,nav_gating.sampling_time)
def get_idx_to_send(data_timestamps,accepted_times,sampling_time):
    idx_to_send =[]
    for ii in range(0,len(data_timestamps)):
        if(np.shape(np.flatnonzero(abs((data_timestamps[ii])-(accepted_times.squeeze())) <= sampling_time/2)>0)[0]):
            idx_to_send.append(ii)
        #    eprint(np.shape(np.flatnonzero(abs((data_timestamps[ii])-(accepted_times.squeeze())) <= sampling_time/2)>0)[0])
    return np.array(idx_to_send)

def get_idx_to_send2(data_timestamps,accepted_times,sampling_time):
    st = time.time()
    idx_to_send =[]
    data_timestamps = cp.expand_dims(cp.asarray(data_timestamps),axis=1)
    accepted_times  = cp.expand_dims(cp.asarray(accepted_times),axis=0)


    ar2 = data_timestamps - accepted_times
    x = cp.flatnonzero(cp.sum(abs(ar2) <= sampling_time/2,axis=1))
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (idx to send):', elapsed_time, 'seconds')
    
    return cp.asnumpy(x)

def get_idx_to_send2_np(data_timestamps,accepted_times,sampling_time):
    st = time.time()
    idx_to_send =[]
    data_timestamps = np.expand_dims((data_timestamps),axis=1)
    accepted_times  = np.expand_dims((accepted_times),axis=0)


    ar2 = data_timestamps - accepted_times
    x = np.flatnonzero(np.sum(abs(ar2) <= sampling_time/2,axis=1))
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (idx to send) Numpy:', elapsed_time, 'seconds')
    
    return (x)

def create_ismrmrd_image(data, reference, field_of_view, index):
        return mrd.image.Image.from_array(
            data,
            acquisition=reference,
            image_index=index,
            image_type=mrd.IMTYPE_MAGNITUDE,
            field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z),
            transpose=False
        )
        
def SmartNavGadget(connection):
    eprint("SmartNav in Python")

    #connection.filter(lambda acq: acq)
    connection.filter(mrd.Acquisition)
    
    # ========================================
    # configuration parameters
    # =======================================
    params = _parse_params(connection.config)

    # Cardiac Binning Params
    if "numCBins" in params:
        numCBins = int(params["numCBins"])
    else:    
        numCBins = 10

    # Respiratory Binning Params
    if "numRBins" in params:
        numRBins = int(params["numRBins"])
    else:    
        numRBins = 8

    # Fixed to "useDC"
    # if "useDC" in params:
    #     if params["useDC"] == 'True':
    #         # 2D spiral cardiac binning available
    #         useDC = True
    #     else:
    #         useDC = False    
    # else:    
    #     # Default to ECG waveform
    useDC = True
           
    if "evenRBins" in params:
        if params["evenRBins"] == 'True':
            evenRBins = True
        else:
            evenRBins = False
    else:
        evenRBins = True
    
    
    eprint("Cardiac numCBins: ", numCBins)
    eprint("Resp useDC: ", useDC)
    eprint("Resp numRBins: ", numRBins)
    eprint("Resp evenBins: ", evenRBins)
    
    # ========================================
    # Accumulate nav data and ECG
    # =======================================

    count = 0
    
    # ECG variables
    ecg_data = []
    acq_tstamp = []

    # Respiratory variables
    navangles = []  
    acq_tstamp = []
    nav_data    = []
    nav_tstamp  = []
    nav_indices = []
    kencode_step = []

    st = time.time()
    acquisition_0 = []

    for acq in connection:
        count= count+1

        # grab first acquisition to create mrd-image later
        if(len(acquisition_0)<1):
                    acquisition_0.append(acq)
        
        # RESP // Collect "DC" respiratory data 
        if(acq.idx.kspace_encode_step_2 == int(connection.header.encoding[0].encodedSpace.matrixSize.z/2)):
            # ideally - we're taking samples 0-19 (or better 5:15) of uncropped data
            nav_data.append(np.array(acq.data[:,0:1])) # nav_data.append(np.array(acq.data[:,6:15]))
            nav_tstamp.append(acq.acquisition_time_stamp)
            nav_indices.append(count-1)    
        
        # make entire data-acq length vector of angles for filtering
        kencode_step.append(acq.idx.kspace_encode_step_1)
        navangles.append(180*cmath.phase(complex(acq.traj[25,0],acq.traj[25,1]))/math.pi)

        # ECG // only grab time stamps and ecg-data (first line only)
        acq_tstamp.append(acq.acquisition_time_stamp)
        ecg_data.append(acq.physiology_time_stamp[0])
        
        # send acquisition on
        connection.send(acq)

    # Performance : get the execution time
    eprint("Total Acq:",count)
    et = time.time()
    
    elapsed_time = et - st
    eprint('Execution time:', elapsed_time, 'seconds')

    eprint(np.shape(ecg_data))
    if len(ecg_data)==0:
        return ValueError('Navigation Data is empty and use DC set to False')
    
    # ========================================
    # RESP
    # ========================================
    traj_angles = cp.asarray(navangles)
    temp_indices = cp.asarray(nav_indices)
    temp_indices = temp_indices[1:None]-1
    kencode_step = cp.asarray(kencode_step)

    kstep_nav   = cp.concatenate((cp.asarray([kencode_step[0]]),kencode_step[temp_indices]),axis=0)
    angles_sorted = cp.zeros(cp.unique(traj_angles).shape)
    angles_sorted[kencode_step] = traj_angles
    navangles    = angles_sorted[kstep_nav]
    
    nav_data_copy   = nav_data.copy()
    nav_tstamp_copy = nav_tstamp.copy()
    nav_angles_copy = navangles.copy()
    acq_tstamp_copy = acq_tstamp.copy()

    st = time.time()   
    respiratory_waveform, samplingTime = estimateGatingSignal(nav_data_copy,nav_tstamp_copy,nav_angles_copy,acq_tstamp_copy) 
    cp.cuda.runtime.deviceSynchronize()

    # hack temp for Gaussian madness -> Do something smarter as they also get cropped in estimateGatingSignal
    nav_tstamp = nav_tstamp[1::3]
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (GatingSignal):', elapsed_time, 'seconds')
    st = time.time()   

    # Hard coded! evenbins = bidirectional, StableBinning = False, EvenBins = True
    acceptedTimes = binning(respiratory_waveform,nav_tstamp,40, True, False, True, numRBins )

    # ========================================
    # ECG
    # ========================================
    ecg_nav_data=2.5*np.array(ecg_data).astype(np.float32)
    ecg_nav_data=ecg_nav_data[np.newaxis,:] # prep for ecg binning function 
    c_samplingTime = float(connection.header.sequenceParameters.TR[0])
    acq_tstamp = np.array( acq_tstamp )

    #Minimum distance betwwen peak 
    maxBPM          = 200 # arbitrary maximum is 200 bpm
    minRR           = 60./(maxBPM*60/60e3) # ms
    minPeakHeight   = minRR
    peaks, _ = find_peaks(ecg_nav_data.flatten(), prominence=minPeakHeight)
    peak_val = ecg_nav_data[0][peaks]
    RR_interval = np.mean(peak_val)
    ecg_freq = float(1000/ RR_interval) 
    
    # Hard coded! evenbins = False, phantomflag = False, arrythmia_detection = True
    bins_index,ecg_freq_final=cardiacbinning(ecg_nav_data,c_samplingTime/2.5,numCBins,ecg_freq,False,False,True)
    Index=np.arange(acq_tstamp.shape[0])

    # ========================================
    # 2D Motion Vector
    # ========================================

    number_of_sets = 1 # hard-coding for the moment, compatiability for phase-contrast
    # cardiac_waveform_smooth = ecg_nav_data # export for inspection // temp

    # Resp accepted times > idx (Pierre's flow version)
    # acceptedTimes=[[] for n in range(numRBins*number_of_sets)]

    resp_bins_index = []
    
    for ii in range(len(acceptedTimes)):
        temp = get_idx_to_send2(acq_tstamp,acceptedTimes[ii], samplingTime)
        resp_bins_index.append(np.concatenate(([np.array(temp.shape[0])],cp.asnumpy(temp))))

    maxSize = 0 
    idx_to_send_2D = []
    
    for idx_c in bins_index:
        for idx_r in resp_bins_index:
            temp_r_c = np.intersect1d(idx_r,idx_c)
            idx_to_send_2D.append( np.concatenate( ([temp_r_c.shape[0]],temp_r_c)))
            #idx_to_send_2D.append(np.intersect1d(idx_r,idx_c))

            if(idx_to_send_2D[-1].shape[0] > maxSize):
                maxSize = idx_to_send_2D[-1].shape[0]

    eprint('Max Size of 2D vector: ', maxSize)
    
    # Send images:
    # Generate an image to send the gating indices  
    field_of_view = connection.header.encoding[0].reconSpace.fieldOfView_mm                  
    imageSize = maxSize #pow(2,math.ceil(math.log2(math.sqrt(maxSize))))
    
    data = np.zeros((imageSize, numRBins,numCBins,1))

    
    for ii in range(0,len(idx_to_send_2D)):
        idx_r = (ii % numRBins)
        idx_c = int(ii / numRBins)
        data[range(0,len(idx_to_send_2D[ii])),idx_r,idx_c,0] = idx_to_send_2D[ii].squeeze()
        eprint(f"IDX {idx_c} {idx_r} : {idx_to_send_2D[ii][:10]} ")
    #data = np.reshape(data,(imageSize, numRBins,numCBins,1))
    image = create_ismrmrd_image(data, acquisition_0[0], field_of_view, ii)
    connection.send(image)
        
        
if __name__ == "__main__":
    gadgetron.external.listen(20000,SmartNavGadget)
    
    
    
    
    