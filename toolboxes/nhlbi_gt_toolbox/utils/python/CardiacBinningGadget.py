
import gadgetron
import ismrmrd as mrd

import numpy as np
import cupy as cp

import cmath
import math
from BinningOnTheFlyGadget import get_idx_to_send,create_ismrmrd_image,binning, eprint,estimateCardiacOrRespiratoryGatingSignal
from scipy.signal import find_peaks
from itertools import groupby
from operator import itemgetter
import scipy
import ismrmrd as mrd
from utils_function import eprint, load_json, parse_params, read_params

def estimated_initial_ecgfreq(selectedSig,samplingTime,freqwindow=0.025,smooth=False):
    """
    Detecting the initial ecg frequency based on the max of the power of the spectrum.

    Parameters
    ----------
    
    selectedSig : np.ndarray, 
        NAV/DC signal 

    samplingTime : float, 
        sampling time 

    freqwindow : float, 
        float Median filter window (default: 0.025 Hz)

    smooth : boolean, 
        Flag smooth the spectrum using a median filter with windows define by freqwindow (default: False)

    Returns
    ------- 

    ecg_freq_initial : np.float64,
        Initial ecg frequency estimated 
    
    """
    fs=1/float(2.5*samplingTime/1000)
    fft_waveform=np.abs(np.fft.fft(np.squeeze(selectedSig)))
    n_samples=cp.shape(selectedSig)[1]
    fr=np.arange(n_samples)*fs/int(n_samples)
    if smooth:
        window_s=int(np.ceil(freqwindow/(fs/n_samples)))
        fft_waveform_smooth=scipy.ndimage.median_filter(fft_waveform,window_s)
    else:
        fft_waveform_smooth=fft_waveform
    ecg_freq_initial=fr[np.argmax(fft_waveform_smooth[:int(n_samples/2)])]

    return ecg_freq_initial

def cardiacbinning(selectedSig,samplingTime,numBins,ecg_freq,evenbins=False,phantomflag=False,arrythmia_detection=False,even_timing=False,stable_binning=False,stable_perc=40):
    """
    Calculating cardiac bins based on the navigator/DC signal. 

    Parameters
    ----------
    
    selectedSig : np.ndarray, 
        NAV/DC signal 

    samplingTime : float, 
        sampling time 

    numBins : int, 
        number of bins 

    ecg_freq : float, 
        initial ecg frequency estimated 

    evenbins : boolean, 
        Flag to obtain uniform bins (default: False)

    phantomflag : boolean, 
        Flag for phantom data (simplified binning: mod(sample index,numBins)) (default: False)

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
    if phantomflag:
        RR_vec=np.arange(n_samples)
        bin_data_label=RR_vec % (numBins)
        RR_seq=list(map(list, zip(RR_vec.tolist(), bin_data_label.tolist())))
        RR_seq.sort(key=itemgetter(1)) # Not necessary now 
        bins=[[x for x,y in data] for (key, data) in groupby(RR_seq, itemgetter(1))]
        final_ecg_freq=1 #1Hz
    else:
        #Minimum distance betwwen peak 
        min_distance=round(0.75* (fs/ecg_freq))
        peaks, _ = find_peaks(selectedSig.flatten(), distance=min_distance)
        #Binnings
        bins=[[] for n in range(numBins)]
        final_ecg_freq=ecg_freq
        if len(peaks)>=2:
            RR_ecg=list()
            peak_list=list(zip(peaks[:-1],peaks[1:],np.diff(peaks).tolist()))
            if arrythmia_detection:
                ln_p=len(peak_list)
                peak_list=[(peak_start,peak_stop,RR_vec) for peak_start,peak_stop,RR_vec in peak_list if RR_vec <1.5*(fs/ecg_freq)]
                eprint('Arrythmia reject (%)...')
                eprint(100*(ln_p-len(peak_list))/ln_p)
            
            if stable_binning:
                bins=[[]]
                for peak_start,peak_stop,RR_vec in peak_list:
                    idx_vec=np.arange(RR_vec)
                    min_idx=RR_vec*(1-stable_perc/100)
                    bins[0].extend((peak_start+idx_vec[idx_vec>=min_idx]).tolist())
                    RR_ecg.append(fs/RR_vec)
            else:
                npts=int(np.round(np.array(peak_list)[:,-1].mean()/(numBins)))
                if even_timing:
                    for peak_start,peak_stop,RR_vec in peak_list:
                        binI=np.pad(np.arange(RR_vec),int(np.ceil(RR_vec/numBins)),mode='wrap')
                        center=(RR_vec/numBins)*np.arange(numBins)
                        center_i=np.argmin(np.abs((binI[None,...]-center[...,None])),1)
                        for idx_w in range(np.sum(np.diff(center_i)<0)):
                            center_i[-1-idx_w]=np.where(binI==binI[center_i[-1-idx_w]])[0][-1]
                        RR_peak_bins=[]
                        for n in range(numBins):
                            RR_peak_bins.append((peak_start+binI[center_i[n]-npts//2:center_i[n]+max(npts//2,1)]).tolist())
                        bins=list(map(lambda x, y:x + y, bins, RR_peak_bins))
                        RR_ecg.append(fs/RR_vec)
                else :
                    for peak_start,peak_stop,RR_vec in peak_list:
                        bin_data_label=np.arange(RR_vec)//(RR_vec/numBins)
                        RR_seq=list(map(list, zip(np.arange(peak_start,peak_stop).tolist(), bin_data_label.tolist())))
                        RR_seq.sort(key=itemgetter(1)) # Not necessary now 
                        RR_peak_bins=[[x for x,y in data] for (key, data) in groupby(RR_seq, itemgetter(1))]
                        bins=list(map(lambda x, y:x + y, bins, RR_peak_bins))
                        RR_ecg.append(fs/RR_vec)
                final_ecg_freq=np.mean(np.array(RR_ecg))
    eprint('ECG FREQ INIT : {} ECG FREQ FINAL : {}'.format(ecg_freq,final_ecg_freq))
    if evenbins:
        len_bins=[len(bins[n]) for n in range(numBins)]
        min_len=min(len_bins)
        bins=[bins[n][:min_len] for n in range(numBins)]

    return bins,final_ecg_freq

def CardiacBinningGadget(connection):
    """
    Parameters
    ---------- 

    params : dict 
        Dictionnary that contains the following information :
                - numBins : int, 
                    number of bins (default: 25)
                - phantom : boolean, 
                    Flag for phantom data (simplified binning: mod(sample index,numBins)) (default: False)
                - respiRate : int,  
                    % of data kept after respiration gating (default: 100)
                - evenbins : boolean, 
                    Flag to obtain uniform bins (default: False)
                - version : string, 
                    Flag to decide on which signal should be calculated the bins ('v1': raw signal, 'v2': filtered signal around +-0.25Hz ecg initial frequency , default: 'v2')
                - arrythmia_detection : boolean, 
                    Remove RR vectors which are too long (>1.5(ecg_freq_initial) (default: True)
                - angular_filteration : boolean,
                    Angular trajectories correction (default=False)  

    Returns
    -------

    idx_to_send : List[np.ndarray], List of binned indexes : 
        - Length of the list is equal to the number of bins * sets    
        - np.ndarray : 1rst element corresponds to the number of indexes of the bin
    
    """
    eprint("-------------Cardiac Binning-------------")

    connection.filter(mrd.Acquisition)
    
    params_init = parse_params(connection.config)
    params={'numBins':25,
            'phantom':False,
            'evenbins': False,
            'arrythmia_detection':True,
            'angular_filteration':False,
            'HRinfo':False,
            'NAV_DC_PHYSIO':1, #0,1,2 (NAV,DC,)
            "gaussian":False,
            "bstar":False,
            "even_timing":True,
            "smoothing": True,
            }

    boolean_keys=['phantom','evenbins','arrythmia_detection','angular_filteration','HRinfo','smoothing','even_timing']
    str_keys=['version']
    int_keys=['numBins','respiRate','NAV_DC_PHYSIO']
    
    params=read_params(params_init,params_ref=params,boolean_keys=boolean_keys,str_keys=str_keys,int_keys=int_keys)
    gaussian_flag=params['gaussian']
    bstar_flag=params['bstar']
    count = 0
    nav_angles = []
    acq_tstamp = []
    nav_data    = []
    nav_tstamp  = []
    nav_indices = []
    data_indices = []
    kencode_step = []
    mrd_header = connection.header

    field_of_view = mrd_header.encoding[0].reconSpace.fieldOfView_mm

    encoding_limits = mrd_header.encoding[0].encodingLimits
    number_of_sets=encoding_limits.set.maximum+1
    eprint(encoding_limits.repetition.maximum+1)
    mz=mrd_header.encoding[0].encodedSpace.matrixSize.z
    if mz == 1:
        eprint("2D acquistion" )
        reco_2D=True
        interleaves=encoding_limits.kspace_encoding_step_1.maximum+1
    else:
        reco_2D=False
    
    acquisition_0 = []

    time0=0
    for acq in connection:
        
        # Navigator acquisition
        if (acq.isFlagSet(mrd.ACQ_IS_RTFEEDBACK_DATA or mrd.ACQ_IS_HPFEEDBACK_DATA or mrd.ACQ_IS_NAVIGATION_DATA)):
            # Use Navigator
            if params['NAV_DC_PHYSIO']==0:
                nav_data.append(np.array(acq.data))
                nav_tstamp.append(acq.acquisition_time_stamp)
                nav_indices.append(count)
        else :
            # Data acquisition
            if params['NAV_DC_PHYSIO']==1:
                if(acq.idx.kspace_encode_step_2 == int(mrd_header.encoding[0].encodedSpace.matrixSize.z/2)):
                    nav_data.append(np.array(acq.data[:,0:1]))
                    nav_tstamp.append(acq.acquisition_time_stamp)
                    nav_indices.append(count)
            elif params['NAV_DC_PHYSIO']==2:
                nav_data.append(np.array([[acq.physiology_time_stamp[0]]]))#[1,1] 1 channels 1 samples
                nav_tstamp.append(acq.acquisition_time_stamp)
                nav_indices.append(count)
            else:
                pass 

            if not bstar_flag:
                if reco_2D:
                    nav_angles.append(2*np.pi*(acq.idx.kspace_encode_step_1/interleaves))
                else :
                    nav_angles.append(180*cmath.phase(complex(acq.traj[25,0],acq.traj[25,1]))/math.pi)
            acq_tstamp.append(acq.acquisition_time_stamp)
            data_indices.append(count)
            kencode_step.append(acq.idx.kspace_encode_step_1) 
            if(len(acquisition_0)<1):
                acquisition_0.append(acq)
                time0= acq.acquisition_time_stamp 
            connection.send(acq)                
        count+=1
    timeAcq=(acq_tstamp[-1]-time0)*2.5

    eprint("Acquisition: Total time {} ms , number: {} ".format(timeAcq,count))
    acceptedTimes=[[] for n in range(params['numBins']*number_of_sets)]

    nav_tstamp=cp.array(nav_tstamp)
    acq_tstamp=cp.array(acq_tstamp)
    nav_indices=cp.array(nav_indices)
    kencode_step=cp.array(kencode_step)
    nav_angles=cp.array(nav_angles)
    
    samples=len(nav_indices)
    [number_channels,nav_RO_sample]=nav_data[0].shape
    nav_data=cp.concatenate(cp.array(nav_data),axis=1)

    if len(nav_data.shape)!=3:
        nav_data = cp.reshape(nav_data,(number_channels,samples,nav_RO_sample))
    
    
    if gaussian_flag:
        print("3D cine: 1 sample every 3 samples")
        nav_data_copy=nav_data[:,1::3,:]
        nav_tstamp_copy=nav_tstamp[1::3]

    if number_of_sets >1 and (params['NAV_DC_PHYSIO']!=2):
        fset=lambda x: cp.mean(cp.reshape(x,[int(x.shape[0]/number_of_sets),number_of_sets]),1)
        fset_nav=lambda x: cp.mean(cp.reshape(x,[number_channels,int(samples/number_of_sets),number_of_sets,nav_RO_sample]),2)
        nav_data=fset_nav(nav_data)
        nav_angles=fset(nav_angles)
        kencode_step=fset(kencode_step).astype(cp.int32)
        nav_indices=((fset(nav_indices)-0.5)/2).astype(cp.int32)
        nav_tstamp_set=fset(nav_tstamp)
    else:
        nav_tstamp_set=nav_tstamp
    if params['angular_filteration']:
        kstep_nav   = cp.concatenate((cp.asarray([kencode_step[0]]),kencode_step[nav_indices[1:]]),axis=0)
        angles_sorted = cp.zeros(cp.unique(nav_angles).shape)
        angles_sorted[kencode_step] = nav_angles
        nav_angles    = angles_sorted[kstep_nav]
    
    nav_data_copy   = nav_data.copy()
    nav_tstamp_copy = nav_tstamp_set.copy()
    nav_angles_copy = nav_angles.copy()

    if params['NAV_DC_PHYSIO']!=2:
        cardiac_waveform, samplingTime = estimateCardiacOrRespiratoryGatingSignal(nav_data_copy,nav_tstamp_copy,nav_angles_copy,filter_freqs=[0.6,0.65,2.0,2.1],filter_errors=[0.01,0.01,0.01],afilter = [params["angular_filteration"],0.1,5]) 
        cp.cuda.runtime.deviceSynchronize()

        ecg_freq=estimated_initial_ecgfreq(cardiac_waveform,samplingTime,smooth=True,freqwindow=0.05)
        if params['smoothing']:
            cardiac_waveform_smooth, samplingTime = estimateCardiacOrRespiratoryGatingSignal(nav_data_copy,nav_tstamp_copy,nav_angles_copy,filter_freqs=[ecg_freq-0.3,ecg_freq-0.25,ecg_freq+0.25,ecg_freq+0.3],afilter = [params["angular_filteration"],0.1,5]) 
        else:
            cardiac_waveform_smooth = cardiac_waveform
    else :
        cardiac_waveform_smooth=nav_data_copy[:,:,0]
        peaks ,_ =find_peaks(cardiac_waveform_smooth.squeeze())
        stime=np.mean(np.diff(2.5*cp.asnumpy(nav_tstamp.squeeze())))
        ecg_freq=1/np.median(np.diff(peaks))
        eprint(ecg_freq) 
        HR=60*ecg_freq/((stime)/1000)
        eprint(HR)
        Time_cardiac=((stime)/1000)/ecg_freq
        numBins=int(Time_cardiac/(30*1e-3))
        eprint(numBins)
        # sampling time = 1 (1000/2.5)*2.5/1000
        # ecg_init = median of diff peaks
        samplingTime =(1000/2.5)
    bins_index,ecg_freq_final=cardiacbinning(cardiac_waveform_smooth,samplingTime,params['numBins'],ecg_freq,evenbins=params['evenbins'],phantomflag=params['phantom'],arrythmia_detection=params['arrythmia_detection'],even_timing=params["even_timing"])
    
    Index=np.arange(samples)
    print(number_of_sets)
    for nbin in range(len(bins_index)):
        bin=np.array(bins_index[nbin])
        for set in range(number_of_sets):
            set_nav_tstamp=bin[np.in1d(bin,Index)]
            acceptedTimes[set*params['numBins']+nbin]=nav_tstamp[set_nav_tstamp*2+set]
    
    if params['NAV_DC_PHYSIO']==2:
        idx_to_send=[]
        maxSize=0
        for idx_a in bins_index:
            idx_a.sort()
            idx_to_send.append(np.concatenate([[len(idx_a)],cp.asnumpy(idx_a)]))
            maxSize=max([len(idx_to_send[-1]),maxSize])
            eprint(idx_to_send[-1][:40])
    else:
        idx_to_send,maxSize=get_idx_to_send(acq_tstamp,acceptedTimes, samplingTime/number_of_sets)
            
    [Nidx,Nr,Nc,Nechoes]=[maxSize,1,len(acceptedTimes),1]
    data = np.zeros([Nidx,Nr,Nc,Nechoes])
    for ii in range(0,len(acceptedTimes)):
        data[range(0,len(idx_to_send[ii])),0,ii,0] = idx_to_send[ii].squeeze()
    
    image = create_ismrmrd_image(data, acquisition_0[0], field_of_view, 0)
    connection.send(image)

    #Add cardiac time (1/ecg_freq/num_bins*1000) micros
    """
    if params['HRinfo']:
        tframes=(1/ecg_freq_final/params['numBins'])*1000*1000
        data = np.zeros((imageSize*imageSize))
        data[0:2]=[1,tframes]
        data = np.reshape(data,(imageSize,imageSize))
        image = create_ismrmrd_image(data, acquisition_0[0], field_of_view,len(acceptedTimes))
        connection.send(image)
    """
    
if __name__ == '__main__':
    gadgetron.external.listen(2000,CardiacBinningGadget)
    
    