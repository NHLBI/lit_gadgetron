
import gadgetron
import ismrmrd as mrd

import numpy as np
import cupy as cp

import cmath
import math
from BinningOnTheFlyGadget import get_idx_to_send,create_ismrmrd_image, eprint,estimateCardiacOrRespiratoryGatingSignal,binning
from CardiacBinningGadget import estimated_initial_ecgfreq,cardiacbinning
from scipy.signal import find_peaks
import time
import ismrmrd as mrd
from utils_function import eprint, parse_params, read_params

def CardiacRespiGadget(connection):
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
    eprint("-------------Cardiac and Respiratory Binning-------------")

    
    params_init = parse_params(connection.config)
    params={'R_numBins':8,
            'C_numBins':10,
            'phantom':False,
            "gaussian":False,
            "bstar":False,
            "samples":40,
            'useDC':True, #0,1,2 (NAV,DC,)
            'C_PHYSIO':True, #0,1,2 (NAV,DC,)
            'R_stableBinning': False,
            'R_evenbins':True,
            'R_bidirectional':False,
            'R_angular_filteration':False,
            'R_binningPercent':40,
            'C_evenbins': False,
            'C_arrythmia_detection':True,
            'C_angular_filteration':False,
            'C_stableBinning': False,
            'C_binningPercent':40,
            'C_HRinfo':False,
            "C_even_timing":True,
            "C_smoothing": True,
            "C_numBins_to_ms":False,
            "C_waveforms":False,
            }
    BPfilter_freqs=   [0.08,0.1,0.45,0.50]
    boolean_keys=['phantom','gaussian','bstar','useDC','C_PHYSIO','R_stableBinning','R_evenbins','R_bidirectional','R_angular_filteration','R_binningPercent',
    'C_evenbins','C_arrythmia_detection','C_angular_filteration','C_HRinfo','C_even_timing','C_smoothing','C_numBins_to_ms','C_stableBinning','C_waveforms']
    str_keys=[]
    int_keys=['R_numBins','C_numBins','R_binningPercent','C_binningPercent','samples']
    
    params=read_params(params_init,params_ref=params,boolean_keys=boolean_keys,str_keys=str_keys,int_keys=int_keys)
    if params['C_PHYSIO'] and params['C_waveforms']:
        connection.filter(lambda input: type(input)==mrd.Acquisition or type(input)==mrd.Waveform)
        print("Receiving Waveforms")
    else:
        connection.filter(lambda input: type(input)==mrd.Acquisition)
    gaussian_flag=params['gaussian']
    bstar_flag=params['bstar']
    numRBins=params['R_numBins']
    numCBins=params['C_numBins']

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
    mz=mrd_header.encoding[0].encodedSpace.matrixSize.z
    ecg_data = []
    ecg_tstamp = []
    ecg_indices = []
    if mz == 1:
        eprint("2D acquistion" )
        reco_2D=True
        interleaves=encoding_limits.kspace_encoding_step_1.maximum+1
    else:
        reco_2D=False
    
    acquisition_0 = []
    initial_params_comments="Respiratory and Cardiac Binning Gadget parameters: "
    for key in params.keys():
            initial_params_comments+=f"{key}={params[key]}, "
    eprint(initial_params_comments)
    time0=0
    waveform_data=[]
    waveform_timestamp=[]
    for acq in connection:
        # Navigator acquisition
        if params['C_waveforms']:
            if type(acq) == mrd.Waveform:
                if acq.waveform_id == 0:
                    waveform_data.append(np.array(acq.data))
                    waveform_timestamp.append((acq.time_stamp + np.arange(0, acq.number_of_samples))*acq.sample_time_us/1000.0)
                continue
        if (acq.isFlagSet(mrd.ACQ_IS_RTFEEDBACK_DATA or mrd.ACQ_IS_HPFEEDBACK_DATA or mrd.ACQ_IS_NAVIGATION_DATA)):
            # Use Navigator
            if not params['useDC']:
                nav_data.append(np.array(acq.data))
                nav_tstamp.append(acq.acquisition_time_stamp)
                nav_indices.append(count)
        else :
            # Data acquisition
            if params['useDC']:
                if(acq.idx.kspace_encode_step_2 == int(mrd_header.encoding[0].encodedSpace.matrixSize.z/2) or bstar_flag):
                    nav_data.append(np.array(acq.data[:,0:1]))
                    nav_tstamp.append(acq.acquisition_time_stamp)
                    nav_indices.append(count)
            if params['C_PHYSIO'] :
                ecg_data.append(np.array([[acq.physiology_time_stamp[0]]]))#[1,1] 1 channels 1 samples
                ecg_tstamp.append(acq.acquisition_time_stamp)
            

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
    

    nav_tstamp=cp.array(nav_tstamp)
    acq_tstamp=cp.array(acq_tstamp)
    nav_indices=cp.array(nav_indices)
    kencode_step=cp.array(kencode_step)
    nav_angles=cp.array(nav_angles)
    
    samples=len(nav_indices)
    [number_channels,nav_RO_sample]=nav_data[0].shape
    nav_data=cp.concatenate(cp.asarray(nav_data),axis=1)

    if len(nav_data.shape)!=3:
        nav_data = cp.reshape(nav_data,(number_channels,samples,nav_RO_sample))
    
    
    if gaussian_flag:
        print("3D cine: 1 sample every 3 samples")
        nav_data=nav_data[:,1::3,:]
        nav_tstamp=nav_tstamp[1::3]

    if bstar_flag:
        nav_data=nav_data[:,::params['samples'],:]
        nav_tstamp=nav_tstamp[::params['samples']]

    if number_of_sets >1:
        fset=lambda x: cp.mean(cp.reshape(x,[int(x.shape[0]/number_of_sets),number_of_sets]),1)
        fset_nav=lambda x: cp.mean(cp.reshape(x,[number_channels,int(samples/number_of_sets),number_of_sets,nav_RO_sample]),2)
        nav_data=fset_nav(nav_data)
        nav_angles=fset(nav_angles)
        kencode_step=fset(kencode_step).astype(cp.int32)
        nav_indices=((fset(nav_indices)-0.5)/2).astype(cp.int32)
        nav_tstamp_set=fset(nav_tstamp)
    else:
        nav_tstamp_set=nav_tstamp
    if params['R_angular_filteration'] or params['C_angular_filteration']:
        kstep_nav   = cp.concatenate((cp.asarray([kencode_step[0]]),kencode_step[nav_indices[1:]]),axis=0)
        angles_sorted = cp.zeros(cp.unique(nav_angles).shape)
        angles_sorted[kencode_step] = nav_angles
        nav_angles    = angles_sorted[kstep_nav]
    
    nav_data_copy   = nav_data.copy()
    nav_tstamp_copy = nav_tstamp_set.copy()
    nav_angles_copy = nav_angles.copy()


    #Respiratory binning
    print (f"Respiratory Binning Nbins {numRBins} stable {params['R_stableBinning']} bidirectionnal {params['R_bidirectional']}")
    print(f'DATA {nav_data_copy.shape}')
    print(f'Tstamp {nav_tstamp_copy.shape}')
    if numRBins==1 and params["R_stableBinning"]==False:
        print("No respiratory binning")
        idx_acceptedTimes=[np.arange(len(nav_tstamp_copy))]
        acceptedTimes=[nav_tstamp_copy.squeeze()]
    else:
        st = time.time()   
        respiratory_waveform, samplingTime = estimateCardiacOrRespiratoryGatingSignal(nav_data_copy,nav_tstamp_copy,nav_angles_copy,filter_freqs=BPfilter_freqs,filter_errors=[0.001,0.001,0.001],afilter = [params["R_angular_filteration"],0.01,5],bstar=bstar_flag,gaussian_flag=gaussian_flag) 
        cp.cuda.runtime.deviceSynchronize()

        et = time.time()
        elapsed_time = et - st
        eprint('Execution time (GatingSignal):', elapsed_time, 'seconds')

        acceptedTimes,idx_acceptedTimes = binning(respiratory_waveform,nav_tstamp_copy.get(),params['R_binningPercent'],params['R_bidirectional'], params['R_stableBinning'], params['R_evenbins'], numRBins)
    if bstar_flag and params['samples']==1:
        resp_bins_index=[]
        for idx_a in idx_acceptedTimes:
            idx_a.sort()
            resp_bins_index.append(np.concatenate([[len(idx_a)],idx_a]))
    else:
        resp_bins_index,maxSize=get_idx_to_send(acq_tstamp,acceptedTimes, samplingTime)

    # Cardiac binning
    print (f"Cardiac Binning Nbins {numCBins} PHYSIO {params['C_PHYSIO']} Nbinorms {params['C_numBins_to_ms']}")
    if params['C_PHYSIO']:
        nav_tstamp=cp.array(ecg_tstamp)
        if params["C_waveforms"]:
            waveform_np=np.concatenate(waveform_data,1)
            waveform_t_np = np.concatenate(waveform_timestamp,0)
            # Find ECG triggers, which are marked as the 14th bit flag
            isTrigger     = (waveform_np[4,:] & (1<<14)) != 0
            isTriggerVA61 = (waveform_np[4,:] & (1<<1)) != 0

            # Maybe >=VA61A data?
            if not any(isTrigger) and any(isTriggerVA61):
                isTrigger = isTriggerVA61
            indsTrigger = np.nonzero(isTrigger)[0]
            trigger_inds=indsTrigger.tolist()
            trigger_inds.insert(0,0)
            ecgtrigger=np.zeros(len(waveform_t_np))
            for i in range(len(trigger_inds)):
                peak_start= trigger_inds[i]
                peak_end = trigger_inds[i+1] if i+1 < len(trigger_inds) else len(ecgtrigger)
                ecgtrigger[peak_start:peak_end]=np.arange(peak_end-peak_start)
            cardiac_waveform_smooth = np.interp(2.5*nav_tstamp.get(), waveform_t_np, ecgtrigger)[None,:]
        else:
            cardiac_waveform_smooth=np.array(ecg_data)[:,:,0]
        
    if ((numCBins==1 and params['C_numBins_to_ms']==False) and params["C_stableBinning"]==False):
        print("No Cardiac binning")
        bins_index=[np.arange(len(nav_tstamp.squeeze()))]
        C_acceptedTimes=[nav_tstamp.squeeze()]
    else :
        if (params["C_stableBinning"]==True and numCBins>1):
            print("Stable Binning : Modifying number of cardiac bins")
            numCBins=1
            params['C_numBins_to_ms']=False
            
        if params['C_PHYSIO']:
            peaks ,_ =find_peaks(cardiac_waveform_smooth.squeeze())
            stime=np.mean(np.diff(2.5*cp.asnumpy(nav_tstamp.squeeze())))
            ecg_freq=1/np.median(np.diff(peaks))
            eprint(ecg_freq) 
            HR=60*ecg_freq/((stime)/1000)
            eprint(HR)
            Time_cardiac=((stime)/1000)/ecg_freq
            if params['C_numBins_to_ms']:
                numCBins=int(Time_cardiac/(params['C_numBins']*1e-3))
                eprint(f"# C bins :{numCBins} using {params['C_numBins']} ms")
            samplingTime =(1000/2.5)
        else :
            cardiac_waveform, samplingTime = estimateCardiacOrRespiratoryGatingSignal(nav_data_copy,nav_tstamp_copy,nav_angles_copy,filter_freqs=[0.6,0.65,2.0,2.1],filter_errors=[0.01,0.01,0.01],afilter = [params["C_angular_filteration"],0.1,5]) 
            cp.cuda.runtime.deviceSynchronize()

            ecg_freq=estimated_initial_ecgfreq(cardiac_waveform,samplingTime,smooth=True,freqwindow=0.05)
            if params['C_smoothing']:
                cardiac_waveform_smooth, samplingTime = estimateCardiacOrRespiratoryGatingSignal(nav_data_copy,nav_tstamp_copy,nav_angles_copy,filter_freqs=[ecg_freq-0.3,ecg_freq-0.25,ecg_freq+0.25,ecg_freq+0.3],afilter = [params["C_angular_filteration"],0.1,5]) 
            else:
                cardiac_waveform_smooth = cardiac_waveform
        bins_index,ecg_freq_final=cardiacbinning(cardiac_waveform_smooth,samplingTime,numCBins,ecg_freq,evenbins=params['C_evenbins'],phantomflag=params['phantom'],arrythmia_detection=params['C_arrythmia_detection'],even_timing=params["C_even_timing"],stable_binning=params['C_stableBinning'],stable_perc=params['C_binningPercent'])
        C_Index=np.arange(len(nav_tstamp.squeeze()))
        print(len(C_Index))
        C_acceptedTimes=[]
        for set in range(number_of_sets):
            for nbin in range(len(bins_index)):
                bin=np.array(bins_index[nbin])
                set_nav_tstamp=bin[np.in1d(bin,C_Index)]
                C_acceptedTimes.append(nav_tstamp[set_nav_tstamp*2+set])
    
    if params['C_PHYSIO']:
        cardiac_index=[]
        for idx_a in bins_index:
            idx_a.sort()
            cardiac_index.append(np.concatenate([[len(idx_a)],np.array(idx_a)]))
    else:
        cardiac_index,maxSize=get_idx_to_send(acq_tstamp,C_acceptedTimes, samplingTime/number_of_sets)

    maxSize = 0 
    idx_to_send_2D = []
    for idx_set in range(number_of_sets):
        for idx_c in cardiac_index:
            for idx_r in resp_bins_index:
                temp_r_c = np.intersect1d(idx_r[1:],idx_c[1:])
                idx_to_send_2D.append( np.concatenate( ([temp_r_c.shape[0]],temp_r_c)))

                if(idx_to_send_2D[-1].shape[0] > maxSize):
                    maxSize = idx_to_send_2D[-1].shape[0]
        eprint('Max Size of 2D vector: ', maxSize)
    
    # Send images:
    # Generate an image to send the gating indices  
    field_of_view = connection.header.encoding[0].reconSpace.fieldOfView_mm                  
    imageSize = maxSize #pow(2,math.ceil(math.log2(math.sqrt(maxSize))))
    max_nchannels=(np.power(2,16)-1) #Bug nchannels in Image Header is a np.uint16 (max value =65535)
    if np.prod([imageSize, numRBins,numCBins,number_of_sets])==imageSize and imageSize>max_nchannels :
        print("Carefull too much data in each bins, required to collapse Respiratory dimension !!!")
        numRBins=int(np.ceil(imageSize/(max_nchannels-1)))
        maxSize=max_nchannels-1
        idx_0=idx_to_send_2D[0][1:]
        idx_to_send_2D = []
        for idx_r in range(numRBins):
            print(idx_r)
            print(idx_r*max_nchannels)
            print(np.min([max_nchannels*(idx_r+1),len(idx_0)]))
            tmp_idx=idx_0[idx_r*maxSize:np.min([maxSize*(idx_r+1),len(idx_0)])]
            idx_to_send_2D.append( np.concatenate( ([tmp_idx.shape[0]],tmp_idx)))
        imageSize=max_nchannels
    data = np.zeros((imageSize, numRBins,numCBins,number_of_sets))
    
        
    for ii in range(0,len(idx_to_send_2D)):
        idx_set=ii//(numRBins*numCBins)
        idx_c=(ii//numRBins) % numCBins
        idx_r=ii%numRBins
        data[range(0,len(idx_to_send_2D[ii])),idx_r,idx_c,idx_set] = idx_to_send_2D[ii].squeeze()
    

    image = create_ismrmrd_image(data, acquisition_0[0], field_of_view, ii)
    connection.send(image)

    
if __name__ == '__main__':
    gadgetron.external.listen(2000,CardiacRespiGadget)
    
    