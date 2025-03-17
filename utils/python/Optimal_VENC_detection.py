import numpy as np 
import ismrmrd as mrd
from time import time 
from copy import deepcopy
import gadgetron
from scipy.ndimage import binary_erosion
from skimage.restoration import unwrap_phase
from utils_function import eprint, load_json, parse_params, read_params, create_ismrmrd_image_fast
import copy
def get_tissue_mask(pha_image,roi_image):
    non_roi_phase=phase_image*~roi_image
    stdev_map=np.std(non_roi_phase,axis=-1)

    # Verification of standard deviation map
    print(np.mean(stdev_map), non_roi_phase[0,0,:].std(), non_roi_phase[0,1,:].std(), non_roi_phase[0,3,:].std(), non_roi_phase[25,5,:].std())
    print("Stdev Map:", stdev_map)
    # Original ROI included bc stdev of ROI set as 0
    tissue_mask = stdev_map < np.mean(stdev_map)
    return tissue_mask  

def CalculateoptimalVENC(pha_image,roi_image,erosion=0):
    image_unwrapped = unwrap_phase(pha_image)
    if erosion >0 :
        roi_image=binary_erosion(roi_image, structure=np.ones((erosion,erosion,1))).astype(np.int8)
    OPTI_val_uw=np.round(np.max(np.abs(image_unwrapped)*roi_image)/np.pi,3)
    OPTI_val=np.round(np.max(np.abs(pha_image)*roi_image)/np.pi,3)
    return OPTI_val_uw,OPTI_val

def similarity_metrics(pred ,gt ,label=1):
    """
    dice : Dice coefficient
    jc : jaccard coefficient
    """
    label_pred = pred == label
    label_gt = gt == label


    intersection = np.count_nonzero(label_pred & label_gt)
    union = np.count_nonzero(label_pred | label_gt)

    num_pred = np.count_nonzero(label_pred)
    numg_gt = np.count_nonzero(label_gt)
    try:
        dice = 2. * intersection / float(num_pred + numg_gt )
    except ZeroDivisionError:
        dice = 0.0

    try:
        jc = float(intersection) / float(union)
    except ZeroDivisionError:
        jc = 0.0

    return dice ,jc

def OptimalVENCGadget(connection):

    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
    matrixSize = hdr.encoding[0].reconSpace.matrixSize
    pxs_spacing=(10,field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y)
    eprint(f"Number of sets :{hdr.encoding[0].encodingLimits.set.maximum+1} ")
    nset=hdr.encoding[0].encodingLimits.set.maximum+1
    VENCs=[200 for i in range(nset-1)]
    OPTI_VEL=[0 for i in range(nset-1)]
    OPTI_VEL_unwrap=[0 for i in range(nset-1)]
    
    for idx in range (len(hdr.userParameters.userParameterLong)):
        if ("VENC" in hdr.userParameters.userParameterLong[idx].name):
            VENC_set=hdr.userParameters.userParameterLong[idx].name.split('_')[-1]
            VENCs[int(VENC_set)] = hdr.userParameters.userParameterLong[idx].value

        
    params_init = parse_params(connection.config)
    params={
        'erosion': 0,
        'serie_index':4,
        'images_index':0,
        }

    boolean_keys=[]
    str_keys=[]
    int_keys=['erosion']

    params=read_params(params_init,params_ref=params,boolean_keys=boolean_keys,str_keys=str_keys,int_keys=int_keys)


    serie_index=params['serie_index']
    ni=params['images_index']
    image_phase=[]
    image_mag=[]
    image_mask=[]
    segment_flag=True
    for img in connection:
        if type(img)==gadgetron.IsmrmrdImageArray:
            eprint('ISMRMD')
            eprint(img.data.shape)
            np_image_phase=np.angle(np.nan_to_num(copy.deepcopy(img.data))).squeeze()
            set_phase=img.headers[0,0,0].set
            VENC_phase=VENCs[set_phase]
            eprint(np_image_phase.shape)
            connection.send(img)
            segment_flag=True
        else:
            eprint(type(img))

            if img.image_series_index ==3:
                if(len(image_mask)<1):
                    image_mask.append(img)

            if img.image_type ==mrd.IMTYPE_PHASE:
                if(len(image_phase)<1):
                    image_phase.append(img)

            if img.image_type ==mrd.IMTYPE_MAGNITUDE:
                if(len(image_mag)<1):
                    image_mag.append(img)      
            if len(image_mag)==1 and len(image_phase):
                img_mag=image_mag.pop()
                img_phase=image_phase.pop()
                eprint(img_phase.data.shape)
                eprint(img_mag.data.shape)
                np_image_phase = copy.deepcopy(np.nan_to_num(img_phase.data)).transpose(3,2,1,0).squeeze()
                print(img._head)
                set_phase=img._head.set
                VENC_phase=VENCs[0]
                eprint('BE carefull need to change')
                segment_flag=True
                connection.send(img_phase)
                connection.send(img_mag)

        if (segment_flag and len(image_mask)==1):
            img_mask=image_mask.pop()
            t1=time()
            eprint(img_mask.data.shape)
            np_roi = np.nan_to_num(deepcopy(img_mask.data)).transpose(3,2,1,0).squeeze() #(x,y,z,phases)
            eprint(np_roi.shape)
            np_image_unwrapped = unwrap_phase(np_image_phase)
            if params['erosion'] >0 :
                np_roi=binary_erosion(np_roi, structure=np.ones((params['erosion'] ,params['erosion'] ,1))).astype(np.int8)
            dice_score=0.8
            c_roi=np.zeros(np_roi.shape)
            nslices=np_roi.shape[-1]
            for n in range(np_roi.shape[-1]):
                # Get dice coefficient if DSC > DSC score, slices is coherent with the others
                QC_segmentation=np.array([int(similarity_metrics(np_roi[...,n],np_roi[...,k],1)[0]>dice_score) for k in range(nslices)]).sum()
                if QC_segmentation > nslices //2:
                    c_roi[...,n]=np_roi[...,n]
                    print(f'Slice : {n}')
                    print(f"VENC Phase {VENC_phase} {VENC_phase*np.round(np.max(np.abs(np_image_phase[...,n])*c_roi[...,n])/np.pi,3)}")
            eprint(np.unravel_index(np.argmax((np.abs(np_image_unwrapped)*c_roi)),c_roi.shape))
            eprint(np.unravel_index(np.argmax((np.abs(np_image_phase)*c_roi)),c_roi.shape))
            OPTI_val_uw=VENC_phase*np.round(np.max(np.abs(np_image_unwrapped)*c_roi)/np.pi,3)
            OPTI_val=VENC_phase*np.round(np.max(np.abs(np_image_phase)*c_roi)/np.pi,3)


            t2=time()
            eprint('Unwrapping in : {} s'.format(t2-t1))
            eprint(OPTI_val_uw)
            eprint(OPTI_val)
            perc_range=1.10 # 110%
            OPTI_VEL[set_phase]=OPTI_val
            OPTI_VEL_unwrap[set_phase]=OPTI_val_uw

            img_unwrapped=create_ismrmrd_image_fast(np_image_unwrapped.transpose(2,1,0)[:,np.newaxis,:,:],field_of_view,ni,serie_index)
            #img_unwrapped._head.user_float[0]=(t2-t1)*1e3 #in ms
            for set_s in range(nset-1):
                img_unwrapped._head.user_int[set_s]=VENCs[set_s] #in ms
                img_unwrapped._head.user_float[2*set_s]=OPTI_VEL[set_s] #in ms
                img_unwrapped._head.user_float[2*set_s+1]=OPTI_VEL_unwrap[set_s]  #in ms
            #img_unwrapped.meta={"OPTI_VAL":int(OPTI_val*perc_range)}
            img_mask._head.user_int[1]=int(OPTI_val_uw*perc_range) #in ms
            connection.send(img_mask)
            connection.send(img_unwrapped)
            ni+=1

if __name__ == '__main__':
    gadgetron.external.listen(2004,OptimalVENCGadget)
