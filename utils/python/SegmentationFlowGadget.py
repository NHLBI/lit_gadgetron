import numpy as np 
import os.path as op
import ismrmrd as mrd
from  onnxruntime import InferenceSession
from utils_function import eprint, load_json, parse_params, read_params
from gadgetron_cmr_segmentation_util import prepare_case_onnx_numpy,predict_case_onnx,export_prediction_from_softmax_onnx
from time import time 
import gadgetron
import nibabel as nib
from scipy.ndimage import binary_erosion
import glob
from utils_function import eprint, load_json, parse_params, read_params,create_ismrmrd_image_fast

def SegmentationFlowGadget(connection):
    # What are the dimension order of the input data ?

    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
    #To verify 
    matrixSize = hdr.encoding[0].reconSpace.matrixSize
    pxs_spacing=(10,field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y)
    params_init = parse_params(connection.config)

    params={'path_onnx':'',
            'path_info_preprocess': 'info_preprocess.json',
            'erosion': 0,
            'savenii': False,
            'MaskThreshold':False,
            'Threshold': 10,
            }


    boolean_keys=['savenii','MaskThreshold']
    str_keys=['path_onnx','path_info_preprocess']
    int_keys=['erosion','Threshold']
    
    params=read_params(params_init,params_ref=params,boolean_keys=boolean_keys,str_keys=str_keys,int_keys=int_keys)

    info_preprocess=load_json(op.abspath(params["path_info_preprocess"]))
    networks_path=glob.glob(op.abspath(params["path_onnx"]))
    eprint(networks_path)

    networks=[InferenceSession(path_network_onnx) for path_network_onnx in networks_path]
    n=0
    serie_index =3
    image_0=[]

    for img in connection:
        if(len(image_0)<1):
            image_0.append(img)
        
        t1=time() 
        image_nnUnetFormat=np.abs(np.nan_to_num(img.data)).transpose(1,0)[np.newaxis,np.newaxis,...]
        eprint(image_nnUnetFormat.shape)

        list_data_test,data_properties=prepare_case_onnx_numpy(image_nnUnetFormat,pxs_spacing,info_preprocess) # ! Modify image_nnUnetFormat data
        data_predicted=predict_case_onnx(list_data_test,info_preprocess,networks)
        prediction=export_prediction_from_softmax_onnx(data_predicted,info_preprocess,data_properties).astype(np.uint16)
        if params["erosion"] >0 :
                    prediction=binary_erosion(prediction, structure=np.ones((1,1,params["erosion"],params["erosion"]))).astype(prediction.dtype)
        t2=time()
        eprint('nnUNet processes in : {} s'.format(t2-t1))
        if params["MaskThreshold"]:
            prediction=(np.abs(np.nan_to_num(img.data)).transpose(0,1,3,2)>=params["Threshold"]).astype(prediction.dtype)
        
        t2=time()
        eprint('nnUNet processec SNR mean map in : {} s'.format(t2-t1))
        
        if params["savenii"]:
            vds='VDSunknown'
            slines='Sunknown'
            nii_volume = nib.Nifti1Image(image_nnUnetFormat[0,...].transpose(2,1,0),  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/image_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_volume, output_path)
            nii_volume = nib.Nifti1Image(img.data,  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/rawimage_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_volume, output_path)
            nii_volume = nib.Nifti1Image(np.abs(img.data),  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/rawimage_mag_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_volume, output_path)
            nii_volume = nib.Nifti1Image(np.angle(img.data),  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/rawimage_phase_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_volume, output_path)
            nii_mask = nib.Nifti1Image(prediction, np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/mask_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_mask, output_path)
        eprint(np.shape(prediction))
        final_prediction=prediction.transpose(0,1,3,2)
        mask=create_ismrmrd_image_fast(prediction,field_of_view,n,serie_index)
        n+=1
        mask._head.user_float[0]=(t2-t1)*1e3 #in ms
        SNR=np.mean(np.abs(np.abs(np.nan_to_num(img.data)).squeeze()[final_prediction.squeeze()!=0]))
        eprint(SNR)
        mask._head.user_float[3]=SNR #in ms
        connection.send(img)
        connection.send(mask)

if __name__ == '__main__':
    gadgetron.external.listen(2009,SegmentationFlowGadget)
