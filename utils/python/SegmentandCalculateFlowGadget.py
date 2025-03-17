import numpy as np 
import os.path as op
import ismrmrd as mrd
from gadgetron_cmr_segmentation_util import prepare_case_onnx_numpy,predict_case_onnx,export_prediction_from_softmax_onnx
from time import time 
import gadgetron
import nibabel as nib
from scipy.ndimage import binary_erosion
import glob
import torch
import onnxruntime as ort
import copy
from utils_function import eprint, load_json, parse_params, read_params,create_ismrmrd_image_fast

def SegmentAndCalculateFlowGadget(connection):

    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
    matrixSize = hdr.encoding[0].reconSpace.matrixSize
    pxs_spacing=(10,field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y)

    params_init = parse_params(connection.config)
    params={'path_onnx':'',
        'path_info_preprocess': 'info_preprocess.json',
        'erosion': 0,
        'savenii': False,
        'MaskThreshold':False,
        'Threshold': 10,
        'serie_index':3,
        'image_index':0
        }

    boolean_keys=['MaskThreshold']
    str_keys=['path_onnx','path_info_preprocess']
    int_keys=['erosion','Threshold']
    
    params=read_params(params_init,params_ref=params,boolean_keys=boolean_keys,str_keys=str_keys,int_keys=int_keys)

    info_preprocess=load_json(op.abspath(params["path_info_preprocess"]))
    networks_path=glob.glob(op.abspath(params["path_onnx"]))
    eprint(networks_path)

    providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
    sess_options = ort.SessionOptions()
    networks=[ort.InferenceSession(path_network_onnx,sess_options=sess_options, providers=providers) for path_network_onnx in networks_path]

    serie_index=params['serie_index']
    n=params['image_index']
    image_phase=[]
    image_mag=[]
    segment_flag=False
    ismrmrd_flag=False
    for img in connection:
        if type(img)==gadgetron.IsmrmrdImageArray:
            print('ISMRMD')
            print(img.data.shape)
            image_nnUnetFormat=np.abs(np.nan_to_num(copy.deepcopy(img.data))).squeeze().transpose([2,0,1])[:,np.newaxis,...]
            print(image_nnUnetFormat.shape)
            connection.send(img)
            segment_flag=True
            ismrmrd_flag=True
        else:
            print(type(img))

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
                image_nnUnetFormat=np.abs(np.nan_to_num(copy.deepcopy(img_mag.data))).transpose(0,1,3,2) #(phases,z,y,x)
                segment_flag=True
                connection.send(img_phase)
                connection.send(img_mag)
            
        if segment_flag:
            segment_flag=False
            t1=time()
            if params["MaskThreshold"]:
                prediction=(image_nnUnetFormat>=params["Threshold"]).astype(np.uint16)
            else:
                list_data_test,data_properties=prepare_case_onnx_numpy(image_nnUnetFormat,pxs_spacing,info_preprocess) # ! Modify image_nnUnetFormat data
                data_predicted=predict_case_onnx(list_data_test,info_preprocess,networks)
                prediction=export_prediction_from_softmax_onnx(data_predicted,info_preprocess,data_properties).astype(np.uint16)
            if params["erosion"] >0 :
                    prediction=binary_erosion(prediction, structure=np.ones((1,1,params["erosion"],params["erosion"]))).astype(prediction.dtype)
            t2=time()
            eprint('nnUNet processes in : {} s'.format(t2-t1))
            print(np.shape(prediction))
            mask=create_ismrmrd_image_fast(prediction.transpose(0,1,3,2),field_of_view,n,serie_index)
            print(mask.data.shape)
            print("end segmentation")
            n+=1
            
            connection.send(mask)



if __name__ == '__main__':
    gadgetron.external.listen(2008,SegmentAndCalculateFlowGadget)
