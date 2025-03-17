from typing import Union, Tuple,List
from nnUnet_utils_numpy import crop_to_nonzero,compute_new_shape,resample_data_or_seg_to_shape ,ZScoreNormalization
from acvl_utils_bounding_boxes import bounding_box_to_slice
from sliding_window_prediction_onnx import predict_sliding_window_return_logits
import numpy as np
from  onnxruntime import InferenceSession
import numpy as np 
import os.path as op
import json

def np_sigmoid(x):
    return 1/(1 + np.exp(-x))

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


####For Ahsan #### 
def eval_case_onnx_numpy(image_array: np.ndarray,pxs_spacing:Tuple[int,...], path_onnx:str, path_info_preprocess:str):
    info_preprocess=load_json(path_info_preprocess)
    network=InferenceSession(path_onnx)
    data_test,data_properties=prepare_case_onnx_numpy(image_array,pxs_spacing,info_preprocess)
    data_predicted=predict_case_onnx(data_test,info_preprocess,network)
    prediction=export_prediction_from_softmax_onnx(data_predicted,info_preprocess,data_properties)
    return prediction

def predict_case_onnx(data_test: np.ndarray,info_preprocess: dict,networks:List[InferenceSession]):
    prediction = predict_sliding_window_return_logits(networks, data_test, info_preprocess['num_seg_heads'],info_preprocess['patch_size'],
                                                      mirror_axes=info_preprocess['inference_allowed_mirroring_axes'],
                                                      tile_step_size=info_preprocess['tile_step_size'],
                            use_gaussian=info_preprocess['use_gaussian'],
                            precomputed_gaussian=info_preprocess['inference_gaussian'])
    
    
    return prediction

def prepare_case_onnx_numpy(data:np.ndarray,pxs_spacing:Tuple[int,...],info_preprocess: dict):
    """

    order of operations is: transpose -> crop -> resample
    so when we export we need to run the following order: resample -> crop -> transpose (we could also run
    transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
    """
    
    transpose_f=info_preprocess['transpose_forward']
    IntProp=info_preprocess['foreground_intensity_properties_per_channel']
    target_spacing = info_preprocess['target_spacing']  # this should already be transposed
    normalization_schemes=info_preprocess['normalization_schemes']
    use_mask_for_norm =info_preprocess['use_mask_for_norm']

    # apply transpose_forward, this also needs to be applied to the spacing!
    data = data.transpose([0, *[i + 1 for i in transpose_f]])
    original_spacing = [pxs_spacing[i] for i in transpose_f]
    shape_before_cropping = data.shape[1:]
    data, seg, bbox = crop_to_nonzero(data, None)
    # resample
    data_properites={'spacing':pxs_spacing,
                     'shape_before_cropping':shape_before_cropping,
                     'bbox_used_for_cropping':bbox,
                     'shape_after_cropping_and_before_resampling':data.shape[1:]
                     }
    
    if len(target_spacing) < len(data.shape[1:]):
        # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
        # in 3d we do not change the spacing between slices
        target_spacing = [original_spacing[0]] + target_spacing
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
    # normalize
    # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
    # longer fitting the images perfectly!

    for c in range(data.shape[0]):
            normalizer = ZScoreNormalization(use_mask_for_norm=use_mask_for_norm[0],intensityproperties=IntProp[str(0)])
            data[c] = normalizer.run(data[c], seg[0])

    old_shape = data.shape[1:]
    data = resample_data_or_seg_to_shape(data, new_shape, original_spacing, target_spacing)
    return data,data_properites



def export_prediction_from_softmax_onnx(predicted_array: np.ndarray,info_preprocess: dict, properties_dict: dict,output_file_name:str=None):
    """
    5D array  (b,c,x,y,z)
    """
    batch_size=predicted_array.shape[0]
    ch=predicted_array.shape[1]
    predicted_array = predicted_array.astype(np.float32)
    # resample to original shape
    current_spacing = info_preprocess['target_spacing']  # this should already be transposed
    
    #Resampling
    predicted_probabilities=np.zeros([batch_size,ch, *properties_dict['shape_after_cropping_and_before_resampling']],dtype=np.float32)
    
    for b in range(ch):
        predicted_probabilities[:,b,...] = resample_data_or_seg_to_shape(predicted_array[:,b,...],properties_dict['shape_after_cropping_and_before_resampling'],current_spacing,properties_dict['spacing'])

    #segmentation does not handle multiple regions  mix(features nnUnet hasregion) 
    predicted_probabilities=np_sigmoid(predicted_probabilities)
    segmentation = predicted_probabilities.argmax(1) #0

    #segmentation = label_manager.convert_logits_to_segmentation(predicted_array)

    # put result in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros([batch_size,*properties_dict['shape_before_cropping']], dtype=np.uint8)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    slicer_batch=tuple([slice(None),*slicer])
    segmentation_reverted_cropping[slicer_batch] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose([0, *[i + 1 for i in info_preprocess['transpose_backward']]])

    if output_file_name is not None:
        del predicted_array

    return segmentation_reverted_cropping
