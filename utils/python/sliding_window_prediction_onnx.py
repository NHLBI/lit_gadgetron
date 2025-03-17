##Adapt script of nnUnet package for only numpy (remove Pytorch dependency)
#from https://github.com/pdaude/nnUNet/blob/1ee971829b0e1a7369b0bf0c598d5a26b0cd2160/nnunetv2/inference/sliding_window_prediction.py


import numpy as np
from typing import Union, Tuple, List
from acvl_utils_numpy  import pad_nd_image
from scipy.ndimage import gaussian_filter
from onnxruntime import InferenceSession


def compute_gaussian(tile_size: Tuple[int, ...], sigma_scale: float = 1. / 8, dtype=np.float16) \
        -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def get_sliding_window_generator(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float,
                                 verbose: bool = False):
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, 'if tile_size has less entries than image_size, len(tile_size) ' \
                                                      'must be one shorter than len(image_size) (only dimension ' \
                                                      'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)
        if verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for d in range(image_size[0]):
            slice_z=d if image_size[0]>1 else slice(None,None,None)
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple([slice(None), slice_z, *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)]])
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)
        if verbose: print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]])
                    yield slicer


def maybe_mirror_and_predict(network: InferenceSession, x: np.ndarray, mirror_axes: Tuple[int, ...] = None) -> np.ndarray:
    input_name=network.get_inputs()[0].name
    x=x.astype(np.float32)
    ort_inputs={input_name: x}
    prediction = network.run(None,ort_inputs)[0]
    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            ort_inputs={input_name: np.flip(x, (2,))}
            prediction += np.flip(network.run(None,ort_inputs)[0], (2,))

        if 1 in mirror_axes:
            ort_inputs={input_name: np.flip(x, (3,))}
            prediction += np.flip(network.run(None,ort_inputs)[0], (3,))

        if 2 in mirror_axes:
            ort_inputs={input_name: np.flip(x, (4,))}
            prediction += np.flip(network.run(None,ort_inputs)[0], (4,))

        if 0 in mirror_axes and 1 in mirror_axes:
            ort_inputs={input_name: np.flip(x, (2, 3))}
            prediction += np.flip(network.run(None,ort_inputs)[0], (2, 3))

        if 0 in mirror_axes and 2 in mirror_axes:
            ort_inputs={input_name: np.flip(x, (2, 4))}
            prediction += np.flip(network.run(None,ort_inputs)[0], (2, 4))

        if 1 in mirror_axes and 2 in mirror_axes:
            ort_inputs={input_name: np.flip(x, (3, 4))}
            prediction += np.flip(network.run(None,ort_inputs)[0], (3, 4))

        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            ort_inputs={input_name: np.flip(x, (2, 3, 4))}
            prediction += np.flip(network.run(None,ort_inputs)[0], (2, 3, 4))

        prediction /= num_predictons
    return prediction


def predict_sliding_window_return_logits(networks:List[InferenceSession],input_image: np.ndarray,
                                         num_segmentation_heads: int,
                                         tile_size: Tuple[int, ...],
                                         mirror_axes: Tuple[int, ...] = None,
                                         tile_step_size: float = 0.5,
                                         use_gaussian: bool = True,
                                         precomputed_gaussian: np.ndarray = None,
                                         verbose: bool = False) -> np.ndarray:

    
    assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

    # if input_image is smaller than tile_size we need to pad it to tile_size.
    data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'constant_values': 0}, True, None)
    

    if use_gaussian:
        if precomputed_gaussian is None:
            gaussian = compute_gaussian(tile_size, sigma_scale=1. / 8)
        else:
            gaussian =precomputed_gaussian
        mn = gaussian.min()
        if mn == 0:
            gaussian.clip_(min=mn)
    else:
        gaussian=1

    slicers = get_sliding_window_generator(data.shape[1:], tile_size, tile_step_size, verbose=False)
    predicted_logits = np.zeros((data.shape[0],num_segmentation_heads, *data.shape[1:]), dtype=np.float16)
    n_predictions = np.zeros(data.shape, dtype=np.float16)
    for sl in slicers:
        workon = data[sl]
        if data.shape[1]==1:
            sl_logits=tuple([slice(None), slice(None),0,*sl[2:]])
        else:
            sl_logits=tuple([slice(None),*sl[1:]])
        for network  in networks :
            prediction = maybe_mirror_and_predict(network, workon, mirror_axes)
            if use_gaussian:
                prediction *= gaussian
            predicted_logits[sl_logits] += prediction
            n_predictions[sl] += gaussian
            

    predicted_logits /= n_predictions[:,None,...]
    
    return predicted_logits[tuple([slice(None),slice(None), *slicer_revert_padding[1:]])]
