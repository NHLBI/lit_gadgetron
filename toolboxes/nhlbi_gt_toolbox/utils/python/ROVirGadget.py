import gadgetron
import ismrmrd as mrd

# non-debugging
#from utils.python.storage_server import Storage
#from utils.python.tcr_utils import rovir_apply, crop_half_FOV
#from utils.python.coil_maps import calculate_csm_walsh_gpu

#debugging
from storage_server import Storage
from tcr_utils import *
from coil_maps import calculate_csm_walsh_gpu

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.linalg import eig
import cupy as cp

from sigpy import fft
from sigpy.mri.app import EspiritCalib

from skimage.morphology import convex_hull_image


def automatic_mask_3D(image, threshold_ratio=0.1):
    """
    Generates a 3D mask by thresholding the image and computing the convex hull 
    for each slice along the z-axis (axis 2).

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array (x, y, z) representing the volume.
    threshold_ratio : float
        The fraction of the maximum intensity to use as a threshold 
        (default is 0.1, i.e., 10%).

    Returns:
    --------
    mask : numpy.ndarray
        A boolean 3D numpy array of the same shape as input image, 
        where True indicates the mask region.
    """
    
    # 1. Basic validation
    if image.ndim != 3:
        raise ValueError("Input image must be a 3D array.")

    # 2. Determine Signal Region (Thresholding)
    # We use the max value of the entire 3D volume
    max_val = np.max(image)
    threshold = max_val * threshold_ratio
    
    # Create initial binary mask
    # We treat any value > threshold as signal
    signal_mask = image > threshold

    # 3. Convex Hull per Slice
    # We assume 'z' is the last axis (axis 2). 
    # If your data is (z, x, y), change the iteration accordingly.
    
    final_mask = np.zeros_like(signal_mask, dtype=bool)
    
    depth = image.shape[2] # z-axis
    
    for z in range(depth):
        slice_2d = signal_mask[:, :, z]
        
        # convex_hull_image requires at least one True value to work.
        # If the slice is empty (no signal), the mask for this slice remains False.
        if np.any(slice_2d):
            # Compute convex hull for this slice
            hull_2d = convex_hull_image(slice_2d)
            final_mask[:, :, z] = hull_2d
        else:
            # Keep as zeros/False
            final_mask[:, :, z] = slice_2d

    # Optional: Fill holes in 3D if needed, though slice-by-slice hull usually suffices 
    # for "wrapping" the object. 
    # To make it a solid 3D block if there are gaps between slices, 
    # one might perform binary_closing or binary_fill_holes in 3D, 
    # but the requirement was specifically "convex hull in each slice".
    return split_mask_by_extent(final_mask, central_ratio=0.7, axis=0)

def split_mask_by_extent(mask, central_ratio=0.8, axis=2):
    """
    Splits a 3D mask into two regions: a central region and an off-region 
    (upper and lower tips), based on the extent of the mask along the specified axis.

    Parameters:
    -----------
    mask : numpy.ndarray
        A 3D boolean numpy array.
    central_ratio : float
        The proportion of the extent to keep in the central mask.
    axis : int
        The dimension along which to calculate extent and split.

    Returns:
    --------
    central_mask : numpy.ndarray
    off_mask : numpy.ndarray
    """
    if mask.ndim != 3:
        raise ValueError("Input mask must be a 3D array.")
    
    # Move target axis to 0 for easy processing
    # If shape is (X, Y, Z) and axis is 2, shape becomes (Z, X, Y)
    mask_moved = np.moveaxis(mask, axis, 0)
    
    # 1. Find indices along axis 0 where mask has positive values
    # mask_moved has shape (Depth, H, W). We collapse (1, 2)
    presence = np.any(mask_moved, axis=(1, 2))
    indices = np.where(presence)[0]

    central_moved = np.zeros_like(mask_moved)
    off_moved = np.zeros_like(mask_moved)

    # Handle empty mask case
    if len(indices) == 0:
        return np.zeros_like(mask), np.zeros_like(mask)

    # 2. Determine extent
    idx_min = indices[0]
    idx_max = indices[-1]
    
    length = idx_max - idx_min
    margin = length * (1.0 - central_ratio) / 2.0
    
    cutoff_bottom = idx_min + margin
    cutoff_top = idx_max - margin

    # 3. Create the masks
    depth = mask_moved.shape[0]
    
    for i in range(depth):
        if presence[i]:
            if cutoff_bottom <= i <= cutoff_top:
                central_moved[i, ...] = mask_moved[i, ...]
            else:
                off_moved[i, ...] = mask_moved[i, ...]

    # 4. Move axis back to original position
    central_mask = np.moveaxis(central_moved, 0, axis)
    off_mask = np.moveaxis(off_moved, 0, axis)
                
    return central_mask, off_mask

def get_ES_mask_3D(im3d, mode='simple', splitud=False, scale_amt=2, margin=10):
    """
    Generates a mask for a 3D image.

    Parameters:
    - im3d: 3D numpy array (the input image).
    - mode: 'simple' will not use thresholding (large rectangles), 'advanced' will use thresholding (smaller regions).
    - splitud: bool, whether to create two masks (default False).
    - scale_amt: float, intensity scaling factor (default 2).

    Returns:
    - mask: 3D numpy array with the mask applied.
    """
    matrix_size = im3d.shape
    ny, nx, n_slice = matrix_size

    mask = np.zeros(matrix_size)

    if mode == 'simple':
        mask_top = np.zeros(matrix_size)
        mask_bot = np.zeros(matrix_size)
        mask_fov = np.zeros(matrix_size)

        mask_top[:int(ny // 2.5), :, :] = 1
        mask_bot[3*ny // 4:, :, :] = 1

        mask = mask_top + mask_bot

        # mask_fov is the center 40% of image
        #mask_fov[ny // 3:2 * ny // 3, nx // 3:2 * nx // 3, (n_slice//6):(n_slice*5//6)] = 1
        mask_fov[ny // 5:4 * ny // 5, nx // 5:4 * nx // 5, (n_slice//6):(n_slice*5//6)] = 1

        return (mask, mask_fov)
    else:
        # Gaussian filter
        im_filt = gaussian_filter(im3d, sigma=2)

        # Split the filtered image if needed
        if splitud:
            im_filt_top = im_filt.copy()
            im_filt_bot = im_filt.copy()
            
            im_filt_top[ny // 2:] = 0
            im_filt_bot[:ny // 2] = 0
            
            im_split = [im_filt_top, im_filt_bot]
        else:
            im_split = [im_filt]

        for im_tmp in im_split:
            M = np.zeros(n_slice)
            row = np.zeros(n_slice, dtype=int)
            col = np.zeros(n_slice, dtype=int)

            for slc in range(n_slice):
                # Find the max value and its index for each slice
                max_val = im_tmp[:, :, slc].max()
                M[slc] = max_val
                row[slc], col[slc] = np.unravel_index(np.argmax(im_tmp[:, :, slc]), im_tmp[:, :, slc].shape)

            win_size = 40

            for slc in range(n_slice):
                row_bounds = [np.clip(row[slc] - win_size, 0, nx),
                            np.clip(row[slc] + win_size, 0, nx)]
                col_bounds = [np.clip(col[slc] - win_size, 0, ny),
                            np.clip(col[slc] + win_size, 0, ny)]

                mask1 = np.zeros(im_tmp.shape[:2])
                mask1[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]] = 1
                image_slice = im_tmp[:, :, slc]
                mask2 = np.zeros_like(mask1)
                mask2[(image_slice * mask1) > (M[slc] / scale_amt)] = 1
                mask[:, :, slc] += mask2

        # Final FOV check: don't select inside FOV
        margin = margin 
        fov_bound2dy = [ny // 2 - (ny // margin), ny // 2 + (ny // margin)]
        fov_bound2dx = [nx // 2 - (nx // margin), nx // 2 + (nx // margin)]
        
        fov_mask = np.zeros((nx, ny))
        fov_mask[fov_bound2dy[0]:fov_bound2dy[1], fov_bound2dx[0]:fov_bound2dx[1]] = 1

        fov_mask = fov_mask[:,:,np.newaxis]
        mask = np.multiply(mask, (1-fov_mask))

        fov_mask = np.ones(mask.shape) * fov_mask

        return (mask, fov_mask)

def rovir_automatic_3d_pk_2(img_cc, csm_image_mc_, margin=5):
    [rovir_sig_, rovir_int_] = automatic_mask_3D(np.abs(img_cc))
    nx, ny, nz = img_cc.shape
    n_coil = csm_image_mc_.shape[-1]
    rovir_sig_mask = rovir_sig_[:,:,:,np.newaxis]
    rovir_int_mask = rovir_int_[:,:,:,np.newaxis]
    signals = (rovir_sig_mask * csm_image_mc_).reshape(-1, n_coil)
    interferences = (rovir_int_mask * csm_image_mc_).reshape(-1, n_coil)

    A = np.conj(signals.T) @ signals
    B = np.conj(interferences.T) @ interferences

    # solve for generalized eigenvalues
    eigvals, eigvecs = eig(B, A)

    # Sort by eigenvalues in descending order
    idx = np.argsort(np.abs(eigvals))[::-1]

    # Normalize eigenvectors
    V = eigvecs[:, np.flip(idx)]

    #V = V / np.linalg.norm(V, axis=0)
    #np.save("W_rovir.npy", V)

    return [V, rovir_sig_, rovir_int_]

def rovir_automatic_3d_pk(img, img_coils, margin=10):
    """
    Automatically selects the signal region from a 3D image.

    Parameters:
    - img: 3D numpy array (the main image data).
    - img_coils: 4D numpy array (coil images).

    Returns:
    - W: The ordered eigenvector matrix.
    """
    [rovir_int_mask, rovir_sig_mask] = get_ES_mask_3D(np.abs(img), splitud=True, scale_amt=3, margin=margin)
    
    # Automatically select signal region
    nx, ny, nz = img.shape
    margin = 5
    margin_z = 3
    
    """
    fov_bound2dy = [int(np.floor((ny / 2) - (ny / margin))), int(np.floor((ny / 2) + (ny / margin)))]
    fov_bound2dx = [int(np.floor((nx / 2) - (nx / margin))), int(np.floor((nx / 2) + (nx / margin)))]
    fov_bound2dz = [int(np.floor((nz / 2) - (nz / margin_z))), int(np.floor((nz / 2) + (nz / margin_z)))]

    rovir_sig_mask[fov_bound2dy[0]:fov_bound2dy[1], 
                   fov_bound2dx[0]:fov_bound2dx[1], 
                   fov_bound2dz[0]:fov_bound2dz[1]] = 1
    """

    n_coil = img_coils.shape[3]
    rovir_sig_mask = rovir_sig_mask[:,:,:,np.newaxis]
    rovir_int_mask = rovir_int_mask[:,:,:,np.newaxis]
    signals = (rovir_sig_mask * img_coils).reshape(-1, n_coil)
    interferences = (rovir_int_mask * img_coils).reshape(-1, n_coil)

    A = signals.T @ signals
    B = interferences.T @ interferences

    # Solve for generalized eigenvalues
    eigvals, eigvecs = eig(B, A)
   # np.linalg.eig(np.linalg.pinv(A) @ B)

    # Sort by eigenvalues in descending order
    idx = np.argsort(np.abs(eigvals))[::-1]

    # Normalize eigenvectors
    V = eigvecs[:, np.flip(idx)]
    V = V / np.linalg.norm(V, axis=0)

    # Reorder W
    W = V

    #np.save("W_rovir.npy", W)

    return W

def remove_zero_padding(arr, axis):
    """
    Remove zero padding along a specified axis in a NumPy array.

    Parameters:
    arr (np.ndarray): Input array with potential zero padding.
    axis (int): Axis along which to remove zero padding.

    Returns:
    np.ndarray: Array with zero padding removed along the specified axis.
    """
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"Invalid axis {axis} for array with {arr.ndim} dimensions.")
    
    # Create a boolean mask where True represents non-zero slices
    non_zero_mask = np.any(arr != 0, axis=tuple(i for i in range(arr.ndim) if i != axis))
    
    # Use the boolean mask to slice the array
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None)  # Keep the full range for the axis we are slicing
    arr_sliced = arr[tuple(slices)]
    
    # Now use the boolean mask to filter out zero-only slices
    arr_trimmed = arr_sliced.take(np.where(non_zero_mask)[0], axis=axis)
    
    return arr_trimmed


def ROVirGadget(connection):
    if connection.header.subjectInformation.patientID is not None:
        subj_str = connection.header.subjectInformation.patientID
    else:
        subj_str = connection.header.measurementInformation.measurementID.split("_")[1]
    storage = Storage("localhost", 9112, subject=subj_str)
    
    setup = None
    bool_espirit = True

    for data in connection:
        if setup is None:
            setup = True
            size_path = f"{connection.header.encoding[0].encodedSpace.matrixSize.x}_{connection.header.encoding[0].encodedSpace.matrixSize.y}_{connection.header.encoding[0].encodedSpace.matrixSize.z}_{data.acq_headers[0].active_channels}"
            csm_image_mc = storage.fetch_latest(name=f"csm_image_mc_{size_path}")
            csm_over = storage.fetch_latest(name=f"csm_over_{size_path}")
            img_cc = remove_zero_padding(np.sum(np.conj(csm_over) * csm_image_mc,axis=0),0).transpose((2, 1, 0))
            csm_image_mc_ = remove_zero_padding(csm_image_mc, 1).transpose((3,2,1,0))

            if bool_espirit:
                ksp = fft(remove_zero_padding(csm_image_mc, 1), axes=[-1, -2, -3])
                ksp = cp.array(ksp)
                mps_rec = EspiritCalib(ksp, show_pbar=True, thresh=0.02, crop=0, device=cp.cuda.Device(device=0)).run()
                
                matrix_keep = int(connection.header.encoding[0].encodedSpace.matrixSize.x*1.5)
                mps_rec = crop_half_FOV(mps_rec, dims=(-1, -2), size=[matrix_keep,matrix_keep])
                storage.store(mps_rec, name=f"csm_{size_path}")

            [W, rovir_sig_, rovir_int_] = rovir_automatic_3d_pk_2(img_cc, csm_image_mc_)

            if True:
                rov_ims = rovir_apply(cp.array(csm_image_mc.transpose(1,2,3,0)), cp.array(W), 10).transpose(3,0,1,2)
                # calculate sensitivity maps slice by slice.
                rov_ims = cp.array(remove_zero_padding(rov_ims, 1))
                csm_rovir = np.zeros(rov_ims.shape, dtype=np.complex64)

                if bool_espirit:
                    ksp = fft(remove_zero_padding(rov_ims, 1), axes=[-1, -2, -3])
                    ksp = cp.array(ksp)
                    csm_rovir = EspiritCalib(ksp, show_pbar=True, device=cp.cuda.Device(device=0)).run()
                else:
                    for sl in range(rov_ims.shape[1]):
                        print(f"{sl} out of ")
                        [csm_, rho] = calculate_csm_walsh_gpu(rov_ims[:,sl,:,:].squeeze())
                        csm_rovir[:,sl:sl+1,:,:] =  csm_
                    #storage.store({f"W":W, "csm_rovir": csm_rovir}, name=f"ROVIR_W_{size_path}")
                storage.store({f"W":W, "csm_rovir": csm_rovir, "rovir_sig": rovir_sig_, "rovir_int": rovir_int_}, name=f"ROVIR_W_{size_path}")
            else:
                storage.store({f"W":W}, name=f"ROVIR_W_{size_path}")

        connection.send(data)
        
if __name__ == '__main__':
    gadgetron.external.listen(2020, ROVirGadget)