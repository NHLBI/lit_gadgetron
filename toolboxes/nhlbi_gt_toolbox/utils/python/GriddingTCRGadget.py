import gadgetron
import numpy as np
import sigpy as sp
from sigpy.linop import NUFFT
import cupy as cp
import time
import ismrmrd as mrd
from sigpy.app import MaxEig
from sigpy.linop import FiniteDifference, Wavelet
import ctypes

import matplotlib.pyplot as plt

from utils.python.storage_server import Storage
from utils.python.tcr_utils import draw_number_indicators, compute_preemphasis_order, gram_schmidt, replace_zero, update_nonant, resize_with_crop_or_pad, analyticaldcf, remove_zero_padding, soft_threshold, crop_half_FOV, online_STCR_ISTA_2_timed, online_TCR_POGM_2, modified_gram_schmidt
from utils.python.tcr_utils import * 

from skimage.util import montage
from skimage.transform import resize
import ismrmrd as mrd



def pad_center_cupy(arr, new_shape, constant_values=0):
    """
    Resizes a CuPy array by padding it from the center.

    This function takes a smaller CuPy array and places it in the center of a
    new, larger array of a specified shape. The extra space is filled with a
    constant value (zero by default).

    Args:
        arr (cp.ndarray): The input CuPy array to be padded.
        new_shape (tuple): The desired output shape for the new array. Each
                           dimension of new_shape must be greater than or equal
                           to the corresponding dimension in the input array's shape.
        constant_values (scalar, optional): The value to use for padding.
                                            Defaults to 0.

    Returns:
        cp.ndarray: A new CuPy array with the specified shape, containing the
                    original array centered within it.

    Raises:
        ValueError: If any dimension in `new_shape` is smaller than the
                    corresponding dimension of the input `arr`.
    """
    # Get the shape of the input array
    old_shape = arr.shape

    # Validate that the new shape is not smaller than the old shape
    if any(new_dim < old_dim for new_dim, old_dim in zip(new_shape, old_shape)):
        raise ValueError("Each dimension of new_shape must be greater than or equal to the old shape.")

    # Calculate the padding widths for each dimensfion.
    # The padding is distributed as evenly as possible on both sides.
    # If the total padding for a dimension is odd, the extra padding
    # element is added to the end.
    pad_width = []
    for new_dim, old_dim in zip(new_shape, old_shape):
        # Total padding needed for the current dimension
        delta = new_dim - old_dim
        # Padding before the array
        pad_before = delta // 2
        # Padding after the array
        pad_after = delta - pad_before
        pad_width.append((pad_before, pad_after))

    # Use cupy.pad to apply the calculated padding
    padded_arr = cp.pad(arr, pad_width, mode='constant', constant_values=constant_values)

    return padded_arr


def create_ismrmrd_image(data, field_of_view, index, acquisition):

    data = 4000 * data / np.max(data)

    im = mrd.image.Image.from_array(
        np.float32(data),
        image_series_index=index,
        acquisition=acquisition,
        image_type=mrd.IMTYPE_MAGNITUDE,
        field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z),
        transpose=True
    )

    return im

# RR add in for iMRI
def send_imri_image(data, mrdHeader,connection,acq_data):
    # imagesOut = [None] * data.shape[-1]

    # Determine max value (12 or 16 bit)
    BitsStored = 12
    # fix this:
    # if (mrdhelper.get_userParameterLong_value(mrdHeader, "BitsStored") is not None):
    #     BitsStored = mrdhelper.get_userParameterLong_value(mrdHeader, "BitsStored")
    maxVal = 2**BitsStored - 1

    #data = abs(data)
    #data = np.flip(data,axis=2) # flip for scanner?


    
    data = np.abs(data)
    data *= maxVal / data.max()
    data = np.flip(data,axis=2) # flip for scanner?
    data = np.around(data)

    print(np.max(np.abs(data)))
    print(np.min(data))

    # add logic for over-sampled slices
    #mrdHeader.encoding[0].encodedSpace.matrixSize.z
    #mrdHeader.encoding[0].encodreconSpaceedSpace.matrixSize.z

    for partition in range(data.shape[-1]):
        # Create new MRD instance for the processed image
        #imagesOut = mrd.Image.from_array(data[...,partition], transpose=False)
        imagesOut = mrd.image.Image.from_array(
        np.float32(np.abs(data[:,:,partition])),
        acquisition=acq_data,
        image_series_index=0,
        repetition=acq_data._head.idx.repetition, #=0,
        image_type=mrd.IMTYPE_MAGNITUDE,
        transpose=True
        )

        #data.data = data.data[:,partition,:,:]

        # Set the header information
        # imagesOut.setHead(mrdhelper.update_img_header_from_raw(imagesOut.getHead(), rawHead[partition]))
        # check everything has been assigned

        res = mrdHeader.encoding[0].reconSpace.fieldOfView_mm.x/mrdHeader.encoding[0].reconSpace.matrixSize.x
        fovx = data.shape[0] * res

        imagesOut.field_of_view = (ctypes.c_float(fovx),
                                ctypes.c_float(fovx),
                                ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.z/int(mrdHeader.encoding[0].encodedSpace.matrixSize.z)))
   
        imagesOut.slice = partition
        
        # Positions need to be set for the scanner to be happy
        # -------------------------------------
        
        #par_thickness = mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.z/mrdHeader.encoding[0].encodedSpace.matrixSize.z

        par_thickness = mrdHeader.encoding[0].reconSpace.fieldOfView_mm.z/mrdHeader.encoding[0].reconSpace.matrixSize.z


        # determine rotation matrix in [PRS] - stacking per lit-python
        R_matrix = np.hstack((np.array(imagesOut.phase_dir).reshape(3,1),np.array(imagesOut.read_dir).reshape(3,1),np.array(imagesOut.slice_dir).reshape(3,1)))

        # partition is "third" dimension in matric coordinates
        #partition_vector = np.array([0,0,(partition-data.shape[-1]/2.0)*par_thickness])
        partition_vector = np.array([0,0,(partition-data.shape[-1]/2.0)*par_thickness + par_thickness*3.5]) # testing slice offset, adding fudge for slice matching to Siemens


        # multiply to determine appropriate offset position
        position_offset = np.matmul(R_matrix,partition_vector)
        for ii in range(3): # mrd wants c-float-array-3, and I dont know how to do a vector cast.. 
            # test casting # imagesOut.position[ii] = ctypes.c_float(imagesOut.position[ii] + position_offset[ii]) # added to test for imri
            imagesOut.position[ii] = imagesOut.position[ii] + position_offset[ii] # added to test for imri
        
        #print(np.array(imagesOut.position)) # you will regret this
        # -------------------------------------


        # Set ISMRMRD Meta Attributes
        # -------------------------------------
        # An update to FIRE might fix a few things, but fixing some meta data manually
        # note this only works with Fang's custom iMRI which helps us out
        # Make overlay comment (need to get SOP instance?)
        meta = mrd.Meta.deserialize(imagesOut.attribute_string)
        #meta['GADGETRON_ImageComment'] = 'Overlay'
        meta['DataRole'] = 'Image'
        meta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        meta['WindowCenter'] = str((maxVal + 1) / 2)
        meta['DataRole'] = str((maxVal + 1))

        # meta = ismrmrd.Meta({'DataRole': 'Image',
        #                             'ImageProcessingHistory': ['FIRE', 'PYTHON'],
        #                             'WindowCenter': str((maxVal + 1) / 2),
        #                             'WindowCenter': str((maxVal + 1))})
        
        # Add image orientation directions to MetaAttributes if not already present
        if meta.get('ImageRowDir') is None:
            meta['ImageRowDir'] = ["{:.18f}".format(imagesOut.getHead().read_dir[0]),
                                   "{:.18f}".format(imagesOut.getHead().read_dir[1]),
                                   "{:.18f}".format(imagesOut.getHead().read_dir[2])]

        if meta.get('ImageColumnDir') is None:
            meta['ImageColumnDir'] = ["{:.18f}".format(imagesOut.getHead().phase_dir[0]),
                                      "{:.18f}".format(imagesOut.getHead().phase_dir[1]),
                                      "{:.18f}".format(imagesOut.getHead().phase_dir[2])]

        meta['DimBoundariesIdc'] = 1
        meta['DimBoundariesIdd'] = 1
        meta['DimBoundariesIde'] = 1
        meta['DimBoundariesSeg'] = 1
        
        metaXml = meta.serialize()
        imagesOut.attribute_string = metaXml

        # Send device overlay
        connection.send(imagesOut)


def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

def GriddingTCRGadget(connection):
    if (connection.header.subjectInformation is not None) and (connection.header.subjectInformation.patientID is not None):
        subj_str = connection.header.subjectInformation.patientID
        print(subj_str)
    else:
        subj_str = connection.header.measurementInformation.measurementID.split("_")[1]
        print(subj_str)

    params = _parse_params(connection.config)

    ## GRIDDING parameters
    if "n_tr_bin" in params:
        n_tr_bin = int(params['n_tr_bin'])
    else:    
        n_tr_bin = 216

    ## MONTAGE parameters
    if "montage" in params:
        if params["montage"] == "True":
            bool_montage = True
        else:
            bool_montage = False
    else:
        bool_montage = False
    if "crop_percent" in params:
        crop_percent = float(params["crop_percent"])
    else:
        crop_percent = 0.6
    if "num_slices" in params:
        n_slices_extract = int(params["num_slices"])
    else:
        n_slices_extract = 1
    if "use_storage_server" in params:
        if params["use_storage_server"] == 'True':
            use_storage_server = True
        else:
            use_storage_server = False
    else:
        use_storage_server = False
    if "alg_type" in params:
        alg_type = params["alg_type"]
    else:
        alg_type = "gridding"
    
    ## STCR Parameters
    if "niter" in params:
        niter = int(params["niter"])
    else:
        niter = 0
    if "lambdat" in params:
        lambdat = float(params["lambdat"]) 
    else:
        lambdat = 0
    if "lambdas" in params:
        lambdas = float(params["lambdas"]) 
    else:
        lambdas = 0
    
    ## ROVir Parameters
    if "n_rovir_coils" in params:
        n_rovir_coils = int(params["n_rovir_coils"])
    else:
        n_rovir_coils = 0
    
    ## Analytical DCF
    if "recompute_dcf" in params:
        if params["recompute_dcf"] == "True":
            recompute_dcf = True
        else:
            recompute_dcf = False
    else:
        recompute_dcf = False
    
    if "recompute_L" in params:
        if params["recompute_L"] == "True":
            recompute_L = True
        else:
            recompute_L = False
    else:
        recompute_L = False
    
    if "time_per_frame_scale" in params:
        time_per_frame_scale = float(params["time_per_frame_scale"])
    else:
        time_per_frame_scale = 0.7

    if "max_frame_discard" in params:
        MAX_FRAME_DISCARD = int(params["max_frame_discard"])
    else:
        MAX_FRAME_DISCARD = 6
    
    center_shift = 0

    storage = Storage("localhost", 9112, subject=subj_str)
    storage_vars = Storage("localhost", 9112)

    # variable initializations
    L = None
    n_frame_discard = 0
    n_frame_discard = MAX_FRAME_DISCARD
    csm_loaded = False 
    n_connections = 0
    kspace_buff = []
    head_buff = []
    first_frame = True
    n_images_sent = 0
    was_lip_computed = False
    recomputed_dcf = False

    maximum_scale = None

    for data in connection:
        st = time.time()

        # for now, very dummy buffer for acquiring the n_tr_bin.
        # this can be optimized to not have to wait for an entire volume, but a simple case for now!
        if n_tr_bin == 0:
            if data.is_flag_set(mrd.ACQ_LAST_IN_SLICE):
                n_tr_bin = data.scan_counter
                print(f"n_tr_bin: {n_tr_bin}")
            else:
                continue

        if csm_loaded is False:
            # setup all the beginning variables!
            csm_loaded = True
            csm_str = f"{connection.header.encoding[0].encodedSpace.matrixSize.x}_{connection.header.encoding[0].encodedSpace.matrixSize.y}_{connection.header.encoding[0].encodedSpace.matrixSize.z}_{data._head.active_channels}"

            print(f"CSM STR: {csm_str}")

            # PK HACK OF EPIC PROPORTIONS
            #csm_str = "112_112_48_21"

            csm = storage.fetch_latest(name=f"csm_{csm_str}")
            if n_rovir_coils > 0:
                W_d = storage.fetch_latest(f"ROVIR_W_{csm_str}")
                W = W_d['W']
                #csm_rovir = W_d['csm_rovir'][0:n_rovir_coils,:,:,:]
                W, th = modified_gram_schmidt(cp.array(W[:,0:n_rovir_coils]))
                csm_rovir = rovir_apply(cp.array(csm.transpose((1,2,3,0))), cp.array(W), n_rovir_coils).transpose((3,0,1,2))
                csm = crop_half_FOV(csm_rovir, dims=(2,3), size=np.flip(csm.shape[1:]))
                # grahm schmidt orthogonalization
                #Q, R = np.linalg.qr(W[:,0:n_rovir_coils])
                #W = cp.array(Q)
                #csm = (csm.transpose((1,2,3,0)) @ W).transpose((3,0,1,2))

            #csm = remove_zero_padding(csm, 1) # gt gridding artifact
            #csm = cp.flip(csm, axis=1)
            csm = cp.asarray(csm).transpose((0,3,2,1))           
            n_coils = csm.shape[0]

            traj_complex = storage.fetch_latest("trajectory")
            traj = np.stack((traj_complex.real, traj_complex.imag), axis=-1)
            dcf = cp.array(storage.fetch_latest(name=f"dcf_{csm_str}"))

            nread = traj.shape[1]*n_tr_bin
            dcf_sqrt = cp.sqrt(dcf[0:nread])
            What = sp.linop.Multiply([csm.shape[0], nread], dcf_sqrt)
            S = sp.linop.Multiply(csm.shape, csm, conj=True)
            R = sp.linop.Sum(csm.shape, [0])
            kspace_buff = cp.zeros((data._head.active_channels, n_tr_bin, traj.shape[1]), dtype=cp.complex64)

            csm_image = storage.fetch_latest(name=f"csm_image_{csm_str}")
            # csm_image = remove_zero_padding(csm_image, 1) # gt gridding artifact
            csm_image = cp.array(csm_image).transpose((2,1,0))
            #csm_image = crop_half_FOV(csm_image, dims=[0,1])

            G = FiniteDifference(csm_image.shape)

            # try to pull the pre-computed L
            L_scale_csm_scale_str = connection.header.measurementInformation.protocolName + "_len_" + str(n_tr_bin)
            try:
                print(f"L_scale_csm: {L_scale_csm_scale_str}")
                storage_scale_fetch = storage.fetch_latest(custom_tags={"scale_factor_python":L_scale_csm_scale_str})
                L = storage_scale_fetch['L']
                print(f"FETCHED CSM L: {L}")
            except:
                print("failed to fetch storage server. Will re-generate....")
            if recompute_L:
                L = None

            print(f"TIME SPENT IN FIRST FRAME: {time.time() - st}")
         
        TR = connection.header.sequenceParameters.TR
        time_per_frame = n_tr_bin * TR[0]

        kspace_buff[:,n_connections,:] = cp.array(data.data)
        head_buff.append(data._head)

        # bin up the data for gridding
        if n_connections == (n_tr_bin-1):
            n_connections = 0
            encode_step_1 = [head_buff[i].idx.kspace_encode_step_1 for i in range(n_tr_bin)]
            encode_step_2 = [head_buff[i].idx.kspace_encode_step_2 for i in range(n_tr_bin)]
            head_buff = []
            n_slice = csm.shape[3]

            trajectory_frame = np.zeros((n_tr_bin, traj.shape[1], 3))
            trajectory_frame[:,:,0:2] = traj[encode_step_1] * csm.shape[1]

            trajectory_frame[:,:,2] = ((np.repeat(encode_step_2, traj.shape[1]).reshape((n_tr_bin,traj.shape[1])) / n_slice) - 0.5) * n_slice
            trajectory_frame = trajectory_frame.reshape((n_tr_bin * traj.shape[1], 3))
            trajectory_frame = cp.array(trajectory_frame)

            if recompute_dcf and not recomputed_dcf:
                print("recomputing the DCF...")
                dcf = analyticaldcf(trajectory_frame.get(), ns=nread)
                dcf_sqrt = cp.sqrt(cp.array(dcf))
                What = sp.linop.Multiply([csm.shape[0], nread], dcf_sqrt) 
                recomputed_dcf = True

            F = NUFFT(csm.shape, trajectory_frame, toeplitz=True)
            #Ah = R * S * F.H * What
            Aframe = What * F * S.H * R.H

            # apply rovir if necessary
            if n_rovir_coils > 0:
                kspace_rov = rovir_apply(kspace_buff.transpose(1,2,0), cp.array(W), n_rovir_coils).transpose(2,0,1)
                y = (kspace_rov.reshape(kspace_rov.shape[0], -1) * dcf_sqrt)
                image = Aframe.H * y 
                n_coils = n_rovir_coils
            else:
                y = (kspace_buff.reshape(kspace_buff.shape[0], -1) * dcf_sqrt)
                image = Aframe.H * y
            # precompute L if necessary
            if L is None:
                L  = MaxEig(Aframe.N, max_iter=40, dtype=image.dtype, device=image.device, show_pbar=True).run()
                print("Lipschitz Constant: {}".format(L))
                n_frame_discard = MAX_FRAME_DISCARD 
                was_lip_computed = True
                storage.store({"L": L}, custom_tags={"scale_factor_python": L_scale_csm_scale_str})
            cp.cuda.stream.get_current_stream().synchronize()
            print(f" time spent GRIDDING {time.time() - st}")

            def cost_stcr(x):
                return 0.5 * cp.square(cp.linalg.norm((Aframe * (x + xn_1) - y).flatten())) + \
                    (lambdat * cp.linalg.norm(x.flatten(), ord=1)) + (lambdas * cp.linalg.norm((G * x).flatten(), ord=1))

            def cost_tcr(x):
                return 0.5 * cp.square(cp.linalg.norm((Aframe * (x + xn_1) - y).flatten())) + \
                    (lambdat * cp.linalg.norm(x.flatten(), ord=1))
            
            stcrt = time.time()
            # do STCR, or skip a few frames if needed to make up for L computation
            if niter > 0 and n_frame_discard == 0:
                if first_frame or (was_lip_computed and n_images_sent == (MAX_FRAME_DISCARD)):
                    max_image = cp.abs(image).max()
                    scale_factor = cp.abs(csm_image).max() / max_image 
                    del_0 = csm_image / scale_factor
                    xn = csm_image / scale_factor
                else:
                    max_image = cp.abs(image).max()
                    scale_factor = cp.abs(csm_image).max() / max_image 

                    lamt = lambdat * max_image 
                    lams = lambdas * max_image

                    if alg_type == "cg":
                        cg_alg = sp.alg.ConjugateGradient(Aframe.H*Aframe, image, xn_1, max_iter=niter)
                        while not cg_alg.done():
                            cg_alg.update()
                        xn = cg_alg.x
                    elif alg_type == "stcr":
                        del_0 = (image - xn_1) / 4 # soft initialization
                        print(time_per_frame)
                        del_0 = online_STCR_ISTA_2_timed(Aframe, G, xn_1, image, lamt, lams, 1/L, mu=0.1, yn=(kspace_buff.reshape(kspace_buff.shape[0], -1) * dcf_sqrt), time_recon=time_per_frame*time_per_frame_scale, deln=del_0)
                        xn = xn_1 + del_0
                    elif alg_type == "tcr":
                        del_0 = cp.zeros(image.shape)
                        del_0 = online_TCR_POGM_2(Aframe, xn_1, image, lamt, (1/L), n_iter=niter, del_0=del_0)
                        xn = xn_1 + del_0
                    elif alg_type == "stcr_l1_approx":
                        xn = xn_1
                        nesterov_dict = None
                        for it in range(niter):
                            delta = 1
                            #xn = gradient_descent_iteration(xn, lambda d : grad_cost(d, xn_1, Aframe, y, G, delta, lamt, lams), 1/(L+(2/delta)))
                            xn, nesterov_dict = gradient_descent_iteration_nesterov(it, xn, lambda d : grad_cost(d, xn_1, Aframe, y, G, delta, lamt, lams), 1/(L+(2/delta)), max_iter=niter, nesterov_dict=nesterov_dict)
                    elif alg_type == "gridding":
                        xn = image
                    
                    print(f" ALG TYPE: {alg_type}")
            else:
                n_frame_discard = max(0, n_frame_discard-1)
                xn = image
            
            print(f" RECON MATRIX SIZE: {xn.shape}")
            # setup image out
            xn_1 = cp.copy(xn)

            # do a fourier interpolation to 2x the matrix size
            #xn = cp.fft.fftshift(cp.fft.fftn(xn, axes=(0,1,2)), axes=(0,1,2))
            #xn = pad_center_cupy(xn, [xn.shape[0]*2,xn.shape[1]*2,xn.shape[2]*2])
            #xn = cp.fft.ifftn(cp.fft.ifftshift(xn, axes=(0,1,2)), axes=(0,1,2))

            # do we want to montage?
            #if bool_montage is True:
            if False:
                if use_storage_server:
                    montage_info = storage_vars.fetch_latest(custom_tags={"montage":"1"})
                    if 'center_shift' in montage_info:
                        center_shift = int(montage_info['center_shift'])
                        n_slices_extract = int(montage_info['n_slices'])
                        # we have images x, y, f. 
                        image_for_m = crop_middle(extract_middle_slices(cp.abs(xn_1).squeeze(), n_slices_extract, 2, center_shift), [0,1], crop_percent)
                        image_for_m = draw_number_indicators(image_for_m, indicator_size=8)
                        image_for_m = image_for_m[:,:,compute_preemphasis_order(image_for_m.shape[2])]
                        image_m = montage(image_for_m.transpose((2,0,1)).get())
                    elif 'oreintation_list' in montage_info:
                        # do something else
                        # list is of type ['a21', 'a22', 'a23', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
                        # number is index, and letter is a for axial, s for sagittal, c for coronal.
                        orientations = montage_info['oreintation_list']
                        image_for_m = []
                        for nonant_str in orientations:
                            image_for_m.append(update_nonant(xn, nonant_str).get())
                        image_for_m = np.array(image_for_m)
                        image_for_m[:,:,compute_preemphasis_order(image_for_m.shape[0])]
                        image_m = montage(image_for_m) 
                    print(f"{montage_info}")
            else:
                image_m = xn_1.get()
            
            # if it's the first image we output, we need the size to always be the same. So do a re-size if needed to match the first frame size.
            if first_frame:
                first_shape = image_m.shape
                first_frame = False
            else:
                if not image_m.shape == first_shape:
                    print("resizing...")
                    image_m = resize(image_m, first_shape) 
            
            # setup image and send.
            n_images_sent = n_images_sent + 1
            if n_images_sent > MAX_FRAME_DISCARD and maximum_scale is None:
                maximum_scale = np.abs(image_m).max() * 0.7
            if maximum_scale is not None:
                # everywhere the image > maximum_scale, set to maximum_scale (but keep the phase)
                #image_m = np.where(np.abs(image_m) > maximum_scale, maximum_scale * np.exp(1j * np.angle(image_m)), image_m)
                pass
            #out_im = create_ismrmrd_image(np.abs(image_m), connection.header.encoding[0].encodedSpace.fieldOfView_mm, 0, data)
            cp.cuda.stream.get_current_stream().synchronize()
            print(f"PYTHON TOTAL ELAPSED TIME: {time.time() - st}")

            if bool_montage:
                out_im = create_ismrmrd_image(np.abs(image_m), connection.header.encoding[0].encodedSpace.fieldOfView_mm, 0, data)
                connection.send(out_im)
            else:
                # we could, instead of sending gridding images, just ignore them.
                if n_images_sent > MAX_FRAME_DISCARD:
                    print("send_imri_image called")
                    send_imri_image(image_m, connection.header,connection, data)
        else:
            n_connections = n_connections + 1

    # cleaup
    storage.close()
    storage_vars.close()
        
if __name__ == '__main__':
    gadgetron.external.listen(2000, GriddingTCRGadget)