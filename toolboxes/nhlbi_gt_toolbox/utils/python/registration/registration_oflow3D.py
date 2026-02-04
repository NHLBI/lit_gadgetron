import warnings
warnings.filterwarnings("ignore")
#import cv2
import opticalflow3D
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates, zoom
from skimage.transform import warp
import numpy as np
import matplotlib.pyplot as plt
import gc
import torch

# from numba import config
# config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True
def get_GPU_most_free():

    numdevices =  torch.cuda.device_count()
    GPU_free=0
    dev_GPU=-1
    for devno in range(numdevices):
        with torch.cuda.device(devno):
            f,t = torch.cuda.mem_get_info() 
            if GPU_free < f:
                GPU_free = f
                dev_GPU=devno
        
    return GPU_free,dev_GPU

def register_images_only_deformation(input_images, ref_index, filter_size=9, gpu_id=0):
    # Set the specific GPU device
    with cp.cuda.Device(gpu_id):
        # Convert images to CuPy arrays and initialize output arrays
        images = cp.asarray(input_images)
        deformation_fields = np.zeros([images.shape[0], 3, images.shape[1], images.shape[2], images.shape[3]])
        nimages, nr, nc, nz = images.shape

        # Reference image
        ref_image = images[ref_index, ...].squeeze()

        cp.cuda.runtime.deviceSynchronize()
        for ind in range(0, nimages):
            mov_image = images[ind, ...].squeeze()
            farneback = opticalflow3D.Farneback3D(iters=5,
                                    num_levels=5,
                                    scale=0.5,
                                    filter_size=filter_size,
                                    presmoothing=0, # Default, none
                                    filter_type="gaussian",
                                    sigma_k=0.05)

            output_vx, output_vy, output_vz, x = farneback.calculate_flow(
                0.05 * cp.abs(ref_image / cp.max(ref_image.ravel())),
                0.05 * cp.abs(mov_image / cp.max(mov_image.ravel())),
                total_vol=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                sub_volume=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                overlap=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                threadsperblock=(8, 8, 8),
            )

            deformation_fields[ind, ...] = np.concatenate([np.expand_dims(output_vx, 0), np.expand_dims(output_vy, 0), np.expand_dims(output_vz, 0)], axis=0)
            deformation_fields[ind, ...] = np.nan_to_num(deformation_fields[ind, ...])


        del output_vx
        del output_vy
        del output_vz
        del farneback
        del x
        del images
        del ref_image

        gc.collect()
        return deformation_fields
    
def register_one_image_only_deformation(mov_image_np, ref_image_np, filter_size=9,gpu_id=0):
    # Set the specific GPU device
    with cp.cuda.Device(gpu_id):
        # Convert images to CuPy arrays and initialize output arrays
        ref_image=cp.asarray(ref_image_np)
        mov_image=cp.asarray(mov_image_np)
        farneback = opticalflow3D.Farneback3D(iters=5,
                                    num_levels=5,
                                    scale=0.5,
                                    filter_size=filter_size,
                                    presmoothing=0, # Default, none
                                    filter_type="gaussian",
                                    sigma_k=0.05)
        
        output_vx, output_vy, output_vz, x = farneback.calculate_flow(
                0.05 * cp.abs(ref_image / cp.max(ref_image.ravel())),
                0.05 * cp.abs(mov_image / cp.max(mov_image.ravel())),
                total_vol=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                sub_volume=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                overlap=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                threadsperblock=(8, 8, 8),
            )
        deformation_fields = np.nan_to_num(np.concatenate([np.expand_dims(output_vx, 0), np.expand_dims(output_vy, 0), np.expand_dims(output_vz, 0)], axis=0))

        del output_vx
        del output_vy
        del output_vz
        del farneback
        del ref_image
        del mov_image
        gc.collect()
        return deformation_fields
    

def register_images(input_images, ref_index, filter_size=9, gpu_id=0):
    # Set the specific GPU device
    with cp.cuda.Device(gpu_id):
        # Convert images to CuPy arrays and initialize output arrays
        images = cp.asarray(input_images)
        output = cp.zeros(images.shape, dtype=cp.complex64)
        deformation_fields = np.zeros([images.shape[0], 3, images.shape[1], images.shape[2], images.shape[3]])
        nimages, nr, nc, nz = images.shape

        # Reference image
        ref_image = images[ref_index, ...].squeeze()

        cp.cuda.runtime.deviceSynchronize()
        farneback = opticalflow3D.Farneback3D(iters=5,
                                    num_levels=5,
                                    scale=0.5,
                                    filter_size=filter_size,
                                    presmoothing=2, # Default, none
                                    filter_type="gaussian",
                                    sigma_k=0.05)
        for ind in range(0, nimages):
            mov_image = images[ind, ...].squeeze()
            

            output_vx, output_vy, output_vz, x = farneback.calculate_flow(
                0.05 * cp.abs(ref_image / cp.max(ref_image.ravel())),
                0.05 * cp.abs(mov_image / cp.max(mov_image.ravel())),
                total_vol=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                sub_volume=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                overlap=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                threadsperblock=(8, 8, 8),
            )

            row_coords, col_coords, slice_coords = np.meshgrid(np.arange(nr), np.arange(nc), np.arange(nz),
                                                               indexing='ij')

            rcomp = cp.real(mov_image)
            icomp = cp.imag(mov_image)

            x = cp.array([row_coords + output_vx, col_coords + output_vy, slice_coords + output_vz])
            y = cp.array([row_coords + output_vx, col_coords + output_vy, slice_coords + output_vz])

            cp.cuda.runtime.deviceSynchronize()

            # Apply deformation to real and imaginary components
            output[ind, :, :, :] = map_coordinates(rcomp, x, mode="wrap") + 1j * map_coordinates(icomp, y, mode="wrap")
            
            cp.cuda.runtime.deviceSynchronize()

            deformation_fields[ind, ...] = np.concatenate([np.expand_dims(output_vx, 0), np.expand_dims(output_vy, 0), np.expand_dims(output_vz, 0)], axis=0)
            deformation_fields[ind, ...] = np.nan_to_num(deformation_fields[ind, ...])

        # display_registered_images(input_images[ref_index, ...].squeeze(),
        #                           np.mean(input_images, axis=0).squeeze(),
        #                           np.mean(output, axis=0).squeeze(),
        #                           input_images.shape[3] / 2)
        cp.cuda.runtime.deviceSynchronize()
        out_np = cp.asnumpy(output)

        del rcomp
        del icomp
        del output_vx
        del output_vy
        del output_vz
        del farneback
        del x
        del images
        del ref_image
        del output

        gc.collect()
        return out_np, deformation_fields
    
def apply_deformations(input_images, deformations,gpu_id):
    # Set the specific GPU device
    with cp.cuda.Device(gpu_id):
        # Convert images to CuPy arrays and initialize output arrays
        images = cp.asarray(input_images)
        output = cp.zeros(images.shape, dtype=cp.complex64)
        
        nimages, nr, nc, nz = images.shape

        row_coords, col_coords, slice_coords = np.meshgrid(np.arange(nr), np.arange(nc), np.arange(nz),
                                                    indexing='ij')
        for ind in range(0, nimages):
            mov_image = images[ind, ...].squeeze()
            

            rcomp = cp.real(mov_image)
            icomp = cp.imag(mov_image)

            output_vx = deformations[ind,0,...].squeeze()
            output_vy = deformations[ind,1,...].squeeze()
            output_vz = deformations[ind,2,...].squeeze()
            
            x = cp.array([row_coords + output_vx, col_coords + output_vy, slice_coords + output_vz])
            y = cp.array([row_coords + output_vx, col_coords + output_vy, slice_coords + output_vz])

            cp.cuda.runtime.deviceSynchronize()

            # Apply deformation to real and imaginary components
            output[ind, :, :, :] = map_coordinates(rcomp, x, mode="wrap") + 1j * map_coordinates(icomp, y, mode="wrap")
            
            cp.cuda.runtime.deviceSynchronize()

            
        cp.cuda.runtime.deviceSynchronize()
        out_np = cp.asnumpy(output)

        del rcomp
        del icomp
        del output_vx
        del output_vy
        del output_vz
        del images
        del output

        gc.collect()
        return out_np
    
    
def display_registered_images(ref_image,mov_image,warped_image,slice_pos=0):
    
    nr, nc, nz = ref_image.shape
    seq_im = np.zeros((nr, nc, nz,  3))
    seq_im[..., 0] = np.abs(cp.asnumpy(mov_image))
    seq_im[..., 1] = np.abs(cp.asnumpy(ref_image))
    seq_im[..., 2] = np.abs(cp.asnumpy(ref_image))

    # build an RGB image with the registered sequence
    reg_im = np.zeros((nr, nc, nz, 3))
    reg_im[..., 0] = np.abs(cp.asnumpy(warped_image))
    reg_im[..., 1] = np.abs(cp.asnumpy(ref_image))
    reg_im[..., 2] = np.abs(cp.asnumpy(ref_image))

    # build an RGB image with the registered sequence
    target_im = np.zeros((nr, nc, nz, 3))
    target_im[..., 0] = np.abs(cp.asnumpy(ref_image))
    target_im[..., 1] = np.abs(cp.asnumpy(ref_image))
    target_im[..., 2] = np.abs(cp.asnumpy(ref_image))

    # build an RGB image with the registered sequence
    m_im = np.zeros((nr, nc, nz, 3))
    m_im[..., 0] = np.abs(cp.asnumpy(mov_image))
    m_im[..., 1] = np.abs(cp.asnumpy(mov_image))
    m_im[..., 2] = np.abs(cp.asnumpy(mov_image))
    
    # build an RGB image with the registered sequence
    warped_im = np.zeros((nr, nc, nz, 3))
    warped_im[..., 0] = np.abs(cp.asnumpy(warped_image))
    warped_im[..., 1] = np.abs(cp.asnumpy(warped_image))
    warped_im[..., 2] = np.abs(cp.asnumpy(warped_image))

    # --- Show the result

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(10, 20))

    ax0.imshow(np.flipud(4*seq_im[:,:,slice_pos,:]/np.max(seq_im[:,:,slice_pos,:].ravel())))
    ax0.set_title("Unregistered sequence")
    ax0.set_axis_off()

    ax1.imshow(np.flipud(4*reg_im[:,:,slice_pos,:]/np.max(reg_im[:,:,slice_pos,:].ravel())))
    ax1.set_title("Registered sequence")
    ax1.set_axis_off()

    ax2.imshow(np.flipud(4*target_im[:,:,slice_pos,:]/np.max(target_im[:,:,slice_pos,:].ravel())))
    ax2.set_title("Target")
    ax2.set_axis_off()

    ax3.imshow(np.flipud(4*warped_im[:,:,slice_pos,:]/np.max(warped_im[:,:,slice_pos,:].ravel())))
    ax3.set_title("Warped")
    ax3.set_axis_off()
    
    ax4.imshow(np.flipud(4*m_im[:,:,slice_pos,:]/np.max(m_im[:,:,slice_pos,:].ravel())))
    ax4.set_title("Warped")
    ax4.set_axis_off()

    fig.tight_layout()
    
    
def findGPUs():
    numdevices =  torch.cuda.device_count()
    memcap = list() 
    for devno in range(numdevices):
        with torch.cuda.device(devno):
            f,t = torch.cuda.mem_get_info()        
        #memcap.append(float(torch.cuda.get_device_properties(devno).total_memory)/float(1024**3))
        memcap.append(float(f)/float(1024**3))
        print(f'Memory: {memcap}')
    
    return np.argsort(np.array(memcap))
    
if __name__ == "__main__":
    images = cp.random.rand(64,64*64*2)
    images = images.reshape((-1,64,64,64))
    register_images(images,0,display_flag=True)
    #frame1 = (cv2.imread('basketball1.png', cv2.IMREAD_GRAYSCALE))
    #frame2 = (cv2.imread('basketball2.png', cv2.IMREAD_GRAYSCALE))

    #nvof = cv2.cuda_NvidiaOpticalFlow_1_0.create(frame1.shape[1], frame1.shape[0], 5, False, False, False, 0)
    
    #flow = nvof.calc(frame1, frame2, None)

    #flowUpSampled = nvof.upSampler(flow[0], frame1.shape[1], frame1.shape[0], nvof.getGridSize(), None)

    #cv2.writeOpticalFlow('OpticalFlow.flo', flowUpSampled)
#
    #nvof.collectGarbage()