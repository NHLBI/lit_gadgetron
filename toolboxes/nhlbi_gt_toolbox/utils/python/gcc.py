import cupy as cp
import numpy as np
import cupyx.scipy.fft as cufft
import matplotlib.pyplot as plt
import torch
import gc
import copy
def fftc(x, dim=2):
    """
    Perform centered FFT along a specified dimension.

    Parameters:
        x (numpy.ndarray): Input array.
        dim (int): Dimension along which to perform the FFT (default is 2).

    Returns:
        numpy.ndarray: Result of the centered FFT.
    """
    return cp.asarray((1 / cp.sqrt(x.shape[dim])) * cp.fft.fftshift(cp.fft.fft(cp.fft.ifftshift(x, axes=dim), axis=dim), axes=dim))
def calc_gcc_mtx(calib_data, dim, ws=1):
    """
    Geometric Decomposition Coil Compression Based on:
    Zhang et. al MRM 2013;69(2):571-82.

    Computes and returns compression matrices for each position in space
    along a chosen dimension.

    Parameters:
        calib_data (numpy.ndarray): A 4D matrix representing [Kx, Ky, Kz, Coils] calibration data
                                    or a 3D matrix representing [Kx, Ky, Coils].
        dim (int): Dimension to perform compression on (1, 2, or 3).
        ws (int): Odd number of window size in space for computing the cc matrices (default is 1).

    Returns:
        numpy.ndarray: The compression matrix.
    """
    # Ensure ws is an odd number
    ws = (ws // 2) * 2 + 1

    # Check if k-space is 2D or 3D
    if calib_data.ndim == 3:
        ndims = 2
        calib_data = cp.expand_dims(calib_data, axis=2)  # Add a singleton dimension for Kz
    else:
        ndims = 3

    # Permute data if compression is done on 2nd or 3rd dimensions
    if dim == 1:
        calib_data = cp.transpose(calib_data, (1, 0, 2, 3))
    elif dim == 2:
        calib_data = cp.transpose(calib_data, (2, 0, 1, 3))
    elif dim < 0 or dim > 2:
        raise ValueError("Error, compression dimension must be 1, 2, or 3")

    

    calib_data = ifftc(calib_data, axis=0)

    Nx, Ny, Nz, Nc = calib_data.shape
    res = cp.zeros_like(calib_data)

    # Check if there's enough data for subspace calculation
    if ws * Ny * Nz < 3 * Nc:
        print("Warning: ratio between data in each slice and number of channels is less than 3 -- "
              "noise could bias results. You should increase ws")

    # Calculate compression matrices for each position in the readout
    # over a sliding window of size ws
    mtx = cp.zeros((Nc, min(Nc, ws * Ny * Nz), Nx), dtype=cp.complex64)

    # Zero-padding function
    # def zpad(data, shape):
    #     pad_width = [(0, max(0, s - d)) for d, s in zip(data.shape, shape)]
    #     return cp.pad(data, pad_width, mode='constant')

    zpim = zpad(calib_data, (Nx + ws - 1, Ny, Nz, Nc))

    for n in range(Nx):
        tmpc = zpim[n:n + ws, :, :, :].reshape(ws * Ny * Nz, Nc)
        U, S, Vh = cp.linalg.svd(tmpc, full_matrices=False)
        mtx[:, :, n] = Vh.T.conj()

    return mtx
def zpad(x, sx, sy=None, sz=None, st=None):
    """
    Zero pads a matrix around its center.

    Parameters:
        x (numpy.ndarray): Input matrix to pad.
        sx (int or list): Target size for the first dimension, or a list of sizes for all dimensions.
        sy (int, optional): Target size for the second dimension.
        sz (int, optional): Target size for the third dimension.
        st (int, optional): Target size for the fourth dimension.

    Returns:
        numpy.ndarray: Zero-padded matrix.
    """
    if isinstance(sx, (list, tuple)):
        s = sx
    else:
        s = [sx]
        if sy is not None:
            s.append(sy)
        if sz is not None:
            s.append(sz)
        if st is not None:
            s.append(st)

    m = x.shape
    if len(m) < len(s):
        m = list(m) + [1] * (len(s) - len(m))

    if all(mi == si for mi, si in zip(m, s)):
        return x
    res = cp.zeros(s, dtype=x.dtype)

    idx = []
    for n in range(len(s)):
        start = s[n] // 2 - m[n] // 2
        end = start + m[n]
        idx.append(slice(start, end))

    res[tuple(idx)] = x
    return res
def align_cc_mtx(in_mtx, ncc=None):
    """
    Align coil compression matrices based on nearest spanning vectors in subspaces.
    This is an implementation based on Zhang et. al MRM 2013;69(2):571-82.

    Parameters:
        mtx (numpy.ndarray): Icput compression matrices of shape [nc, nc, slices].
        ncc (int, optional): Number of compressed virtual coils to align. Defaults to all.

    Returns:
        numpy.ndarray: Aligned compression matrices.
    """
    import copy

    mtx = copy.deepcopy(in_mtx)
    sx, sy, nc = mtx.shape

    if ncc is None:
        ncc = sy

    if sx == sy and ncc == sx:
        print("Warning: number of aligned coils is the same as physical coils, this will do nothing. "
                "Either crop mtx or align fewer coils.")

    # Align everything based on the middle slice
    n0 = int(nc / 2)-1
    A00 = copy.deepcopy(mtx[:, 0:ncc, n0])

    # Align backwards to the first slice
    A0 = copy.deepcopy(A00)
    for n in range(n0 - 1, -1, -1):
        A1 = copy.deepcopy(mtx[:, 0:ncc, n])
        C = cp.matmul(A1.T.conj(), A0)
        U, S, Vh = cp.linalg.svd(C, full_matrices=False)
        P = cp.dot(Vh.T.conj(), U.T.conj())
        mtx[:, 0:ncc, n] = cp.matmul(A1, P.T.conj())
        A0 = copy.deepcopy(mtx[:, 0:ncc, n])

    # Align forward to the last slice
    A0 = copy.deepcopy(A00)
    for n in range(n0 + 1, nc):
        A1 = copy.deepcopy(mtx[:, 0:ncc, n])
        C = cp.matmul(A1.T.conj(), A0)
        U, S, Vh = cp.linalg.svd(C, full_matrices=False)
        P = cp.matmul(Vh.T.conj(), U.T.conj())
        mtx[:, 0:ncc, n] = cp.matmul(A1, P.T.conj())
        A0 = copy.deepcopy(mtx[:, 0:ncc, n])

    return mtx

# Perform IFFT along the first dimension
def ifftc(data, axis):
    shape = cp.array(data.shape)
    return cp.asarray(cp.sqrt(cp.prod(shape[axis]))*cufft.ifftshift(cufft.ifftn(cufft.fftshift(data, axes=axis), axes=axis), axes=axis))

# Perform FFT along the first dimension
def fftc(data, axis):
    shape = cp.array(data.shape)
    return (1/cp.asarray(cp.sqrt(cp.prod(shape[axis])))) * cp.asarray(cufft.ifftshift(cufft.fftn(cufft.fftshift(data, axes=axis), axes=axis), axes=axis))

def CC(DATA, mtx, dim=None,nofft=False):
    """
    Geometric Decomposition Coil Compression Based on:
    Zhang et. al MRM 2013;69(2):571-82.

    The function uses compression matrices computed by calc_gcc_mtx or align_cc_mtx
    to project the data onto them.

    Parameters:
        DATA (numpy.ndarray): A 4D matrix representing [Kx, Ky, Kz, Coils] data
                                or a 3D matrix representing [Kx, Ky, Coils].
        mtx (numpy.ndarray): Aligned compression matrices from calc_gcc_mtx or align_cc_mtx.
        dim (int, optional): Dimension to perform compression on. If missing, the function
                                assumes SCC-type compression matrix.

    Returns:
        numpy.ndarray: Data rotated to the compressed space. The first ncc coils are the
                        compressed & aligned virtual coils.
    """
    
    # Check if 2D or 3D data
    if DATA.ndim == 3:
        ndims = 2
        DATA = cp.expand_dims(DATA, axis=2)  # Add a singleton dimension for Kz
    else:
        ndims = 3

    ncc = mtx.shape[1]

    # Assume SCC if no dimension is used
    if dim is None:
        if mtx.ndim > 2:
            raise ValueError("Compression matrix is probably GCC, but no dimension was entered")
        
        Nx, Ny, Nz, Nc = DATA.shape
        DATA = DATA.reshape((Nx * Ny * Nz, Nc))
        res = cp.matmul(DATA, mtx)
        res = res.reshape((Nx, Ny, Nz, ncc))
        # Squeeze back to two dimensions if necessary
        if ndims == 2:
            res = cp.squeeze(res)
        return res

    # GCC-based compressions
    else:
        # Permute data if compression is done on 2nd or 3rd dimension
        if dim == 1:
            DATA = cp.transpose(DATA, (1, 0, 2, 3))
        elif dim == 2:
            DATA = cp.transpose(DATA, (2, 0, 1, 3))
        elif dim > 2 or dim < 0:
            raise ValueError("Error, compression dimension must be 1, 2, or 3")

        Nx, Ny, Nz, Nc = DATA.shape
        #print(DATA.shape)
        
        im = ifftc(DATA, axis=0) if not nofft else DATA
        res = cp.zeros((Nx, Ny, Nz, ncc), dtype=cp.complex64)
       # print(im.shape)
        # print("res:",res.shape)
        # print("mtx:",mtx.shape)
        # Rotate data by compression matrices
        for n in range(Nx):
            tmpc = im[n, :, :, :].reshape((Ny * Nz, Nc))
            # print("tmpc:",tmpc.shape)
            # print(mtx[:, :, n].shape)
            # print(cp.sum(mtx[:, :, n]**2))
            # print(cp.sum(tmpc**2))

            res[n, :, :, :] = cp.matmul(tmpc, mtx[:, :, n]).reshape((Ny, Nz, ncc))

        res = fftc(res, axis=0) if not nofft else res
        #print(res.shape)

        # Permute back if necessary
        if dim == 1:
            res = cp.transpose(res, (1, 0, 2, 3))
        elif dim == 2:
            res = cp.transpose(res, (1, 2, 0, 3))
       # print(res.shape)

        # Squeeze back to two dimensions if necessary
        if ndims == 2:
            res = cp.squeeze(res)

        return res
def fft2c(x):
    """
    Perform centered 2D FFT on the input array.

    Parameters:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Result of the centered 2D FFT.
    """
    S = x.shape
    fctr = S[0] * S[1]

    x = x.reshape(S[0], S[1], cp.prod(S[2:]))

    res = cp.zeros_like(x, dtype=cp.complex64)
    for n in range(x.shape[2]):
        res[:, :, n] = (1 / cp.sqrt(fctr)) * cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(x[:, :, n])))

    res = res.reshape(S)
    return res
def ifft2c(x):
    """
    Perform centered 2D IFFT on the input array.

    Parameters:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Result of the centered 2D IFFT.
    """
    S = x.shape
    fctr = S[0] * S[1]

    x = x.reshape(S[0], S[1], cp.prod(S[2:]))

    res = cp.zeros_like(x, dtype=cp.complex64)
    for n in range(x.shape[2]):
        res[:, :, n] = cp.sqrt(fctr) * cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(x[:, :, n])))

    res = res.reshape(S)
    return res
def calc_scc_mtx(calib_data):
    """
    Coil Compression using a single compression matrix. Based on Huang et al.,
    MRM 2008;26:133-141.

    Computes and returns a single compression matrix.

    Parameters:
        calib_data (numpy.ndarray): A 4D matrix representing [Kx, Ky, Kz, Coils] calibration data
                                    or a 3D matrix representing [Kx, Ky, Coils].

    Returns:
        numpy.ndarray: The compression matrix.
    """
    # Check if k-space is 2D or 3D
    if calib_data.ndim == 3:
        ndims = 2
        calib_data = cp.expand_dims(calib_data, axis=2)  # Add a singleton dimension for Kz
    else:
        ndims = 3

    Nx, Ny, Nz, Nc = calib_data.shape
    calib_data = calib_data.reshape((Nx * Ny * Nz, Nc))

    # Perform SVD
    U, S, Vh = cp.linalg.svd(calib_data, full_matrices=False)

    # Return the appropriate matrix
    if calib_data.shape[0] > calib_data.shape[1]:
        mtx = Vh.T.conj()
    else:
        mtx = U

    return mtx
def crop(x, sx, sy=None, sz=None, st=None):
    """
    Crop a matrix around its center.

    Parameters:
        x (numpy.ndarray): Input matrix to crop.
        sx (int or list): Target size for the first dimension, or a list of sizes for all dimensions.
        sy (int, optional): Target size for the second dimension.
        sz (int, optional): Target size for the third dimension.
        st (int, optional): Target size for the fourth dimension.

    Returns:
        numpy.ndarray: Cropped matrix.
    """

    if isinstance(sx, (list, tuple)):
        s = sx
    else:
        s = [sx]
        if sy is not None:
            s.append(sy)
        if sz is not None:
            s.append(sz)
        if st is not None:
            s.append(st)

    m = x.shape
    if len(s) < len(m):
        s.extend([1] * (len(m) - len(s)))

    if all(mi == si for mi, si in zip(m, s)):
        return x

    idx = []
    for n in range(len(s)):
        start = m[n] / 2 - s[n] / 2
        end = start + s[n]
        idx.append(slice(start, end))

    return x[tuple(idx)]

def imshow3(data, clim, dispm):
    """Display 3D data as a grid of 2D images."""
    rows, cols = dispm
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if(rows>1):
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < data.shape[2]:
                    axes[i, j].imshow(data[:, :, idx], clim=clim, cmap='gray')
                    axes[i, j].axis('off')
    else:
        for j in range(cols):
            idx =j
            if idx < data.shape[2]:
                axes[j].imshow(data[:, :, idx], clim=clim, cmap='gray')
                axes[j].axis('off')
    plt.show()
    
def sos(data):
    """Sum of squares across the last axis."""
    return np.sqrt(np.sum(np.abs(data)**2, axis=-1))

def calc_ecc_mtx(data_in, dim, ncc=None, ks=8):
    """
    ESPIRiT-based coil compression.
    Based on D. Bahri et al., "ESPIRiT-Based Coil Compression for Cartesian Sampling" ISMRM 2013 pp:2657.

    Computes and returns compression matrices for each position in space along a chosen dimension.
    ECC is similar to GCC but is slower and less biased by noise.

    Parameters:
        data (numpy.ndarray): A 4D matrix representing [Kx, Ky, Kz, Coils] data
                                or a 3D matrix representing [Kx, Ky, Coils].
        dim (int): Dimension to perform compression on.
        ncc (int, optional): Approximate number of target virtual coils. If missing, it is estimated based on singular values.
        ks (int, optional): ESPIRiT kernel size (default is 8).

    Returns:
        numpy.ndarray: Compression matrix.
    """
    import copy
    data = copy.deepcopy(data_in)  # Avoid modifying the original data
    if ncc is None:
        ncc = []

    # Check if data is 2D or 3D
    if data.ndim == 3:
        ndims = 2
        data = cp.expand_dims(data, axis=2)  # Add a singleton dimension for Kz
    else:
        ndims = 3

    # Permute data if compression is done on 2nd or 3rd dimension
    if dim == 1:
        data = cp.transpose(data, (1, 0, 2, 3))
    elif dim == 2:
        data = cp.transpose(data, (2, 0, 1, 3))
    elif dim > 2 or dim < 0:
        raise ValueError("Error, compression dimension must be 1, 2, or 3")

    # Perform IFFT
    Nx, Ny, Nz, Nc = data.shape
    data = data.reshape((Nx, Ny * Nz, Nc))

    # Construct an ESPIRiT kernel
    k, S = dat2kernel(data, [ks, 1])

    if not ncc:
        idx = cp.min(cp.where(S < S[0] * 0.02)[0])
    else:
        idx = cp.min(cp.array([np.ceil(len(S) * ncc * 1.2 / Nc), len(S)]).astype(int))

    # Crop kernels
    k = k[:, :, :, :idx]

    # Compute compression matrices by eigen decomposition in image space
    mtx, W = kernel_eig(k, [Nx, 1])

    # Compression matrices must be conjugated for compatibility with ESPIRiT SENSE code
    mtx = cp.conj(cp.transpose(mtx, (2, 3, 0, 1)))

    # Set most important coils to be first
    mtx = mtx[:, ::-1, :]

    return mtx
def dat2kernel(data, kSize):
    """
    Perform k-space calibration step for ESPIRiT and create k-space kernels.

    Parameters:
        data (numpy.ndarray): Calibration data [kx, ky, coils].
        kSize (list or tuple): Size of the kernel (e.g., kSize=[6, 6]).

    Returns:
        tuple: 
            - kernel (numpy.ndarray): K-space kernels matrix (not cropped), which correspond to
                                        the basis vectors of overlapping blocks in k-space.
            - S (numpy.ndarray): Singular values of the calibration matrix.
    """
    sx, sy, nc = data.shape
    imSize = [sx, sy]

    tmp = im2row(data, kSize)
    tsx, tsy, tsz = tmp.shape
    A = tmp.reshape(tsx, tsy * tsz)
  #  print(A.shape)
    U, S, Vh = cp.linalg.svd(A, full_matrices=False)

    kernel = Vh.T.conj().reshape(kSize[0], kSize[1], nc, Vh.shape[0])
   # S = cp.diag(S).flatten()

    return kernel, S

def im2row(im, winSize):
    """
    Convert an image into overlapping rows based on a given window size.

    Parameters:
        im (numpy.ndarray): Input image of shape (sx, sy, sz).
        winSize (list or tuple): Window size [winSize_x, winSize_y].

    Returns:
        numpy.ndarray: Output matrix of shape ((sx-winSize[0]+1)*(sy-winSize[1]+1), prod(winSize), sz).
    """
    sx, sy, sz = im.shape
    res = cp.zeros(((sx - winSize[0] + 1) * (sy - winSize[1] + 1), int(cp.prod(cp.array(winSize))), sz), dtype=im.dtype)
    count = 0
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count += 1
            res[:, [count - 1], :] = im[x:sx - winSize[0] + x + 1, y:sy - winSize[1] + y + 1, :].reshape(
                (sx - winSize[0] + 1) * (sy - winSize[1] + 1), 1, sz
            )
    return res

def kernel_eig(kernel, im_size):
    """
    Perform ESPIRiT step II -- eigen-value decomposition of a k-space kernel in image space.

    Parameters:
        kernel (numpy.ndarray): K-space kernels computed with dat2kernel (4D).
        im_size (list or tuple): The size of the image to compute maps for [sx, sy].

    Returns:
        tuple:
            - EigenVecs (numpy.ndarray): Images representing the Eigenvectors. (sx, sy, num_coils, num_coils)
            - EigenVals (numpy.ndarray): Images representing the EigenValues. (sx, sy, num_coils)
                                            The last are the largest (close to 1).
    """
    nc = kernel.shape[2]
    nv = kernel.shape[3]
    k_size = [kernel.shape[0], kernel.shape[1]]

    # Rotate kernel to order by maximum variance
    k = cp.transpose(kernel, (0, 1, 3, 2)).reshape(-1, nc)

    if k.shape[0] < k.shape[1]:
        u, s, v = cp.linalg.svd(k, full_matrices=False)
    else:
        u, s, v = cp.linalg.svd(k, full_matrices=False)

    k = k @ v.T.conj()
    kernel = k.reshape(k_size[0], k_size[1], nv, nc)
    kernel = cp.transpose(kernel, (0, 1, 3, 2))

    KERNEL = cp.zeros((im_size[0], im_size[1], kernel.shape[2], kernel.shape[3]), dtype=cp.complex64)
    for n in range(kernel.shape[3]):
        KERNEL[:, :, :, n] = fftc(
            zpad(cp.conj(kernel[::-1, ::-1, :, n]) * np.sqrt(im_size[0] * im_size[1]),
                    [im_size[0], im_size[1], kernel.shape[2]]),axis=[0,1]
        )
    KERNEL /= np.sqrt(np.prod(k_size))

    EigenVecs = cp.zeros((im_size[0], im_size[1], nc, min(nc, nv)), dtype=cp.complex64)
    EigenVals = cp.zeros((im_size[0], im_size[1], min(nc, nv)), dtype=cp.float32)

    for x in range(im_size[0]):
        for y in range(im_size[1]):
            mtx = KERNEL[x, y, :, :]

            # Perform SVD
            u, s, vh = cp.linalg.svd(mtx, full_matrices=False)

            # Adjust phase
            ph = cp.exp(-1j * cp.angle(u[0, :]))
            u = cp.dot(v.T.conj(), u * ph)

            EigenVals[x, y, :] = s[::-1]
            EigenVecs[x, y, :, :] = u[:, ::-1]

    return EigenVecs, EigenVals

def gcc_compress_all(image,calibX,ncc,cctype='GCC'):
    
    in_image = copy.deepcopy(image)
    CHA=np.shape(in_image)[-1]
    compressed_data_a,mtxra =gcc_compress(in_image,calibX,0,(CHA+ncc)//2)

    CHA_r=np.shape(compressed_data_a)[-1]
    compressed_data_pa,mtxpa =gcc_compress(compressed_data_a,calibX,1,(CHA_r+ncc)//2)
    
    compressed_data_za,mtxza =gcc_compress(compressed_data_pa,calibX,2,ncc)

    mtx_all=[mtxra,mtxpa,mtxza]

    return compressed_data_za,mtx_all


def gcc_compress(image,calibX,dim,ncc,cctype='GCC'):
    import copy
    in_image = copy.deepcopy(image)

    print(f'GCC_image_shape: {image.shape}')
    
    if(image.shape[-1]<ncc):
        ncc = image.shape[-1]
        print(f'Warning: Number of coils in the image is less than ncc, setting ncc to {ncc}')
    GPU_freeM,dev_num=get_GPU_most_free()
    print(f"Free GPU {GPU_freeM}")
    if dev_num!=-1:
        with cp.cuda.Device(dev_num):
            DATA = cp.asarray(ifftc(in_image,axis=[0,1,2]))

            calibSize = np.array(DATA.shape)
            if(dim==1):
                calibSize[dim-1] = calibX
                calibSize[dim+1] = calibX
            if(dim==0):
                calibSize[dim+1] = calibX
                calibSize[dim+2] = calibX
            
            if(dim==2):
                calibSize[dim-1] = calibX
                calibSize[dim-2] = calibX

            print(calibSize)
            calib = cp.asarray(crop(DATA, list(calibSize)))
            
            if cctype == 'GCC':
                mtx = calc_gcc_mtx(calib,dim,1)
                aligned_mtx = align_cc_mtx(mtx[:,:ncc,:])
            elif cctype == 'ECC':
                mtx = calc_ecc_mtx(calib,dim,1).squeeze()
                aligned_mtx = align_cc_mtx(mtx[:,:ncc,:])
            else:
                mtx = calc_scc_mtx(calib)    
            
            print(mtx.shape)

            clear_cuda_cache()    
            if cctype=='SCC':
                compressed_data         = fftc(CC(DATA,mtx),axis=[0,1,2])
                clear_cuda_cache()    

                return cp.asnumpy(compressed_data),cp.asnumpy(mtx)
            else:
                print("DATA:", DATA.shape)

                compressed_data         = fftc(CC(DATA,mtx,dim),axis=[0,1,2])
                compressed_data_aligned = fftc(CC(DATA,aligned_mtx,dim),axis=[0,1,2])
                clear_cuda_cache()
                print("completed GCC")
                return cp.asnumpy(compressed_data_aligned).astype(np.complex64), cp.asnumpy(aligned_mtx).astype(np.complex64)

import torch 
import cupy as cp
import gc

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

def apply_gcc_compress(in_image,mtx,dim):
    GPU_freeM,dev_num=get_GPU_most_free()
    print(f"Free GPU {GPU_freeM}")
    if dev_num!=-1:
        with cp.cuda.Device(dev_num):
            DATA = cp.asarray(ifftc(in_image,axis=[0,1,2])).astype(np.complex64)
            print("norm DATA:", cp.linalg.norm(DATA.ravel()))
            
            mtx = cp.asarray(mtx).astype(np.complex64)

            clear_cuda_cache()    
            print("Start compression")

            compressed_data         = cp.asnumpy(fftc(CC(DATA,mtx,dim),axis=[0,1,2]))
            print("norm DATA:", cp.linalg.norm(compressed_data.ravel()))
            clear_cuda_cache()    
            print("completed GCC")
    return compressed_data.astype(np.complex64)
 
def clear_cuda_cache():
    numdevices =  torch.cuda.device_count()
    for devno in range(numdevices):
        with torch.cuda.device(devno):
            gc.collect()
            
            f,t = torch.cuda.mem_get_info() 
            # print(f"GPU : {devno} Total Memory : {t/(1024**2)} MB Free Memory : {f/(1024**2)} MB")  
            # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            # print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2} MB")
            torch.cuda.empty_cache()

        with cp.cuda.Device(devno):
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            # print(f"GPU : {devno} Total Memory : {mempool.total_bytes()/(1024**2)} MB Used Memory : {mempool.used_bytes()/(1024**2)} MB")


            mempool.free_all_blocks()
            
            # Optionally clear any pinned memory (CPU-GPU pinned transfers)
            pinned_mempool.free_all_blocks()
            
            # Clear variables stored on GPU
            cp._default_memory_pool.free_all_blocks()
 
            # Avoiding Device Synchronization for Faster Execution
            # By default, many CuPy operations perform implicit device synchronization (waiting for all
            # previous operations to complete before continuing). You can explicitly manage synchronization
            # to improve performance by deferring synchronization calls:
            cp.cuda.stream.get_current_stream().synchronize()  # Manually synchronize
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            # print(f"GPU : {devno} Total Memory : {mempool.total_bytes()/(1024**2)} MB Used Memory : {mempool.used_bytes()/(1024**2)} MB")
 
