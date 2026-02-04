import numpy as np 
import cupy as cp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sigpy import thresh
import torch 
import cupy as cp
import time
import copy
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

def mult_MH(input,dim,lambda_a,alpha_a):
    print("MULTI_MH")
    return LR(input,dim,lambda_a,alpha_a)

def prox_LLR(input,dim,alpha_a,block=32):
    t0 = time.time()  
    GPU_freeM,dev_num=get_GPU_most_free()
    print(f"Free Memory GPU {GPU_freeM} GPU num {dev_num}")
    print(f"Input parameters {input.shape} dim {dim} {alpha_a} {block}")
    if (alpha_a>=1):
        print(f"Careful the threshold is too high {alpha_a}")
    i_shape=input.shape
    nele=np.prod(i_shape)#Data size
    block_shape=np.array(([block]*(len(i_shape)-1)+[i_shape[-1]]))
    bshape=tuple(block_shape)
    #bshape=tuple([block]*(len(i_shape)-1)+[i_shape[-1]])
    
    random_shift=np.concatenate([np.random.rand((len(i_shape)-1)),np.ones(1)])
    
    rand_step_shape=(block_shape*random_shift).astype(int)[:,None]
    #Ensuring at least shift by 1
    min_step=np.array(([block//2]*(len(i_shape)-1)+[i_shape[-1]]))
    step_shape=tuple(np.max(np.concatenate([rand_step_shape,min_step[:,None]],axis=-1),axis=-1))
    print(step_shape)
    d_tmp = view_as_windows(input, bshape,step=step_shape)
    #d_tmp=view_as_blocks(input,bshape)
    d_input=d_tmp.reshape([-1,np.prod(bshape[:-1]),i_shape[-1]])
    
    
    t1 = time.time()  
    data_o=np.zeros(i_shape,dtype=np.complex64)
    d_index=view_as_windows(np.arange(nele).reshape(i_shape), bshape,step=step_shape).reshape([-1,np.prod(bshape[:-1]),i_shape[-1]])   
    #d_index=view_as_blocks(np.arange(nele).reshape(i_shape), bshape).reshape([-1,np.prod(bshape[:-1]),i_shape[-1]])   
    
    print(d_index.shape)
    f_index=1/np.unique(d_index, return_counts= True)[1]
    print(f_index.shape)
    svd_max_number=1e8/np.prod(bshape)
    t2 = time.time()  
    #d_tmp=view_as_blocks(input,bshape)
    #tr=[]
    #for n in range(len(i_shape)):
    #    tr.extend([n,n+len(i_shape)])
    if dev_num!=-1:
        with cp.cuda.Device(dev_num):
            N=int(np.prod(d_input.shape)/np.prod(bshape))
            batchsize=int(np.ceil((svd_max_number)))
            print(batchsize)
            batch=np.arange(0,N-1,min(N,batchsize)).tolist()
            batch.append(N)
            batch=np.array(batch).astype(np.uint)
            
            for b_idx in range(len(batch)-1):
                t3 = time.time()  
                batch_start=batch[b_idx]
                batch_end=batch[b_idx+1]

                batch_input = cp.asarray(d_input[batch_start:batch_end,:])
                t3 = time.time()  
                # Perform SVD on the batch
                u, s, vh = cp.linalg.svd(batch_input, full_matrices=False)
                t4 = time.time()  
                #if s_max is None:
                s_max = cp.max(s,axis=-1) 
                print(s_max.shape)
                t5 = time.time()  
                # Soft thresholding
                s_thresh = thresh.soft_thresh(alpha_a * s_max[:,None], s)
                print(f"Singular-values after thresholding ( {alpha_a} {s_max}): {s_thresh}")
                # Reconstruct the batch after thresholding
                batch_output = cp.matmul(u, s_thresh[..., None] * vh)
                t6 = time.time() 
                # Assign the processed batch back to the corresponding columns in the output
                index_batch=d_index[batch_start:batch_end,:].flatten()
                data_o[np.unravel_index(index_batch,i_shape)] += batch_output.get().flatten()*f_index[index_batch]
                del batch_input
                del u
                del s
                del vh
                del batch_output
                gc.collect()
                t7 = time.time() 
            #data_o=data_o.reshape(d_tmp.shape).transpose(tr).reshape(i_shape)
    t8 = time.time()
    print(f"Time A {t8-t0} ms  B {t8-t7} ms C {t8-t7} ms D {t7-t6} ms E {t6-t5} ms F {t5-t4} ms G {t4-t3} ms H {t3-t2} ms J {t2-t1} ms K {t1-t0} ms ")  
    return data_o

def prox_LR(input,dim,lambda_a,alpha_a):
    t0 = time.time()  
    GPU_freeM,dev_num=get_GPU_most_free()
    print(f"Free Memory GPU {GPU_freeM} GPU num {dev_num}")
    print(f"Input parameters {input.shape} dim {dim} {lambda_a} {alpha_a}")
    if (alpha_a*lambda_a>=1):
        print(f"Careful the threshold is too high {alpha_a} {lambda_a}")
    i_shape=input.shape
    d_input=input.reshape([-1,i_shape[-1]])
    #Data size
    nele=np.prod(i_shape)
    t1 = time.time()  
    data_o=np.zeros(d_input.shape,dtype=np.complex64)
    lambda_a=1
    cmplex_bytes=np.zeros((1,1),dtype=np.complex64).nbytes# 8
    data_o_bytes=cmplex_bytes*nele
    print(f"data_size {nele} {data_o_bytes}")
    svd_max_number=1e8//input.shape[-1]
    t2 = time.time()  
    if dev_num!=-1:
        with cp.cuda.Device(dev_num):
            ratio_d_GPU=(2*data_o_bytes)/(GPU_freeM*0.5)
            if nele < svd_max_number:
                batch=np.array([0,int(nele/input.shape[-1])]).astype(np.uint)
                # do LR
            else:
                N=int(nele/input.shape[-1])
                batchsize=int(np.ceil((svd_max_number)))
                print(batchsize)
                batch=np.arange(0,N-1,min(N,batchsize)).tolist()
                batch.append(N)
                batch=np.array(batch).astype(np.uint)
            #s_max=None
            
            for b_idx in range(len(batch)-1):
                t3 = time.time()  
                batch_start=batch[b_idx]
                batch_end=batch[b_idx+1]

                batch_input = cp.asarray(d_input[batch_start:batch_end,:])
                # Perform SVD on the batch
                u, s, vh = cp.linalg.svd(batch_input, full_matrices=False)
                print(f"Singular-values before thresholding (batch {batch_start}-{batch_end}): {s}")
                t4 = time.time()  
                #if s_max is None:
                s_max = cp.max(s) 
                t5 = time.time()  
                # Soft thresholding
                s_thresh = thresh.soft_thresh(lambda_a* alpha_a * s_max, s)
                print(f"Singular-values after thresholding ({lambda_a} {alpha_a} {s_max}): {s_thresh}")
                # Reconstruct the batch after thresholding
                batch_output = cp.matmul(u, s_thresh[..., None] * vh)
                t6 = time.time() 
                # Assign the processed batch back to the corresponding columns in the output
                data_o[batch_start:batch_end,:] = batch_output.get()
                del batch_input
                del u
                del s
                del vh
                del batch_output
                gc.collect()
                t7 = time.time() 
            data_o=data_o.reshape(i_shape)
    t8 = time.time()
    print(f"Time A {t8-t0} ms  B {t8-t7} ms C {t8-t7} ms D {t7-t6} ms E {t6-t5} ms F {t5-t4} ms G {t4-t3} ms H {t3-t2} ms J {t2-t1} ms K {t1-t0} ms ")  
    return data_o


def LR(input,dim,lambda_a,alpha_a):
    t0 = time.time()  
    GPU_freeM,dev_num=get_GPU_most_free()
    print(f"Free Memory GPU {GPU_freeM} GPU num {dev_num}")
    print(f"Input parameters {input.shape} dim {dim} {lambda_a} {alpha_a}")
    if (alpha_a*lambda_a>=1):
        print(f"Careful the threshold is too high {alpha_a} {lambda_a}")
    i_shape=input.shape
    d_input=input.reshape([-1,i_shape[-1]])
    #Data size
    nele=np.prod(i_shape)
    t1 = time.time()  
    data_o=np.zeros(d_input.shape,dtype=np.complex64)
    lambda_a=1
    cmplex_bytes=np.zeros((1,1),dtype=np.complex64).nbytes# 8
    data_o_bytes=cmplex_bytes*nele
    print(f"data_size {nele} {data_o_bytes}")
    svd_max_number=10e6
    t2 = time.time()  
    if dev_num!=-1:
        with cp.cuda.Device(dev_num):
            ratio_d_GPU=(2*data_o_bytes)/(GPU_freeM*0.5)
            if nele < svd_max_number:
                batch=np.array([0,int(nele/input.shape[-1])]).astype(np.uint)
                # do LR
            else:
                N=int(nele/input.shape[-1])
                batchsize=int(np.ceil((svd_max_number)))
                print(batchsize)
                batch=np.arange(0,N-1,min(N,batchsize)).tolist()
                batch.append(N)
                batch=np.array(batch).astype(np.uint)
            #s_max=None
            
            for b_idx in range(len(batch)-1):
                t3 = time.time()  
                batch_start=batch[b_idx]
                batch_end=batch[b_idx+1]

                batch_input = cp.asarray(d_input[batch_start:batch_end,:])
                # Perform SVD on the batch
                u, s, vh = cp.linalg.svd(batch_input, full_matrices=False)
                print(f"Singular-values before thresholding (batch {batch_start}-{batch_end}): {s}")
                t4 = time.time()  
                #if s_max is None:
                s_max = cp.max(s) 
                t5 = time.time()  
                # Soft thresholding
                s_thresh = thresh.soft_thresh(lambda_a* alpha_a * s_max, s)
                print(f"Singular-values after thresholding ({lambda_a} {alpha_a} {s_max}): {s_thresh}")
                #s_thresh=(s_max/s_thresh[0])*s_thresh
                print(f"Singular-values after thresholding LR({lambda_a} {alpha_a} {s_max}): {s_thresh}")
                # Reconstruct the batch after thresholding
                det=1/np.prod(s)
                batch_output = cp.matmul(u, det*s_thresh[..., None] * vh)
                t6 = time.time() 
                # Assign the processed batch back to the corresponding columns in the output
                data_o[batch_start:batch_end,:] = batch_output.get()
                t7 = time.time() 
            data_o=data_o.reshape(i_shape)
    t8 = time.time()
    print(f"Time A {t8-t0} ms  B {t8-t7} ms C {t8-t7} ms D {t7-t6} ms E {t6-t5} ms F {t5-t4} ms G {t4-t3} ms H {t3-t2} ms J {t2-t1} ms K {t1-t0} ms ")  
    return data_o

def gradient(input,dim,lambda_a,alpha_a):

    GPU_freeM,dev_num=get_GPU_most_free()
    print(f"Free Memory GPU {GPU_freeM} GPU num {dev_num}")
    print(f"Input parameters {input.shape} dim {dim} {lambda_a} {alpha_a}")
    i_shape=input.shape
    d_input=input.reshape([-1,i_shape[-1]])
    #Data size
    nele=np.prod(i_shape)
    cmplex_bytes=np.zeros((1,1),dtype=np.complex64).nbytes# 8
    data_o_bytes=cmplex_bytes*nele
    print(f"data_size {nele} {data_o_bytes}")
    svd_max_number=10e6
    g=np.zeros((1,1),dtype=np.complex64)
    if dev_num!=-1:
        with cp.cuda.Device(dev_num):
            if nele < svd_max_number:
                batch=np.array([0,int(nele/input.shape[-1])]).astype(np.uint)
                # do LR
            else:
                N=int(nele/input.shape[-1])
                batchsize=int(np.ceil((svd_max_number/input.shape[-1])))
                print(batchsize)
                batch=np.arange(0,N-1,min(N,batchsize)).tolist()
                batch.append(N)
                batch=np.array(batch).astype(np.uint)
            for b_idx in range(len(batch)-1):
                batch_start=batch[b_idx]
                batch_end=batch[b_idx+1]

                batch_input = cp.asarray(d_input[batch_start:batch_end,:])
                # Perform SVD on the batch
                u, s, vh = cp.linalg.svd(batch_input, full_matrices=False)
                
                g+=lambda_a * cp.sum(cp.abs(s)).item()
                print(f"Gradients {g} batch {b_idx}")
                
    return g


def mult_M(input,dim,lambda_a,alpha_a):
    print("MULTI_M")
    return LR(input,dim,lambda_a,alpha_a)


def mult_MH_M(input,dim,lambda_a,alpha_a):
    print("MULTI_MH_M")
    return LR(input,dim,lambda_a,alpha_a)

# Extract from skimage 
import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided

#__all__ = ['view_as_blocks', 'view_as_windows']


def view_as_blocks(arr_in, block_shape):
    """Block view of the input n-dimensional array (using re-striding).

    Blocks are non-overlapping views of the input array.

    Parameters
    ----------
    arr_in : ndarray, shape (M[, ...])
        Input array.
    block_shape : tuple
        The shape of the block. Each dimension must divide evenly into the
        corresponding dimensions of `arr_in`.

    Returns
    -------
    arr_out : ndarray
        Block view of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_blocks
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> B = view_as_blocks(A, block_shape=(2, 2))
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[2, 3],
           [6, 7]])
    >>> B[1, 0, 1, 1]
    13

    >>> A = np.arange(4*4*6).reshape(4,4,6)
    >>> A  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]],
           [[24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40, 41],
            [42, 43, 44, 45, 46, 47]],
           [[48, 49, 50, 51, 52, 53],
            [54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71]],
           [[72, 73, 74, 75, 76, 77],
            [78, 79, 80, 81, 82, 83],
            [84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95]]])
    >>> B = view_as_blocks(A, block_shape=(1, 2, 2))
    >>> B.shape
    (4, 2, 3, 1, 2, 2)
    >>> B[2:, 0, 2]  # doctest: +NORMALIZE_WHITESPACE
    array([[[[52, 53],
             [58, 59]]],
           [[[76, 77],
             [82, 83]]]])
    """
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length " "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out


def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.

    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).

    Parameters
    ----------
    arr_in : ndarray, shape (M[, ...])
        Input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.

    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.

    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_windows
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])

    >>> A = np.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])

    >>> A = np.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (
        (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
    ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out