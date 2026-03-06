
import ismrmrd as mrd
import numpy as np
import cupy as cp
import time

from typing import Tuple, Union
import mrdhelper


def extract_middle_slices(array, n, dim, shift=0):
    """
    Extract the middle `n` slices from a 3D NumPy array along a specified dimension.

    Parameters:
    array (np.ndarray): The 3D NumPy array from which slices will be extracted.
    n (int): The number of middle slices to extract.
    dim (int): The dimension along which to extract slices (0, 1, or 2).

    Returns:
    np.ndarray: A 4D NumPy array containing the middle `n` slices along the specified dimension.
    """
    if dim not in [0, 1, 2]:
        raise ValueError("Dimension must be 0, 1, or 2.")
    
    if n <= 0:
        raise ValueError("Number of slices `n` must be greater than 0.")
    
    # Get the size of the specified dimension
    shape = array.shape
    size = shape[dim]
    
    if size < n:
        raise ValueError("The array dimension size is smaller than the number of slices requested.")
    
    # Calculate the start and end indices for slicing
    mid_index = size // 2
    mid_index = mid_index + shift 
    half_n = n // 2
    
    start = max(mid_index - half_n, 0)
    end = min(mid_index + half_n + (n % 2), size)
    
    if dim == 0:
        return array[start:end, :, :]
    elif dim == 1:
        return array[:, start:end, :]
    elif dim == 2:
        return array[:, :, start:end]

def crop_middle(data, dims, perc):
    # function to crop data along axes specified in dims, perc = percentage of slices to keep
    n_slices_keep = [int(np.floor(data.shape[i] * perc)) for i in range(len(data.shape))]
    middle_slice = [int(np.floor(data.shape[i] / 2)) + 1 for i in range(len(data.shape))]
    bounds = [(int(middle_slice[i] - np.floor(n_slices_keep[i]/2)), int(middle_slice[i] + np.floor(n_slices_keep[i]/2))) for i in range(len(data.shape))]
    slices = [slice(bounds[i][0], bounds[i][1]) if i in dims else slice(None) for i in range(len(bounds))]
    dat_out =  data[tuple(slices)]

    return dat_out

def gram_schmidt(A):
    [m,n] = A.shape
    Q = cp.zeros((m,n), dtype=cp.complex64)
    R = cp.zeros((m,n), dtype=cp.complex64)
    for j in range(n):
        v = A[:,j]
        for i in range(j-1):
            R[i,j] = cp.dot(Q[:,i].conj().T, A[:,j])
            v = v - R[i,j]*Q[:,i]
        R[j,j] = cp.linalg.norm(v)
        Q[:,j] = v/R[j,j]
    return Q

def modified_gram_schmidt(A):
    m, n = A.shape
    Q = cp.zeros((m, n), dtype=cp.complex128) # Use complex128 for better precision
    R = cp.zeros((n, n), dtype=cp.complex128)
    
    V = A.copy().astype(cp.complex128)
    
    for i in range(n):
        R[i, i] = cp.linalg.norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]
        
        for j in range(i + 1, n):
            R[i, j] = cp.dot(Q[:, i].conj(), V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
            
    return Q, R


def rovir_apply(data, W, n_coil_keep):
    W = cp.copy(W)
    n_coil = W.shape[0]
    W = W[:, 0:n_coil_keep]
    data_size = data.shape

    #W = cp.copy(gram_schmidt(W))
    #[Q, R] = np.linalg.qr(W)
    #W = np.copy(Q)

    # ensure data.shape[-1] == n_coil 
    assert data.shape[-1] == n_coil

    data = data.reshape([np.prod(data_size[0:-1]),n_coil])
    data = data @ W
    data = data.reshape(list(data_size[0:-1]) + [n_coil_keep])
    return data

def online_tcr_ISTA(A, xn_1, ATyn, lam, max_eig, n_iter=40, del_0=None, cost_fn=None):
    costs = []
    if del_0 is None:
        del_n = cp.zeros(xn_1.shape)
    else:
        del_n = cp.copy(del_0)

    for i in range(n_iter):
        del_n = soft_threshold(del_n - (max_eig*A.N*(del_n+xn_1)) + (max_eig*ATyn), lam)
        del_n = crop_FOV_edge(del_n, 0.8)

        if cost_fn is not None:
            costs.append(cost_fn(del_n+xn_1) + (lam * cp.abs(cp.sum(del_n))))
            
    if cost_fn is not None:
        return [del_n, costs]

    return del_n

# online TCR reconstruction
def online_tcr_POGM(A, xn_1, ATyn, lam, step, n_iter=40, del_0=None, cost_fn=None):
    # understanding variable mapping <->
    # xk = del_n
    # yk = new variable
    # tk = new variable
    costs = []
    if del_0 is None:
        del_n = cp.zeros(xn_1.shape)
    else:
        del_n = cp.copy(del_0)
    
    w_n_1 = del_n
    theta_n_1 = 1
    
    z_n_1 = del_n 
    del_n_1 = del_n
    
    gamma_n_1 = 1

    for i in range(n_iter):
        if (i+1) < n_iter:
            theta_n = 0.5 * (1 + cp.sqrt((4 * cp.square(theta_n_1)) + 1))
        else:
            theta_n = 0.5 * (1 + cp.sqrt((8 * cp.square(theta_n_1)) + 1)) 

        gamma_n = step * (((2 * theta_n_1) + theta_n - 1) / (theta_n))
        w_n = del_n_1 - (step*A.N*(del_n_1+xn_1)) + (step*ATyn)

        z_n = w_n + (((theta_n_1 - 1) / (theta_n)) * (w_n - w_n_1)) + (((theta_n_1) / (theta_n)) * (w_n - del_n_1)) + \
            (((theta_n_1 - 1) / ((1/step)*gamma_n_1*theta_n)) * (z_n_1 - del_n_1))
        del_n = soft_threshold(z_n, lam)
        del_n = crop_FOV_edge(del_n, 0.8)

        # update theta, del, gamma, w, and z.
        theta_n_1 = theta_n
        del_n_1 = del_n
        gamma_n_1 = gamma_n
        w_n_1 = w_n
        z_n_1 = z_n

        if cost_fn is not None:
            costs.append(cost_fn(del_n+xn_1) + (lam * cp.abs(cp.sum(del_n))))

    if cost_fn is not None:
        return [del_n, costs]
    return del_n

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = cp.ogrid[:h, :w]
    dist_from_center = cp.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


# crop edges of the FOV
def crop_FOV_edge(x, amt=1):
    if amt > 1:
        print("AMT NEEDS TO BE LESS THAN 1")
        amt = 1
    x_fft = cp.fft.fftshift(cp.fft.fft2(x, axes=[0,1]), axes=[0,1])
    mask = create_circular_mask(x.shape[0], x.shape[1], radius=(x.shape[0]*amt/2))
    mask = cp.expand_dims(mask, 2)
    x_fft = cp.multiply(x_fft , mask)
    x_recomp = cp.fft.ifft2(cp.fft.ifftshift(x_fft, axes=[0,1]), axes=[0,1])
    return x_recomp

def norm(x):
    # L2 norm of vectorized x.
    # flatten x
    norm = (cp.sum(cp.square(cp.abs(x.flatten()))))
    return norm


def cg(Af, b, x0, niter=20, tol=1e-6):
    x = cp.copy(x0)
    r = Af(x) - b
    p = cp.copy(r)
    for i in range(niter):
        print(f"iter {i}")
        Ap = Af(p)
        rsold = cp.sum(cp.square(cp.abs(r)))
        alpha = rsold / np.vdot(p, Ap) 
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = cp.sum(cp.square(cp.abs(r)))
        if rsnew < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def FISTA_iteration(iter, xk_1, gradf, step_size, fista_dict=None, threshold=0, max_iter=10):
    if iter == 0:
        fista_dict = {
            'yk_1': xk_1,
            'tk_1': 1
        }
    elif iter >= max_iter:
        print("not iterating, max_iter reached")
        return [xk_1, fista_dict]
    tk_1 = fista_dict['tk_1']
    yk_1 = fista_dict['yk_1']

    tk = 0.5 * (1 + cp.sqrt((4 * cp.square(tk_1)) + 1))
    xk = yk_1 - (step_size * gradf(yk_1))
    xk = soft_threshold(xk, threshold)
    yk = xk + (((tk_1 - 1) / tk) * (xk - xk_1))
    return [xk, {'yk_1': yk, 'tk_1': tk}]

def POGM_iteration(iter, xk_1, gradf, step_size, pogm_dict=None, threshold=0, max_iter=10):
    if iter == 0:
        pogm_dict = {
            'zk_1': xk_1,
            'thetak_1': 1,
            'wk_1': xk_1,
            'gammak_1': 1
        }
    elif iter >= max_iter:
        print("not iterating, max_iter reached")
        return [xk_1, pogm_dict]
    thetak_1 = pogm_dict['thetak_1']
    gammak_1 = pogm_dict['gammak_1']
    zk_1 = pogm_dict['zk_1']
    wk_1 = pogm_dict['wk_1']

    if iter < max_iter:
        thetak = 0.5 * (1 + cp.sqrt((4 * cp.square(thetak_1)) + 1))
    else:
        thetak = 0.5 * (1 + cp.sqrt((8 * cp.square(thetak_1)) + 1))

    gammak = step_size * (((2 * thetak_1) + thetak - 1) / (thetak))
    wk = xk_1 - (step_size * gradf(xk_1))

    zk = wk + (((thetak_1 - 1) / thetak) * (wk - wk_1)) + (((thetak_1) / thetak) * (wk - xk_1)) + \
            + (((thetak_1 - 1) * step_size / (gammak_1 * thetak)) * (zk_1 - xk_1))
    
    xk = soft_threshold(zk, threshold)
    return [xk, {'zk_1': zk, 'thetak_1': thetak, 'wk_1': wk, 'gammak_1': gammak}]

def online_TCR_POGM_2(A, xn_1, ATyn, lam, step, n_iter=40, del_0=None, cost_fn=None):

    # construct gradf function
    gradf = lambda x: A.N * (x + xn_1) - ATyn

    if del_0 is None:
        del_n = cp.zeros(xn_1.shape)
    else:
        del_n = cp.copy(del_0)

    #initialize variables
    pogm_dict = None
    costs = []

    for i in range(n_iter):
        [del_n, pogm_dict] = POGM_iteration(i, del_n, gradf, step, pogm_dict, lam*step, n_iter)
        del_n = crop_FOV_edge(del_n, 0.6)
        if cost_fn is not None:
            costs.append(cost_fn(del_n+xn_1))

    if cost_fn is not None:
        return [del_n, costs]
    
    return del_n

# set up an online STCR reconstruction
def online_STCR_ISTA_timed(E, G, xn_1, Ahyn, lambdat, lambdas, step_size, mu=1 , yn=None, deln=None, time_recon=200, cost_fn=None):

    start_time = time.time()
    running = True

    costs = []
    if deln is None:
       deln = cp.zeros(xn_1.shape, Ahyn.dtype)
    # initialize etan
    zn = G*(xn_1 + deln)
    etan = cp.zeros(zn.shape, Ahyn.dtype)

    niter = 0

    while running:
        zn = soft_threshold(G*(deln+xn_1) + etan, lambdas/mu)
        inner_iter = 1 

        for i_i in range(inner_iter):
            deln = soft_threshold(deln - step_size *(E.H * E * (deln+xn_1) - Ahyn + (G.H*(G*(deln+xn_1) - zn + etan)*mu)), lambdat*step_size)
            #cost = norm(E * deln - yn + E*xn_1) + norm((cp.sqrt(mu) * (G * deln)) - (cp.sqrt(mu) * (zn-etan-(G*xn_1)))) + (lambdat * (cp.sum(cp.abs(deln))))
        etan = etan - zn + (G*(deln+xn_1))
        if cost_fn is not None:
            costs.append(cost_fn(deln))
        
        cp.cuda.stream.get_current_stream().synchronize()
        elapsed = time.time() - start_time

        niter = niter+1
        if elapsed*1000 > time_recon:
            running = False
        

    print(f"niter {niter}")
    if cost_fn is not None:
        return [deln, costs]
    return deln


# set up an online STCR reconstruction
def online_STCR_ISTA(E, G, xn_1, Ahyn, lambdat, lambdas, step_size, mu=1 , yn=None, deln=None, niter=20, cost_fn=None):
    costs = []
    if deln is None:
       deln = cp.zeros(xn_1.shape, Ahyn.dtype)
    # initialize etan
    zn = G*(xn_1 + deln)
    etan = cp.zeros(zn.shape, Ahyn.dtype)
    for n_i in range(niter):
        zn = soft_threshold(G*(deln+xn_1) + etan, lambdas/mu)
        inner_iter = 1 
        """
        # cg instead?
        Af = lambda x: (E.H * E * x + (mu * (G.H * G * x)))
        y = (E.H * (yn + E*xn_1)) + (mu * (G.H * (zn - etan - (G*xn_1))))
        deln = cg(Af, y, deln, niter=inner_iter)
        """
        for i_i in range(inner_iter):
            deln = soft_threshold(deln - step_size *(E.H * E * (deln+xn_1) - Ahyn + (G.H*(G*(deln+xn_1) - zn + etan)*mu)), lambdat*step_size)
            #cost = norm(E * deln - yn + E*xn_1) + norm((cp.sqrt(mu) * (G * deln)) - (cp.sqrt(mu) * (zn-etan-(G*xn_1)))) + (lambdat * (cp.sum(cp.abs(deln))))
        etan = etan - zn + (G*(deln+xn_1))
        if cost_fn is not None:
            costs.append(cost_fn(deln))
    if cost_fn is not None:
        return [deln, costs]
    return deln


# set up an online STCR reconstruction
def online_STCR_ISTA_2(E, G, xn_1, Ahyn, lambdat, lambdas, step_size, mu=1 , yn=None, deln=None, niter=20, cost_fn=None):

    costs = []
    if deln is None:
       deln = cp.zeros(xn_1.shape, Ahyn.dtype)
    # initialize etan
    zn = G*(xn_1 + deln)
    etan = cp.zeros(zn.shape, Ahyn.dtype)
    pogm_dict_outer = None
    for n_i in range(niter):
        gradout = lambda x: (zn - etan - (G*(deln+xn_1)))
        step_size_outer = 1

        inner_iter = 2 
        pogm_dict_inner = None
        for i_i in range(inner_iter):
            gradf = lambda x:(E.H * E * (deln+xn_1) - Ahyn + (G.H*(G*(deln+xn_1) - zn + etan)*mu) )
            [deln, pogm_dict_inner] = FISTA_iteration(i_i, deln, gradf, step_size, pogm_dict_inner, lambdat*step_size, inner_iter)

        #zn = soft_threshold(G*(deln+xn_1) + etan, lambdas/mu)
        [zn, pogm_dict_outer] = FISTA_iteration(n_i, zn, gradout, step_size_outer, pogm_dict_outer, lambdas/mu, niter)

        etan = etan - zn + (G*(deln+xn_1))
        if cost_fn is not None:
            costs.append(cost_fn(deln))
    if cost_fn is not None:
        return [deln, costs]
    return deln

# set up an online STCR reconstruction
def online_STCR_ISTA_2_timed(E, G, xn_1, Ahyn, lambdat, lambdas, step_size, mu=1 , yn=None, deln=None, time_recon=200, cost_fn=None):

    start_time = time.time()
    running = True
    n_i = 0

    costs = []
    if deln is None:
       deln = cp.zeros(xn_1.shape, Ahyn.dtype)
    # initialize etan
    zn = G*(xn_1 + deln)
    etan = cp.zeros(zn.shape, Ahyn.dtype)
    pogm_dict_outer = None
    while running:
        gradout = lambda x: (zn - etan - (G*(deln+xn_1)))
        step_size_outer = 1

        inner_iter = 1 
        pogm_dict_inner = None
        for i_i in range(inner_iter):
            gradf = lambda x:(E.H * E * (deln+xn_1) - Ahyn + (G.H*(G*(deln+xn_1) - zn + etan)*mu) )
            [deln, pogm_dict_inner] = FISTA_iteration(i_i, deln, gradf, step_size, pogm_dict_inner, lambdat*step_size, inner_iter)

        #zn = soft_threshold(G*(deln+xn_1) + etan, lambdas/mu)
        [zn, pogm_dict_outer] = FISTA_iteration(n_i, zn, gradout, step_size_outer, pogm_dict_outer, lambdas/mu, 100)

        etan = etan - zn + (G*(deln+xn_1))
        if cost_fn is not None:
            costs.append(cost_fn(deln))
        
        cp.cuda.stream.get_current_stream().synchronize()
        elapsed = time.time() - start_time
        print(elapsed)

        n_i = n_i + 1
        if elapsed*1000 > time_recon:
            running = False
    print(n_i)
    if cost_fn is not None:
        return [deln, costs]
    return deln



def crop_half_FOV(image, dims=(0,1), size=None):
    h,w = (image.shape[dims[0]], image.shape[dims[1]])
    if size is None:
        crop_h = h // 2
        crop_w = w // 2
        size = [crop_h, crop_w]

    start_indices = [
        (h - size[0]) // 2,
        (w - size[1]) // 2
    ]

    print(start_indices)
    print(size)

    # Create a slice object for each dimension
    slices = [slice(None) for i in range(len(image.shape))]
    slices[dims[0]] = slice(start_indices[0], start_indices[0] + size[0])
    slices[dims[1]] = slice(start_indices[1], start_indices[1] + size[1])

    # slices = [slice(start_indices[0], start_indices[0] + size[0]),
    #           slice(start_indices[1], start_indices[1] + size[1])]

    # Use numpy's advanced indexing to crop the image
    return image[tuple(slices)]


def soft_threshold(x, threshold):
    x_phase = cp.angle(x)
    # do I need cp.sign(x)? Probably only if it is a real numpber. maybe, 'x_phase' covers both.
    return cp.maximum(cp.abs(x) - threshold, 0)  * cp.exp(1j*x_phase)


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

def analyticaldcf(trajectory, adc_dwell=1e-6, ns=1):
    kxx = np.array(trajectory[:,0])
    kyy = np.array(trajectory[:,1])
    kzz = np.array(trajectory[:,2])

    # kx = kx[ndiscard:,0]
    # ky = ky[ndiscard:,0]
    gx = np.diff(np.concatenate(([0], kxx)), axis=0)/adc_dwell/42.58e6
    gy = np.diff(np.concatenate(([0], kyy)), axis=0)/adc_dwell/42.58e6

    # Analytical DCF formula
    # 1. Hoge RD, Kwan RKS, Bruce Pike G. Density compensation functions for spiral MRI. 
    # Magnetic Resonance in Medicine. 1997;38(1):117-128. doi:10.1002/mrm.1910380117
    cosgk = np.cos(np.arctan2(kxx, kyy) - np.arctan2(gx, gy))
    w = np.sqrt(kxx*kxx+kyy*kyy)*np.sqrt(gx*gx+gy*gy)*np.abs(cosgk)
    w[-int(ns//2):] = w[-int(ns//2)] # need this to correct weird jump at the end and improve SNR
    w = w/np.max(w)
    return w

def resize_with_crop_or_pad(image, target_shape):
    """
    Resize a 2D numpy array by center cropping and/or zero padding.
    
    Args:
        image: Input numpy array of shape (nx, ny)
        target_shape: Tuple (target_x, target_y) for desired output shape
    
    Returns:
        Resized numpy array of shape target_shape
    """
    current_shape = image.shape
    target_x, target_y = target_shape
    current_x, current_y = current_shape
    
    # Handle x dimension
    if current_x > target_x:
        # Need to crop in x dimension
        start_x = (current_x - target_x) // 2
        end_x = start_x + target_x
        image = image[start_x:end_x, :]
    elif current_x < target_x:
        # Need to pad in x dimension
        pad_x = target_x - current_x
        pad_before_x = pad_x // 2
        pad_after_x = pad_x - pad_before_x
        image = np.pad(image, ((pad_before_x, pad_after_x), (0, 0)), mode='constant', constant_values=0)
    
    # Handle y dimension
    current_y = image.shape[1]  # Update current_y after potential x operations
    if current_y > target_y:
        # Need to crop in y dimension
        start_y = (current_y - target_y) // 2
        end_y = start_y + target_y
        image = image[:, start_y:end_y]
    elif current_y < target_y:
        # Need to pad in y dimension
        pad_y = target_y - current_y
        pad_before_y = pad_y // 2
        pad_after_y = pad_y - pad_before_y
        image = np.pad(image, ((0, 0), (pad_before_y, pad_after_y)), mode='constant', constant_values=0)
    
    return image


def update_nonant(image, str_update, crop_percent=0.75):
    # first, extract the image from xn based on the str_update
    if str_update[0] == 'c':
        # coronal
        image_nonant = image[:, int(str_update[1:]), :]
    if str_update[0] == 's':
        # sagittal, so extract the image directly
        image_nonant = image[:,:,int(str_update[1:])]
    if str_update[0] == 'a':
        # axial
        image_nonant = image[int(str_update[1:]), :, :]
    
    # since each view is of different sizes (xy, xz, yz) we want to make sure the final image is zero-padded or cropped appropriately
    image_size = (int(image.shape[0] * crop_percent), int(image.shape[1] * crop_percent))
    image_nonant = resize_with_crop_or_pad(image_nonant, image_size)
    return image_nonant

    
def compute_preemphasis_order(n_slices):
    """
    Compute the 1-based ordering needed to pre-compensate for a 90-degree CLOCKWISE rotation
    after column-major montage arrangement.
    
    Args:
        n_slices: Total number of image slices (must be a perfect square)
    
    Returns:
        List of 1-based indices to reorder input array before montage
    """
    # Verify it's a perfect square
    grid_size = int(np.ceil(np.sqrt(n_slices)))

    desired_final = np.arange(1, n_slices + 1).reshape(grid_size, grid_size)
    
    before_rotation = np.rot90(desired_final, k=1)  # k=1 is counterclockwise
    
    preemphasis_values = before_rotation.T.flatten() - 1

    # take any preemphasis_values that are > n_slices and remove them (prevent wrong slicing)
    preemphasis_values = preemphasis_values[preemphasis_values < n_slices]

    return preemphasis_values.tolist()


def draw_number_indicators(image_array: np.ndarray, 
                          indicator_size: int = 20,
                          position: str = 'top-left',
                          border_width: int = 2,
                          fill_color: Union[int, float] = 255,
                          border_color: Union[int, float] = 0) -> np.ndarray:
    """
    Draw geometric indicators (squares) on images to represent their index numbers.
    Each image gets a number of squares equal to its 1-based index.
    
    Args:
        image_array: Input array of shape (height, width, num_images)
        indicator_size: Size of each square indicator
        position: Where to place indicators ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        border_width: Width of the border around each square
        fill_color: Color/intensity for the filled part of squares
        border_color: Color/intensity for the border of squares
    
    Returns:
        Modified image array with number indicators drawn on each image
    """
    
    # Create a copy to avoid modifying the original array
    result_array = image_array.copy()
    
    # Get dimensions
    height, width, num_images = image_array.shape
    
    for i in range(num_images):
        # Number of squares to draw (1-indexed)
        num_squares = i + 1
        
        # Calculate starting position based on position parameter
        if position == 'top-left':
            start_x, start_y = 5, 5
            dx, dy = indicator_size + 5, 0  # Horizontal arrangement
        elif position == 'top-right':
            start_x = width - 5 - num_squares * (indicator_size + 5) + 5
            start_y = 5
            dx, dy = indicator_size + 5, 0
        elif position == 'bottom-left':
            start_x, start_y = 5, height - indicator_size - 5
            dx, dy = indicator_size + 5, 0
        elif position == 'bottom-right':
            start_x = width - 5 - num_squares * (indicator_size + 5) + 5
            start_y = height - indicator_size - 5
            dx, dy = indicator_size + 5, 0
        else:
            # Default to top-left
            start_x, start_y = 5, 5
            dx, dy = indicator_size + 5, 0
        
        # Draw the squares
        for square_idx in range(num_squares):
            x = int(start_x + square_idx * dx)
            y = int(start_y + square_idx * dy)
            
            # Make sure we don't go out of bounds
            if x + indicator_size >= width or y + indicator_size >= height:
                break
                
            # Draw filled square
            result_array[y:y+indicator_size, x:x+indicator_size, i] = fill_color
            
            # Draw border
            if border_width > 0:
                # Top and bottom borders
                result_array[y:y+border_width, x:x+indicator_size, i] = border_color
                result_array[y+indicator_size-border_width:y+indicator_size, x:x+indicator_size, i] = border_color
                
                # Left and right borders
                result_array[y:y+indicator_size, x:x+border_width, i] = border_color
                result_array[y:y+indicator_size, x+indicator_size-border_width:x+indicator_size, i] = border_color
    
    return result_array

def replace_zero(array): 
    
    for i in range(len(array)) :
        if array[i] == 0 : 
            array[i] = 1
    return array

def ram_schmidt(A,norm=True,row_vect=False):
    """Orthonormalizes vectors by gram-schmidt process
    
    Parameters
    -----------
    A : ndarray,
    Matrix having vectors in its columns
    
    norm : bool,
    Do you need Normalized vectors?
    
    row_vect: bool,
    Does Matrix A has vectors in its rows?
    
    Returns 
    -------
    G : ndarray,
    Matrix of orthogonal vectors 
    
    """
    if row_vect :
        # if true, transpose it to make column vector matrix
        A = A.T
    
    no_of_vectors = A.shape[1]
    G = A[:,0:1].copy() # copy the first vector in matrix
    # 0:1 is done to to be consistent with dimensions - [[1,2,3]]
    
    # iterate from 2nd vector to number of vectors
    for i in range(1,no_of_vectors):
        
        # calculates weights(coefficents) for every vector in G
        numerator = A[:,i].dot(G)
        denominator = np.diag(np.dot(G.T,G)) #to get elements in diagonal
        weights = np.squeeze(numerator/denominator)
        
        # projected vector onto subspace G 
        projected_vector = np.sum(weights * G,
                                  axis=1,
                                  keepdims=True)
        
        # orthogonal vector to subspace G
        orthogonalized_vector = A[:,i:i+1] - projected_vector
        
        # now add the orthogonal vector to our set 
        G = np.hstack((G,orthogonalized_vector))
        
    if norm :
        # to get orthoNORMAL vectors (unit orthogonal vectors)
        # replace zero to 1 to deal with division by 0 if matrix has 0 vector
        G = G/replace_zero(np.linalg.norm(G,axis=0))
    
    if row_vect:
        return G.T
    
    return G

def fair(x, delta):
    return delta**2 * (cp.abs(x / delta) - cp.log(1 + cp.abs(x / delta)))

def d_fair(x, delta):
    return x / (1 + cp.abs(x / delta))

def grad_cost(x, x_prev, A, y, D, delta, lambdat, lambdas):
    AhAx = A.H * (A * x - y)
    spatial = lambdas * (D.H * d_fair(D * x, delta))
    temporal = lambdat * d_fair(x - x_prev, delta)
    return AhAx + spatial + temporal

def gradient_descent_iteration(xk_1, gradf, step_size):
    xk = xk_1 - (step_size * gradf(xk_1))
    return xk

def gradient_descent_iteration_nesterov(iter, xk_1, gradf, step_size, nesterov_dict=None, max_iter=10):
    if iter == 0:
        nesterov_dict = {
            'yk_1': xk_1,
            'tk_1': 0
        }
    elif iter >= max_iter:
        print("not iterating, max_iter reached")
        return [xk_1, nesterov_dict]
    tk_1 = nesterov_dict['tk_1']
    yk_1 = nesterov_dict['yk_1']

    tk = 0.5 * (1 + cp.sqrt((4 * cp.square(tk_1)) + 1))
    xk = yk_1 - (step_size * gradf(yk_1))
    yk = xk + (((tk_1 - 1) / tk) * (xk - xk_1))
    return [xk, {'yk_1': yk, 'tk_1': tk}]