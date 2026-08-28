import json
from pathlib import Path
import os
import re
from pathlib import Path
import os.path as op
import ismrmrd
from typing import List, Union, Tuple
import numpy as np


def sort_by_indexes(lst:List, indexes:Union[List[str],List[int]], reverse:bool=False) -> List:
    """
    Sort a list based on a list of indexes

    Parameters
    ----------
    
    lst : List,
        List

    indexes : Union[List[str],List[int]],
        List of index

    reverse : bool (optional, default : False),
        flag to reverse the sorting

    Returns
    -------

    sorted_lst : List,
       Sorted List

    """
    return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: x[0], reverse=reverse)]

def read_images_h5(filename:str)->List[np.ndarray]:
    """
    Getting the images from ISMRMD file

    Parameters
    ----------
    
    filename : str,
        ISMRMD file path

    Returns
    -------

    list_img : List[np.ndarray],
        List of images

    """
    list_img=[]
    with ismrmrd.File(filename,'r') as mrd:
        for key_img in list(mrd.find_images()):
            img=np.array(mrd[key_img].images.data) #np.array(mrd[key_img].images.data).T
            #Complex dtype: dtype([('real', '<f4'), ('imag', '<f4')])
            if len(img.dtype)==2:
                img=img['real']+1j*img['imag']
            list_img.append(img)
    return list_img

def read_headers_h5(filename:str)->List[dict]:
    """
    Getting the headers from ISMRMD file

    Parameters
    ----------
    
    filename : str,
        ISMRMD file path

    Returns
    -------

    list_headers : List[dict],
        List of ISMRMRD headers transformed in dictionnary

    """
    list_headers=[]
    with ismrmrd.File(filename,'r') as mrd:
        for key_img in list(mrd.find_images()):
            list_headers.append([dict(zip(mrd[key_img].images.headers[i].dtype.names,mrd[key_img].images.headers[i])) for i in range(mrd[key_img].images.headers.shape[0])])
    return list_headers

def read_h5(filename:str, ordered:str='image_series_index')-> Tuple[np.ndarray,dict]:
    """
    Getting the headers from ISMRMD file

    Parameters
    ----------
    
    filename : str,
        ISMRMD file path

    ordered : str (optional, default :'image_series_index'), #'image_index'
        string used for ordering the images and headers 

    Returns
    -------

    images : List[np.ndarray],
        List of images

    headers : List[dict],
        List of ISMRMRD headers transformed in dictionnary

    """
    headers = read_headers_h5(filename)
    images = read_images_h5(filename)
    if not(ordered == ""):
        indexes=[header[0][ordered] for header in headers]
        images=sort_by_indexes(images,indexes)
        headers=sort_by_indexes(headers,indexes)
    return images,headers

def resolve_env_path(value:str):
    """
    value: str
        A string that may contain an environment variable reference in the format ${env:VAR_NAME}.
    """
    match = re.fullmatch(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}", value)
    if match:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value:
            return Path(env_value)
    return None

def get_cmake_install_prefix(settings_path:str="/opt/code/gadgetron/.vscode/settings.json"):
    """
    settings_path: str
        Path to the settings.json file. Default is .vscode/settings.json
     Returns:
        str: The CMAKE_INSTALL_PREFIX value from settings.json or a default path if not found.
     Raises:
        ValueError: If the path specified in CMAKE_INSTALL_PREFIX does not exist.
     Notes:
        - The function checks for both "CMAKE_INSTALL_PREFIX" and "cmake.configureSettings.CMAKE_INSTALL_PREFIX" keys in the settings.json file.
        - If neither key is found, it defaults to "/opt/package/".
    """
    # Check well-known install prefixes first (RT container, dev container)
    well_known_prefixes = [
        os.environ.get("GADGETRON_HOME"),
        "/opt/conda/envs/gadgetron",
        "/opt/package",
    ]
    for prefix in well_known_prefixes:
        if prefix and op.isdir(op.join(prefix, "bin")):
            return prefix

    settings_file = Path(settings_path)
    cmake_settings =""
    if settings_file.exists():
        with open(settings_file) as f:
            raw = f.read()
            cleaned_str = re.sub(r'//.*', '', raw)
            # Remove /* ... */ comments
            cleaned_str = re.sub(r'/\*.*?\*/', '', cleaned_str, flags=re.DOTALL)

            cleaned_str=re.sub(r',\s*([\]}])', r'\1', cleaned_str)
            settings = json.loads(cleaned_str)
            if "CMAKE_INSTALL_PREFIX" in settings:
                cmake_settings= settings["CMAKE_INSTALL_PREFIX"]
            elif "cmake.configureSettings" in settings:
                cmake_settings = settings["cmake.configureSettings"].get("CMAKE_INSTALL_PREFIX")
            else:
                cmake_settings="/opt/package/"
    else:
        cmake_settings="/opt/package/"
    if not cmake_settings or not op.exists(cmake_settings):
        cmake_settings = resolve_env_path(cmake_settings) if cmake_settings else None
    if not cmake_settings or not op.exists(cmake_settings):
        raise ValueError(f"Path {cmake_settings} does not exist")
    return cmake_settings

def get_gadgetron_bin_path():
    """
    Retrieves the path to the Gadgetron binary from the CMAKE_INSTALL_PREFIX setting.
    Returns:
        str: The path to the Gadgetron binary.
    Raises:
        ValueError: If the CMAKE_INSTALL_PREFIX path does not exist.
    """
    cmake_install_prefix = get_cmake_install_prefix()
    gadgetron_bin_path = op.join(cmake_install_prefix, "bin")
    if not op.exists(gadgetron_bin_path):
        raise ValueError(f"Gadgetron binary path {gadgetron_bin_path} does not exist.")
    return gadgetron_bin_path

def get_gadgetron_config_path():
    """
    Retrieves the path to the Gadgetron config from the CMAKE_INSTALL_PREFIX setting.
    Returns:
        str: The path to the Gadgetron config.
    Raises:
        ValueError: If the CMAKE_INSTALL_PREFIX path does not exist.
    """
    cmake_install_prefix = get_cmake_install_prefix()
    gadgetron_config_path = op.join(cmake_install_prefix, "share","gadgetron","config")
    if not op.exists(gadgetron_config_path):
        raise ValueError(f"Gadgetron config path {gadgetron_config_path} does not exist.")
    return gadgetron_config_path