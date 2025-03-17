import sys
import json
import ismrmrd as mrd

def eprint(*args, **kwargs):
    """_summary_
    """
    print(*args, file=sys.stderr, **kwargs)

def load_json(file: str):
    """_summary_

    Args:
        file (str): _description_

    Returns:
        _type_: _description_
    """
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def parse_params(xml):
    """_summary_

    Args:
        xml (_type_): _description_

    Returns:
        _type_: _description_
    """
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

def read_params(params_init,params_ref,boolean_keys=[],str_keys=[],int_keys=[],float_keys=[]):
    """_summary_

    Args:
        params_init (_type_): _description_
        params_ref (_type_): _description_
        boolean_keys (list, optional): _description_. Defaults to [].
        str_keys (list, optional): _description_. Defaults to [].
        int_keys (list, optional): _description_. Defaults to [].
        float_keys (list, optional): _description_. Defaults to [].
    """

    for bkey in boolean_keys:
        if bkey in params_init:
            params_ref[bkey]=params_init[bkey]=='True'
    for skey in str_keys:
         if skey in params_init:
            params_ref[skey]=params_init[skey]
    for ikey in int_keys:
        if ikey in params_init:
            params_ref[ikey]=int(params_init[ikey])
    for fkey in float_keys:
        if fkey in params_init:
            params_ref[ikey]=float(params_init[ikey])
    
    return params_ref

def create_ismrmrd_image_fast(data, field_of_view, index,serie_index,image_type=mrd.IMTYPE_MAGNITUDE,transpose=False):
        return mrd.image.Image.from_array(
            data,
            image_index=index,
            image_series_index=serie_index,
            image_type=image_type,
            field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z),
            transpose=transpose
        )