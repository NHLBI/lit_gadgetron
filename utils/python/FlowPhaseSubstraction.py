import gadgetron
import numpy as np 
import ismrmrd as mrd
from utils_function import eprint, load_json, parse_params, read_params,create_ismrmrd_image_fast
        
def FlowPhaseSubstraction_gadget(connection):
    imgs=[]
    img_hdr=[]
    n=0
    #print(sys.stdout)
    params = parse_params(connection.config)
    
    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm

    eprint(f"Number of sets :{hdr.encoding[0].encodingLimits.set.maximum+1} ")
    nset=hdr.encoding[0].encodingLimits.set.maximum+1
    list_item=[[] for n in range(nset)]
    list_flags=[False for n in range(nset)]
    eprint("----------------------------TEST------------------------")

    connection.filter(lambda input: type(input) == gadgetron.IsmrmrdImageArray)
    for item in connection:
        if item.data.dtype==np.complex64:
            set_item=item.headers[0,0,0].set
            list_item[set_item]=item
            list_flags[set_item]=True
            eprint(list_flags)
            eprint(item.data.shape)   
            if not(False in list_flags):
                eprint('TWO sets')
                reference_img=list_item[0].data
                
                
                for i in range(1,len(list_item)):
                    set_img=list_item[i].data
                    #slices because number of Phase could be different between sets
                    s=np.s_[:,:,:,:,:min([reference_img.shape[-3],set_img.shape[-3]]),:,:]
                    #std::polar((std::abs(data1[i]) + std::abs(data2[i])) / 2.0f, std::arg(data1[i]) - std::arg(data2[i]));
                    new_img=0.5*(np.abs(set_img[s])+np.abs(reference_img[s]))*np.exp(1j*(np.angle(reference_img[s])-np.angle(set_img[s])))
                    eprint(i)
                    eprint(new_img.shape)
                    img_array=gadgetron.IsmrmrdImageArray(data=new_img,headers=list_item[i-1].headers,meta=list_item[0].meta)
                    connection.send(img_array)
        
if __name__ == "__main__":
    gadgetron.external.listen(21000,FlowPhaseSubstraction)