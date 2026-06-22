import gadgetron
import pyigtl
import numpy as np

import pyigtl.comm
import socket
import struct
import fcntl
import sys
import signal
import time


def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

def SlicerGadget(connection):
    print("STARTING SLICER GADGET")

    params = _parse_params(connection.config)

    if "local_port" in params:
        local_port = int(params["local_port"])
    else:
        local_port = 9004

    # try to load frame_discard from a file, if it doesn't exist default to 6
    try:
        frame_discard = int(np.load("frame_discard.npy"))
    except:
        frame_discard = 6
    print(f"Frame discard set to: {frame_discard}")

    rep_count = frame_discard+1 
    rep_buffer = []
    pyigtl.comm.OpenIGTLinkServer.__init__ = patched_init
    igt_server = pyigtl.OpenIGTLinkServer(local_port, local_server=False, iface="0.0.0.0")
    for data in connection:
        st = time.time()
        if data.repetition == rep_count:
            rep_buffer.append(data.data)
        else:
            im = pyigtl.ImageMessage(np.flip((np.stack(rep_buffer,axis=0).squeeze()).transpose(2,1,0), axis=0))
            igt_server.send_message(im)
            #print(f"Slicer Gadget Sent to 3D Slicer in: {time.time() - st}s")
            rep_buffer = [data.data]
            rep_count = data.repetition
        #im = pyigtl.ImageMessage(np.fliplr(data.data.squeeze().transpose(1,2,0)))
        #igt_server.send_message(im)
        #print(f"Slicer Gadget Sent to 3D Slicer in: {time.time() - st}s")
        connection.send(data)
    igt_server.server_close()



# Monkey patch the OpenIGTLinkServer class BEFORE using it
def patched_init(self, port=None, local_server=True, iface=None, start_now=True):
    pyigtl.comm.OpenIGTLinkBase.__init__(self)
    self.port = port
    
    if iface is None:
        iface = 'eth0'
    
    if local_server:
        self.host = "127.0.0.1"
    else:
        # Check if iface is already an IP address
        try:
            socket.inet_aton(iface)  # This will succeed if it's a valid IP
            self.host = iface  # Use the IP directly
        except socket.error:
            # It's an interface name, so resolve it
            if sys.platform.startswith('win32'):
                self.host = socket.gethostbyname(socket.gethostname())
            elif sys.platform.startswith('linux'):
                soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    ifname = iface
                    # Fix: encode the string to bytes for Python 3
                    self.host = socket.inet_ntoa(fcntl.ioctl(
                        soc.fileno(), 0x8915, 
                        struct.pack('256s', ifname.encode('utf-8')[:15])
                    )[20:24])
                except:
                    ifname = 'lo'
                    self.host = socket.inet_ntoa(fcntl.ioctl(
                        soc.fileno(), 0x8915, 
                        struct.pack('256s', ifname.encode('utf-8')[:15])
                    )[20:24])
            else:
                self.host = iface
    
    # Continue with the rest of the original initialization
    pyigtl.comm.SocketServer.TCPServer.allow_reuse_address = True
    pyigtl.comm.SocketServer.TCPServer.__init__(self, (self.host, self.port), pyigtl.comm.TCPRequestHandler)
    
    # Signal handlers
    self._previous_signal_handlers = {}
    self._previous_signal_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
    self._previous_signal_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)

    if start_now:
        self.start()


if __name__ == '__main__':
    # start with a monkey patched igt_server __init__
    pyigtl.comm.OpenIGTLinkServer.__init__ = patched_init
    gadgetron.external.listen(2020, SlicerGadget)