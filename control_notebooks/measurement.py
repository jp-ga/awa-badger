import sys
sys.path.append("../")
import pandas as pd
import time
from plugins.interfaces.diagnostics import AWAEPICSImageDiagnostic, ROI
from epics import caput, caget
import matplotlib.pyplot as plt

screen_name = "13ARV1"
TESTING = False
DYG14_IP = "192.168.2.111"
DYG15_IP = "192.168.2.139"

OPTIONS = dict(
    n_fitting_results=60,
    target_charge=1.2e-9,
    target_charge_pv="AWAVXI11ICT:Ch4",
    charge_atol=0.1e-9,
)

DYG14_ROI = ROI(
    xcenter = 556,
    ycenter = 806,
    xwidth = 700,
    ywidth = 700
)
DYG15_ROI = ROI(
    xcenter = 591,
    ycenter = 1079,
    xwidth = 700,
    ywidth = 700
)

def get_DYG14():
    return AWAEPICSImageDiagnostic(
        screen_name=screen_name,
        ip_address=DYG14_IP,
        alias="DYG14",
        resolution_suffix=None,
        roi=DYG14_ROI,
        **OPTIONS
    )

def get_DYG15():
    return AWAEPICSImageDiagnostic(
        screen_name=screen_name,
        ip_address=DYG15_IP,
        alias="DYG15",
        resolution_suffix=None,
        roi=DYG15_ROI,
        **OPTIONS
    )

def set_camera(diagnostic, testing=False):
    # stop the current camera
        print(f"setting camera {diagnostic.alias}")
        caput("13ARV1:cam1:Acquire", 0)
        time.sleep(2)

        print(str(caget("13ARV1:cam1:GC_SetCameraName")))
        current_ip = str(caget("13ARV1:cam1:GC_SetCameraName"))
        if not current_ip == diagnostic.ip_address:
            # set the new camera IP address
            print(f"setting IP address {diagnostic.ip_address}")
            caput("13ARV1:cam1:GC_SetCameraName", diagnostic.ip_address)
            time.sleep(2)
    
            # set the gain
            # start the new camera
            print(f"setting gain")
    
            if diagnostic.alias == "DYG14":
                caput("13ARV1:cam1:Gain", 5.0)
            elif diagnostic.alias == "DYG15":
                caput("13ARV1:cam1:Gain", 15.0)
            else:
                raise RuntimeError()
                
            time.sleep(2)
    
            # start the new camera
            print(f"starting acquisition")
            caput("13ARV1:cam1:Acquire", 1)
    
            time.sleep(2)
        else:
            print("ip address already set")
            # start the new camera
            print(f"starting acquisition")
            caput("13ARV1:cam1:Acquire", 1)

def set_background(diagnostic):
    set_camera(diagnostic)
    print("please shutter beam")
    input()
    diagnostic.measure_background()
    
    print("please un-shutter beam")
    input()
    
    plt.imshow(diagnostic.background_image)
    print(f"background file: {diagnostic.background_file}")
    diagnostic.test_measurement()

    return diagnostic
