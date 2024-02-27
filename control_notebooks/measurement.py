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
DYG7_IP = "192.168.2.106" # rad hard camera on DYG7

OPTIONS = dict(
    n_fitting_results=60,
    target_charge=1.2e-9,
    target_charge_pv="AWAVXI11ICT:Ch4",
    charge_atol=0.1e-9,
)

DYG7_ROI = ROI(
    xcenter = 605,
    ycenter = 548,
    xwidth = 700,
    ywidth = 700
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
        gain=1.0, # 1.0 for 5 nC 5.0 for 1 nC,
        **OPTIONS
    )

def get_DYG15():
    return AWAEPICSImageDiagnostic(
        screen_name=screen_name,
        ip_address=DYG15_IP,
        alias="DYG15",
        resolution_suffix=None,
        roi=DYG15_ROI,
        gain=15.0,
        **OPTIONS
    )

def get_DYG7():
    return AWAEPICSImageDiagnostic(
        screen_name=screen_name,
        ip_address=DYG7_IP,
        alias="DYG7",
        resolution_suffix=None,
        roi=DYG7_ROI,
        gain=16.0,
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
            caput("13ARV1:cam1:Gain", diagnostic.gain)
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

def set_background(diagnostic, skip_test_measurement=False):
    set_camera(diagnostic)
    print("please shutter beam")
    input()
    diagnostic.measure_background()
    
    print("please un-shutter beam")
    input()
    
    plt.imshow(diagnostic.background_image)
    print(f"background file: {diagnostic.background_file}")
    if not skip_test_measurement:
        diagnostic.test_measurement()

    return diagnostic
