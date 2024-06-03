import yaml
CAMERA_CONFIG = yaml.safe_load(
    open("../awa_config/awa_camera_config.yml")
)

import sys
sys.path.append("../../")
from copy import deepcopy
from plugins.interfaces.diagnostics import (
    AWAFrameGrabberDiagnostic, AWABlackflyDiagnostic, ROI
)

def load_camera(camera_name: str):
    """ code to get camera config and initialize camera from yaml config file"""
    camera_config = deepcopy(CAMERA_CONFIG[camera_name])
    camera_type = camera_config.pop("type")

    roi = ROI(
        xcenter=int(camera_config["center"][0]),
        ycenter=int(camera_config["center"][1]),
        xwidth=int(camera_config["radius"]*1.75),
        ywidth=int(camera_config["radius"]*1.75)
    )

    if camera_type == "frame_grabber":
        image_diagnostic = AWAFrameGrabberDiagnostic(
            roi=roi, 
            alias=camera_name, 
            video_number=camera_config["video_number"]
        )

    elif camera_type == "blackfly":
        image_diagnostic = AWABlackflyDiagnostic(
            screen_name=camera_name,
            ip_address=camera_config["ip_address"],
            alias=camera_name,
            resolution_suffix=None,
            roi=roi,
            gain=camera_config["gain"], # 1.0 for 5 nC 5.0 for 1 nC,
        )
    else:
        raise RuntimeError(f"cannot load camera type {type}")

    image_diagnostic.set_camera()

    return image_diagnostic