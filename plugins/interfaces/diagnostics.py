import json
import os.path
import time
from copy import copy
from time import sleep
from typing import Union, List, Optional
from pprint import pprint

import h5py
import numpy as np
import pandas as pd
import yaml
from epics import PV
from matplotlib import patches, pyplot as plt
from pydantic import BaseModel, PositiveFloat, PositiveInt
from epics import caget_many, caput, caget
from plugins.interfaces.utils.fitting_methods import fit_gaussian_linear_background


class ROI(BaseModel):
    xcenter: int
    ycenter: int
    xwidth: int
    ywidth: int

    @property
    def bounding_box(self):
        return [self.xcenter - int(self.xwidth/2),
                self.ycenter - int(self.ywidth/2),
                self.xwidth, self.ywidth]

    def crop_image(self, img):
        x_size, y_size = img.shape

        if self.xwidth > x_size or self.ywidth > y_size:
            raise ValueError(
                f"must specify ROI that is smaller than the image, "
                f"image size is {img.shape}"
            )

        bbox = self.bounding_box
        img = img[bbox[0]: bbox[0] + bbox[2], bbox[1]: bbox[1] + bbox[3]]

        return img


class AWAEPICSImageDiagnostic(BaseModel):
    screen_name: str

    alias: Optional[str] = None
    array_data_suffix: str = "image1:ArrayData"
    array_n_cols_suffix: str = "image1:ArraySize0_RBV"
    array_n_rows_suffix: str = "image1:ArraySize1_RBV"
    resolution_suffix: Union[str, None] = "RESOLUTION"
    resolution: float = 1.0
    beam_shutter_pv: str = None
    extra_pvs: List[str] = []

    background_file: str = None
    save_image_location: Union[str, None] = None
    roi: ROI = None

    min_log_intensity: float = 4.0
    bounding_box_half_width: PositiveFloat = 3.0
    wait_time: PositiveFloat = 1.0
    n_fitting_restarts: PositiveInt = 1
    visualize: bool = False
    verbose: bool = True
    return_statistics: bool = False
    threshold: float = 0.0
    apply_bounding_box_constraint: bool = True

    target_charge: Optional[PositiveFloat] = None
    target_charge_pv: Optional[str] = None
    charge_atol: Optional[PositiveFloat] = 0.1

    testing: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # create PV objects
        self._pvs = [
            PV(name) for name in self.pv_names
        ]
        if self.beam_shutter_pv is not None:
            self._shutter_pv_obj = PV(self.beam_shutter_pv)

    def measure_beamsize(self, n_shots: int = 1, fit_image=True, **kwargs):
        """
        conduct a multi-shot measurement to get the beam size from images, returns
        sizes in units of `resolution`

        allows attaching extra information to dataset via kwargs
        """
        results = []
        images = []
        start_time = time.time()

        while len(results) < n_shots:

            # get image and PV's at the same time
            img, extra_data, raw_img = self.get_processed_image()

            # check if charge measurement is inside window, if it is skip to another
            # measurement
            if self.target_charge is not None:
                assert self.target_charge_pv is not None
                charge_error = abs(
                    extra_data[self.target_charge_pv] - self.target_charge
                )
                print(f"measured charge:  {extra_data[self.target_charge_pv]}")
                # add message, wait and continue
                if charge_error > self.charge_atol:
                    if self.verbose:
                        print(
                            f"charge error {charge_error:.2} "
                            f"outside atol {self.charge_atol}"
                        )
                    sleep(self.wait_time)
                    continue

            if fit_image:
                result = self.calculate_beamsize(img, raw_img)

                # convert beam size results to microns
                if result["Sx"] is not None:
                    result["Sx"] = result["Sx"] * self.resolution
                    result["Sy"] = result["Sy"] * self.resolution
            else:
                result = {}

            results += [result | extra_data]
            images += [img]

            sleep(self.wait_time)

        # combine data into a single dictionary output
        if n_shots == 1:
            outputs = results[0]
        else:
            # collect results into lists
            outputs = pd.DataFrame(results).reset_index().to_dict(orient="list")
            outputs.pop("index")

            # if the number of nans is greater than half but less than all of
            # the number of shots this could be an inconsistent measurement -- raise a warning
            #n_nans = np.array(outputs["Sx"]).isna().sum()
            #if n_nans > n_shots / 2 or n_nans < n_shots:
            #    warning.warn(
            #        "The number of invalid measurements is greater than half the" + \
            #        "number of shots but is not all of the measurements." + \
            #        "This could indicate consistency issues in the measurement."
            #    )

            # create numpy arrays from lists
            outputs = {key: list(np.array(ele)) for key, ele in outputs.items()}

            # if specified, modify dictionary elements to return
            # statistics of numerical lists
            if self.return_statistics:
                new_outputs = {}
                for name, ele in outputs.items():
                    if isinstance(ele, list):
                        if isinstance(ele[0], float):
                            new_outputs[name] = np.array(ele).mean()
                            new_outputs[f"{name}_var"] = np.array(ele).var()
                        else:
                            new_outputs[name] = ele
                    else:
                        new_outputs[name] = ele

                outputs = new_outputs

        # if specified, save image data to location based on time stamp
        if self.save_image_location is not None:
            if self.alias is not None:
                name = self.alias
            else:
                name = self.screen_name.replace(":", "_")
            save_filename = os.path.join(
                self.save_image_location, f"{name}_{int(start_time)}.h5"
            )
            screen_stats = json.loads(self.model_dump_json())
            with h5py.File(save_filename, "w") as hf:
                dset = hf.create_dataset("images", data=np.array(images))
                for name, val in (outputs | kwargs | screen_stats).items():
                    if val is not None:
                        try:
                            dset.attrs[name] = val
                        except TypeError:
                            dset.attrs[name] = str(val)

            outputs["save_filename"] = save_filename

        return outputs

    def test_measurement(self):
        """test the beam size measurement w/o saving data"""
        old_visualize_state = copy(self.visualize)
        old_save_location = copy(self.save_image_location)
        self.visualize = True
        self.save_image_location = None
        results = self.measure_beamsize(n_shots=1)
        self.visualize = old_visualize_state
        self.save_image_location = old_save_location

        return results

    @property
    def pv_names(self) -> list:
        suffixes = [
            self.array_data_suffix,
            self.array_n_cols_suffix,
            self.array_n_rows_suffix,
        ]
        if self.resolution_suffix is not None:
            suffixes += [self.resolution_suffix]

        return [f"{self.screen_name}:{ele}" for ele in suffixes]

    @property
    def background_image(self) -> Union[np.ndarray, float]:
        if self.background_file is not None:
            return np.load(self.background_file)
        else:
            return 0.0

    def get_processed_image(self):
        raw_img, extra_data = self.get_raw_data()
        img = copy(raw_img)

        # subtract background
        img = img - self.background_image
        img = np.where(img >= 0, img, 0)

        # crop image if specified
        if self.roi is not None:
            img = self.roi.crop_image(img.T).T

        return img, extra_data, raw_img

    def measure_background(self, n_measurements: int = 5, file_location: str = None):
        loc = copy(self.save_image_location)
        file_location = file_location or loc

        if self.alias is not None:
            name = self.alias
        else:
            name = self.screen_name.replace(":", "_")

        filename = os.path.join(file_location, f"{name}_background.npy")

        print("please shutter beam")
        input()
        images = []
        for i in range(n_measurements):
            images += [self.get_raw_data()[0]]
            sleep(self.wait_time)
            
        print("please un-shutter beam")
        input()
                
        # restore shutter state
        if self.beam_shutter_pv is not None:
            self._shutter_pv_obj.put(old_shutter_state)

        # return average
        images = np.stack(images)
        mean = images.mean(axis=0)

        np.save(filename, mean)
        self.background_file = filename

        return mean

    def calculate_beamsize(self, img, raw_img):
        # apply threshold
        img = img - self.threshold
        img = np.where(img >= 0, img, 0)

        # visualize image
        if self.visualize:
            print("displaying image")
            fig, ax = plt.subplots(2,1)
            c = ax[0].imshow(raw_img, origin="lower")
            ax[1].imshow(img, origin="lower")

            fig.colorbar(c)

        # if image is below min intensity threshold avoid fitting
        log10_total_intensity = np.log10(img.sum())
        if log10_total_intensity < self.min_log_intensity:
            print(f"log10 image intensity {log10_total_intensity} below threshold")

            result = {
                "Cx": np.NaN,
                "Cy": np.NaN,
                "Sx": np.NaN,
                "Sy": np.NaN,
                "bb_penalty": np.NaN,
                "total_intensity": 10**log10_total_intensity,
                "log10_total_intensity": log10_total_intensity,
            }
            return result

        else:
            print("fitting image")
            fits = self.fit_image(img)
            centroid = fits["centroid"]
            sizes = fits["rms_sizes"]

            # do analysis if fits return all good values
            if np.all(~np.isnan(np.stack((centroid, sizes)))):
                # get beam region bounding box
                n_stds = self.bounding_box_half_width
                pts = np.array(
                    (
                        centroid - n_stds * sizes,
                        centroid + n_stds * sizes,
                        centroid - n_stds * sizes * np.array((-1, 1)),
                        centroid + n_stds * sizes * np.array((-1, 1)),
                    )
                )
                roi_c = np.array([self.roi.xcenter, self.roi.ycenter])
                roi_radius = self.roi.xwidth / 2
                
                # visualization
                if self.visualize:
                    ax[1].plot(*centroid, "+r")
                    ax[0].plot(*roi_c, ".r")

                    rect = patches.Rectangle(
                        pts[0], 
                        *sizes * n_stds * 2.0, 
                        facecolor="none", edgecolor="r"
                    )
                    ax[1].add_patch(rect)

                    # plot bounding circle
                    circle = patches.Circle(
                        roi_c, self.roi.xwidth/2,
                        facecolor="none", edgecolor="r"
                    )
                    ax[0].add_patch(circle)

                    circle2 = patches.Circle(
                        (
                            self.roi.xwidth/2,
                            self.roi.xwidth/2
                        ), 
                        self.roi.xwidth/2,
                        facecolor="none", edgecolor="r"
                    )
                    ax[1].add_patch(circle2)
                    
                temp = pts - np.array((
                        self.roi.xwidth/2,
                        self.roi.xwidth/2
                    ))
                distances = np.linalg.norm(
                    temp,
                    axis=1
                )
                # subtract radius to get penalty value
                bb_penalty = np.max(distances) - roi_radius

                log10_total_intensity = fits["log10_total_intensity"]

                result = {
                    "Cx": centroid[0],
                    "Cy": centroid[1],
                    "Sx": sizes[0],
                    "Sy": sizes[1],
                    "bb_penalty": bb_penalty,
                    "total_intensity": fits["total_intensity"],
                    "log10_total_intensity": log10_total_intensity,
                }

                # set results to none if the beam extends beyond the roi and the
                # bounding box constraint is active
                if bb_penalty > 0 and self.apply_bounding_box_constraint:
                    for name in ["Cx", "Cy", "Sx", "Sy"]:
                        result[name] = np.NaN

            else:
                result = {
                    "Cx": np.NaN,
                    "Cy": np.NaN,
                    "Sx": np.NaN,
                    "Sy": np.NaN,
                    "bb_penalty": np.NaN,
                    "total_intensity": fits["total_intensity"],
                    "log10_total_intensity": log10_total_intensity,
                }

            if self.visualize:
                pprint(result)

            return result

    def fit_image(self, img):
        x_projection = np.sum(img, axis=0)
        y_projection = np.sum(img, axis=1)

        # subtract min value from projections
        x_projection = x_projection - x_projection[:10].min()
        y_projection = y_projection - y_projection[:10].min()

        para_x = fit_gaussian_linear_background(
            x_projection, visualize=self.visualize, n_restarts=self.n_fitting_restarts
        )
        para_y = fit_gaussian_linear_background(
            y_projection, visualize=self.visualize, n_restarts=self.n_fitting_restarts
        )

        return {
            "centroid": np.array((para_x[1], para_y[1])),
            "rms_sizes": np.array((para_x[2], para_y[2])),
            "total_intensity": img.sum(),
            "log10_total_intensity": np.log10(img.sum()),
        }

    def yaml(self):
        return yaml.dump(self.dict(), default_flow_style=None, sort_keys=False)

    def dump_yaml(self, fname):
        """dump data to file"""
        output = json.loads(self.json())
        with open(fname, "w") as f:
            yaml.dump(output, f)        

class AWABlackflyDiagnostic(AWAEPICSImageDiagnostic):
    ip_address: str
    gain: PositiveFloat = 1.0

    def set_camera(self):
        print(f"setting camera {self.alias}")
        caput("13ARV1:cam1:Acquire", 0)
        time.sleep(2)

        print(str(caget("13ARV1:cam1:GC_SetCameraName")))
        current_ip = str(caget("13ARV1:cam1:GC_SetCameraName"))
        if not current_ip == self.ip_address:
            # set the new camera IP address
            print(f"setting IP address {self.ip_address}")
            caput("13ARV1:cam1:GC_SetCameraName", self.ip_address)
            time.sleep(2)
    
            # set the gain
            # start the new camera
            print(f"setting gain")
            caput("13ARV1:cam1:Gain", self.gain)
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

    def get_raw_data(self) -> (np.ndarray, dict):
        if self.testing:
            img = np.zeros((2000, 2000))
            img[800:-800, 900:-900] = 1
            self.resolution = 1.0
            extra_data = {
                "ICT1": np.random.randn()*0.1 + 1.0,
                "ICT2": np.random.randn()*0.1 + 1.0
            }
        else:
            # get pvs
            results = caget_many(self.pv_names)
            if results[0] is None:
                raise RuntimeError(
                    "epics returned no image values, make sure blackfly phobeus is active"
                )
            results[0] = np.uint16(results[0])

            e_pvs = copy(self.extra_pvs)
            if self.target_charge_pv is not None:
                e_pvs += [self.target_charge_pv]
            extra_data = dict(zip(e_pvs, caget_many(e_pvs)))
            img, nx, ny = results[0], results[1], results[2]
            img = img.reshape(ny, nx)

        return img, extra_data


class AWAFrameGrabberDiagnostic(AWAEPICSImageDiagnostic):
    screen_name: str = "AWANIFrameGrabber"
    array_data_suffix: str = "N/A"
    array_n_cols_suffix: str = "N/A"
    array_n_rows_suffix: str = "N/A"
    resolution_suffix: Union[str, None] = None
    video_number: int


    @property
    def pv_names(self) -> list:
        return ["AWANIFG:ImgData"]

    def set_camera(self):
        input(f"change BNC to V{self.video_number}") 

    def get_raw_data(self) -> (np.ndarray, dict):
        if self.testing:
            img = np.zeros((2000, 2000))
            img[800:-800, 900:-900] = 1
            self.resolution = 1.0
            extra_data = {
                "ICT1": np.random.randn()*0.1 + 1.0,
                "ICT2": np.random.randn()*0.1 + 1.0
            }
        else:
            # get pvs
            results = caget_many(self.pv_names)
            e_pvs = copy(self.extra_pvs)
            if self.target_charge_pv is not None:
                e_pvs += [self.target_charge_pv]
            extra_data = dict(zip(e_pvs, caget_many(e_pvs)))
            img = results[0]
            if img is None:
                raise RuntimeError(
                    "epics returned no image values, make sure frame grabber is active"
                )
            img = img.reshape(480, 640)

        return img, extra_data
