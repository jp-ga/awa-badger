from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from plugins.environments.environment import Environment, validate_observable_names
from plugins.interfaces.interface import Interface
from pydantic import Field, PositiveFloat

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# from plugins.interfaces.awa_interface import AWAInterface
# from plugins.interfaces.camera import AWACamera


class AWAEnvironment(Environment):
    name = "awa_environment"
    interface: Interface  # = AWAInterface(camera_app=AWACamera(None))

    target_charge_PV: str = "AWAVXI11ICT:Ch1"
    target_charge: Optional[PositiveFloat] = Field(
        None, description="magnitude of target charge in nC"
    )
    fractional_charge_deviation: PositiveFloat = Field(
        0.1, description="fractional deviation from target charge allowed"
    )

    def __init__(
            self, varaible_file: str, observable_file: str, interface: Interface,
            **kwargs
    ):
        # process variable and observable files to det variables and observables
        variable_info = pd.read_csv(varaible_file).set_index("NAME")
        observable_info = pd.read_csv(observable_file).set_index("NAME").T

        _variables = variable_info[["MIN", "MAX"]].T.to_dict()
        _observables = list(observable_info.keys())

        for name in _variables:
            _variables[name] = [_variables[name]["MIN"], _variables[name]["MAX"]]

        super(AWAEnvironment, self).__init__(
            variables=_variables,
            observables=_observables,
            interface=interface,
            **kwargs,
        )

    @validate_observable_names
    def get_observables(self, observable_names: List[str]) -> Dict:
        """make measurements until charge range is within bounds"""

        while True:
            if self.target_charge is not None:
                observable_names += [self.target_charge_PV]

            # remove duplicates
            observable_names = list(set(observable_names))

            # if a screen measurement is involved
            base_observable_names = [ele.split(":")[0] for ele in observable_names]
            screen_name = "13ARV1"

            if screen_name in base_observable_names:
                measurement = self.get_screen_measurement(
                    screen_name, observable_names
                )
            else:
                # otherwise do normal epics communication
                measurement = self.interface.get_channels(observable_names)

            if self.target_charge is not None:
                charge_value = measurement[self.target_charge_PV] * 1e9
                if self.is_inside_charge_bounds(charge_value):
                    break
                else:
                    print(f"charge value {charge_value} is outside bounds")
            else:
                break

        return measurement

    def get_screen_measurement(self, screen_name, extra_pvs=None):
        # roi_readbacks = [
        #    "ROI1:MinX_RBV",
        #    "ROI1:MinY_RBV",
        #    "ROI1:SizeX_RBV",
        #    "ROI1:SizeY_RBV",
        # ]

        # centroid_readbacks = [
        #    "Stats1:CentroidX_RBV",
        #    "Stats1:CentroidY_RBV",
        #    "Stats1:SigmaX_RBV",
        #    "Stats1:SigmaY_RBV",
        # ]

        extra_pvs = extra_pvs or []

        # construct list of all PVs necessary for measurement
        # observation_pvs = [
        #    f"{screen_name}:{pv_name}" for pv_name in roi_readbacks + centroid_readbacks
        # ] + extra_pvs

        # get rid of duplicate PVs
        # observation_pvs = list(set(observation_pvs))

        # do measurement and sort data
        observation_pvs = [
                              "13ARV1:image1:ArrayData",
                              "13ARV1:image1:ArraySize0_RBV",
                              "13ARV1:image1:ArraySize1_RBV"
                          ] + extra_pvs

        observation_pvs = list(set(observation_pvs))
        measurement = self.interface.get_channels(observation_pvs)

        img = measurement.pop("13ARV1:image1:ArrayData")
        img = img.reshape(
            measurement["13ARV1:image1:ArraySize1_RBV"],
            measurement["13ARV1:image1:ArraySize0_RBV"]
        )
        roi_data = np.array((100, 200, 1000, 1000))
        threshold = 200

        beam_data = get_beam_data(img, roi_data, threshold, visualize=False)
        measurement.update(
            {f"{screen_name}:{name}": beam_data[name] for name in beam_data}
        )
        return measurement

    def is_inside_charge_bounds(self, value):
        """test to make sure that charge value is within bounds"""
        if self.target_charge is not None:
            return (
                    self.target_charge * (1.0 - self.fractional_charge_deviation)
                    <= value
                    <= self.target_charge * (1.0 + self.fractional_charge_deviation)
            )
        else:
            return True


def get_beam_data(img, roi_data, threshold, visualize=True):
    """
        A method for processing raw screen images with a beam.

        As part of the analysis this function adds a bounding box (BB) around the beam
        distribution. The maximum BB distance from ROI cente is usable as a
        constraint, referred here as a `penalty` value. If less than zero, BB is
        entirely inside circular ROI, otherwise it is outside ROI. If the penalty
        function is positive, centroid and rms values returned are Nans.

        Returns a dict containing the following elements
            "Cx": beam centroid location in x
            "Cy": beam centroid location in y
            "Sx": rms beam size in x
            "Sy": rms beam size in y
            "penalty": penalty function value

        a region of interest (ROI) is specified as
        :                +------------------+
        :                |                  |
        :              height               |
        :                |                  |
        :               (xy)---- width -----+

        Parameters
        ----------
        img : np.ndarray
            n x m image data
        roi_data : List[str]
            list containing roi bounding box elements [x, y, width, height]
        threshold: int
            value to subtract from raw image, negative values after subtraction are
            set to zero
        visualize: bool, default: False
            flag to plot image and bounding box after processing

        Returns
        -------
        results : dict
            results dict
        """

    cropped_image = img[
        roi_data[0]:roi_data[0] + roi_data[2],
        roi_data[1]:roi_data[1] + roi_data[3]
    ]

    filtered_image = gaussian_filter(cropped_image, 3.0)

    thresholded_image = np.where(
        filtered_image - threshold > 0, filtered_image - threshold, 0
    )

    total_intensity = np.sum(thresholded_image)

    cx, cy, sx, sy = calculate_stats(thresholded_image)
    c = np.array((cx, cy))
    s = np.array((sx, sy))

    # get beam region
    n_stds = 2
    pts = np.array(
        (
            c - n_stds * s,
            c + n_stds * s,
            c - n_stds * s * np.array((-1, 1)),
            c + n_stds * s * np.array((-1, 1))
        )
    )

    # get distance from beam region to ROI center
    roi_c = np.array((roi_data[2], roi_data[3])) / 2
    roi_radius = np.min((roi_c * 2, np.array(thresholded_image.shape))) / 2

    # visualization
    if visualize:
        fig, ax = plt.subplots()
        c = ax.imshow(thresholded_image, origin="lower")
        ax.plot(cx, cy, "+r")
        fig.colorbar(c)

        rect = patches.Rectangle(pts[0], *s * n_stds * 2.0, facecolor='none',
                                 edgecolor="r")
        ax.add_patch(rect)

        circle = patches.Circle(roi_c, roi_radius, facecolor="none", edgecolor="r")
        ax.add_patch(circle)
        # ax2 = ax.twinx()
        # ax2.plot(thresholded_image.sum(axis=0))
        ax.set_ylim(0, 1000)

    distances = np.linalg.norm(pts - roi_c, axis=1)

    # subtract radius to get penalty value
    penalty = np.max(distances) - roi_radius

    # penalize no beam
    if total_intensity < 10000:
        penalty = np.linalg.norm(roi_c)

    results = {
        "Cx": cx,
        "Cy": cy,
        "Sx": sx,
        "Sy": sy,
        "penalty": penalty,
    }

    if penalty > 0:
        for name in ["Cx", "Cy", "Sx", "Sy"]:
            results[name] = None

    return results


def calculate_stats(img):
    rows, cols = img.shape
    row_coords = np.arange(rows)
    col_coords = np.arange(cols)

    m00 = np.sum(img)
    m10 = np.sum(col_coords[:, np.newaxis] * img.T)
    m01 = np.sum(row_coords[:, np.newaxis] * img)

    Cx = m10 / m00
    Cy = m01 / m00

    m20 = np.sum((col_coords[:, np.newaxis] - Cx) ** 2 * img.T)
    m02 = np.sum((row_coords[:, np.newaxis] - Cy) ** 2 * img)

    sx = (m20 / m00) ** 0.5
    sy = (m02 / m00) ** 0.5

    return Cx, Cy, sx, sy


def rectangle_union_area(llc1, s1, llc2, s2):
    # Compute the intersection of the two rectangles
    x1, y1 = llc1
    x2, y2 = llc2
    w1, h1 = s1
    w2, h2 = s2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap

    # Compute the areas of the two rectangles
    rect1_area = w1 * h1
    rect2_area = w2 * h2

    # Compute the area of the union
    union_area = rect1_area + rect2_area - overlap_area

    return union_area
