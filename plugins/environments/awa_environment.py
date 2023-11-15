from typing import Dict, List, Optional
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from plugins.environments.environment import Environment, validate_observable_names
from plugins.interfaces.interface import Interface
from pydantic import Field, PositiveFloat, PositiveInt

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
    n_samples: PositiveInt = 5

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

        measurements = []
        if self.target_charge is not None:
            observable_names += [self.target_charge_PV]
            
        # if a screen measurement is involved
        base_observable_names = [ele.split(":")[0] for ele in observable_names]
        screen_name = "13ARV1"
        
        # remove duplicates
        observable_names = list(set(observable_names))
        
        # remove names with screen name in it
        observable_names = [ele for ele in observable_names if not screen_name in ele]
            
        for i in range(self.n_samples):
            while True:
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
                        pass
                        #print(f"charge value {charge_value} is outside bounds")
                else:
                    break
            measurements += [measurement]
            time.sleep(0.75)

        def add_suffix(series, suffix):
            vm = pd.Series([])
            for k in series.keys():
                vm[k + suffix] = series[k]
            return vm

        # create a dataframe
        df = pd.DataFrame(measurements)
        mean_results = df.mean()
        std_results = add_suffix(df.std(), "_std")

        return pd.concat([mean_results, std_results]).to_dict()

    def get_screen_measurement(self, screen_name, extra_pvs=None, visualize=False):
        extra_pvs = extra_pvs or []

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
        roi_data = np.array((350, 700, 600, 600))
        threshold = 150

        beam_data = get_beam_data(img, roi_data, threshold, visualize=visualize)
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

