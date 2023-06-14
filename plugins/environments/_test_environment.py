from typing import List, Dict

from pydantic import BaseModel, Field, validator, create_model

from plugins.interfaces.interface import Interface


class EnvironmentVariables(BaseModel):
    interface: Interface
    x1: float = Field(default=0.0, le=10.0, ge=-10.0, description="x1 power supply")
    x2: float = Field(default=1.0, le=1.0, ge=-10.0, description="x2 power supply")

    @validator("*")
    def validate(cls, value, values, config, field):
        name = field.name
        if name != "interface":
            interface = values["interface"]
            interface.set_channels({name: value})

        return value

    class Config:
        validate_all = True
        validate_assignment = True


class DummyInterface(Interface):
    def set_channels(self, channel_inputs: Dict[str, float]):
        for name, val in channel_inputs.items():
            print(f"setting {name} to {val}")

    def get_channels(self, channels: List[str]) -> Dict[str, float]:
        pass


d = DummyInterface(name="dummy")
v = EnvironmentVariables(interface=d)
v.x1 = 8.0

print(v.interface == d)
print(v.__fields__)
print(v.json())
