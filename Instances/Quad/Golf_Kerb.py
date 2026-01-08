import numpy as np
from VehicleModel4Wh import Vehicle
from Instances.Quad.base import Golf, MRF13570R1269S
currveh = Golf()
currtire = MRF13570R1269S()
instance = Vehicle(
    np.array([1071.5, 285.25, 1350.214]), #r_A -  Top of the Kingpin
    np.array([936.25, 410, 960.2469]), # Tierod Outer Ball Joint
    np.array([954.83, 176.07, 991.0]), # Tierod Inner Ball Joint
    np.array([1000, 490, 887.2469]), # Wheel Center
    np.array([999.9981, 411.2266, 825.4318]), # Bottom of the Kingpin
    r_La = np.array([919.9981, 123.229, 875.5855]), # Lower A-Arm Bush Front
    r_Lb = np.array([1079.9, 123.229, 875.5855]), # Lower A-Arm Bush Rear
    r_strut= np.array([1021.3, 363.0444, 924.0705]), # Strut Top Mount
    GVW = 606.16,
    b = 799.72,
    CG_height = 304.33,
    slr = currveh.slr,
    dlr = currveh.dlr,
    initial_camber = currveh.initial_camber,
    toe_in = currveh.toe_in,
    twf = currveh.twf,
    twr= currveh.twr,
    wb = currveh.wb,
    wheel_rate_f = currveh.wr_front,
    wheel_rate_r = currveh.wr_rear,
    tire_stiffness_f = currveh.tire_stiffness_front,
    tire_stiffness_r = currveh.tire_stiffness_rear,
    pinion = currveh.pinion,
    tirep = currveh.tirep,
    dila = currveh.dila,
    assumed_rack_stroke = currveh.assumed_rack_stroke,
    linkage_effort = currveh.linkage_effort,
    tiredata = currtire.tiredata,
    CF_Loads = currtire.CF_Loads,
    CF_Stiffnessrad = currtire.CF_Stiffnessrad,
    CF_pneumatictrail = currtire.pneumatictrail
)