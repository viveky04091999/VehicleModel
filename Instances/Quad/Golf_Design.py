import numpy as np
from VehicleModel4Wh import Vehicle
from Instances.Quad.base import Golf, MRF13570R1269S
currveh = Golf()
currtire = MRF13570R1269S()
instance = Vehicle(
    np.array([983.1855, 226.27, 1487.214]), #r_A -  Top of the Kingpin
    np.array([846.75, 309.18, 1126.7506]), # Tierod Outer Ball Joint
    np.array([984.75, 20.67, 1103.56]), # Tierod Inner Ball Joint
    np.array([899.7526, 488.495, 999.94]), # Wheel Center
    np.array([898.75, 368.88001, 936.8678]), # Bottom of the Kingpin
    r_La = np.array([998.75, 104.97, 940.9685]), # Lower A-Arm Bush Front
    r_Lb = np.array([798.75, 104.97, 940.9685]), # Lower A-Arm Bush Rear
    r_strut= np.array([912.26, 320.302, 1081.5647]), # Strut bottom Mount
    GVW = 597,
    b = 493.95,
    CG_height = 322.73,# CG height from the wheel centre
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