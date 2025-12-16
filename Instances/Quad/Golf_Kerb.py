import numpy as np
from VehicleModel4Wh import Vehicle
from Instances.Quad.base import Golf, MRF13570R1269S
currveh = Golf()
currtire = MRF13570R1269S()
instance = Vehicle(
    np.array([1014.24, 376.91, 989.64]), #r_A -  Top of the Kingpin
    np.array([1073.57+18, 414.32-104, 1062.40]), # Tierod Outer Ball Joint
    np.array([922.74, 16.50, 1097.83+25]), # Tierod Inner Ball Joint
    np.array([1008.76, 500.00, 961.43]), # Wheel Center
    np.array([1002.90, 385.26, 931.31]), # Bottom of the Kingpin
    r_La = np.array([948.76, 130.00, 951.43]), # Lower A-Arm Bush Front
    r_Lb = np.array([1100.76, 130.00, 951.43]), # Lower A-Arm Bush Rear
    r_Ua = np.array([948.76, 100.00, 1027.43]), # Upper A-Arm Bush Front
    r_Ub = np.array([1100.76, 100.00, 1027.43]), # Upper A-Arm Bush Rear
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