import numpy as np
from VehicleModel4Wh import Vehicle
from Instances.Quad.base import Golf, MRF13570R1269S
currveh = Golf()
currtire = MRF13570R1269S()
instance = Vehicle(
    np.array([1082.18, 237.44, 1465.55]), #r_A -  Top Strut Mount
    np.array([934.31, 307.28, 1027.78]), #np.array([918.02+3, 339.28, 1076.53]) + np.array([-12,-50,-20]) + np.array([+15.5,0,0])+ np.array([10,28,-40]), # Tierod Outer Ball Joint
    np.array([1094.40, 20.67, 1103.56]), # Tierod Inner Ball Joint
    np.array([1008.54, 500.00-50, 971.05]), # Wheel Center
    np.array([1008.54,	339.63,	910.38]), # Lower Ball Joint
    r_La = np.array([907.19, 67.47, 943.44]), # Lower A-Arm Bush Front
    r_Lb = np.array([1008.54, 67.47, 943.44]), # Lower A-Arm Bush Rear
    r_strut = np.array([1022.25, 291.87,1035.45]), # Lower Strut Mount
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