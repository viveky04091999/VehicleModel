import numpy as np
from VehicleModel4Wh import Vehicle
from Instances.Quad.base import Golf, MRF13570R1269S
currveh = Golf()
currtire = MRF13570R1269S()
instance = Vehicle(
    np.array([983.185, 226.27, 1487.214]), #r_A -  Top of the Kingpin
    np.array([845.5, 309.5, 1100.214]), # Tierod Outer Ball Joint
    np.array([984.4, 20.6, 1103.56]), # Tierod Inner Ball Joint
    np.array([898.75, 487.5, 972.02469]), # Wheel Center
    np.array([898.75, 367.138, 911.35]), # Bottom of the Kingpin
    r_La = np.array([798.75, 104.97, 940.96]), # Lower A-Arm Bush Front
    r_Lb = np.array([998.75, 104.97, 940.96]), # Lower A-Arm Bush Rear
    r_strut= np.array([911.3, 320.3444, 1055.75]), # Strut Top Mount
    GVW = 369,
    b = 581.59,
    CG_height = 233.87,
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