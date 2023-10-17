from CP import crystal
from driver import deform

# --------------- input ----------------------------------------
grain_props_shear={
  "el": [1500.0,0.0,0.3],               # E|C11, 0|C12, v|C44
  "hard": [0.0,1.0e-7,1.0e-7,1.0,1.0],  # b, rho_0, rho_inf, gamma_inf, tau_0
  "euler": [0.62832,0.31416,-1.57080]   # Z1,X,Z2
}

deform_props_shear={
  "F": [0.0,0.02,0.0,0.0],              # F_11, F_12, F_21, F_22
  "no_inc": 100
}

# -------------------------------
grain_props_hardening={
  "el": [72000.0,0.0,0.3],
  "hard": [2.86e-7,1.0e7,1.0e9,0.4,18.0],
  "euler": [0.1,0.2,0.3]
}

deform_props_hardening={
  "F": [0.0,0.2,0.0,0.0],
  "no_inc": 100
}

# --------------------------------------------------------------

grain=crystal(grain_props_shear)
shear_small=deform(grain,deform_props_shear)
shear_small.run()
shear_small.plot()

grain=crystal(grain_props_hardening)
shear_hard=deform(grain,deform_props_hardening)
shear_hard.run()
shear_hard.plot()
