## Wave Length Sensitivity Test ##

import numpy as np
from waveErosion_classV3 import waveErosion
import os

directory = '/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_LambSensitivity'

if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Created directory: {directory}")
else:
    print(f"Directory already exists.")

def beaufort_scale(wave_H):
        if wave_H < 0.1:
            Ss = 0  # Calm
        elif wave_H < 0.5:
            Ss = 1  # Light air
        elif wave_H < 1.25:
            Ss = 2  # Light breeze
        elif wave_H < 2.5:
            Ss = 3  # Gentle breeze
        elif wave_H < 4.0:
            Ss = 4  # Moderate breeze
        elif wave_H < 6.0:
            Ss = 5  # Fresh breeze
        elif wave_H < 9.0:
            Ss = 6  # Strong breeze
        elif wave_H < 12.5:
            Ss = 7  # Near gale
        elif wave_H < 16.0:
            Ss = 8  # Gale
        elif wave_H < 20.0:
            Ss = 9  # Strong gale
        elif wave_H < 25.0:
            Ss = 10  # Storm
        elif wave_H < 30.0:
            Ss = 11  # Violent storm
        else:
            Ss = 12  # Hurricane
        return Ss

d_ocean = 500
d = 250
d_ice = 100
B = 1
dz = 1
Nz = int(d/dz)
ice_idx = -int(d_ice/dz)
z = np.linspace(-d, 0, Nz)
domain_x = np.array([d_ocean, d, d_ice, dz, B])

modelType = 1
meltType = 0
printCheck = 0
model_x = np.array([modelType, meltType, printCheck])

startTime = 0
endTime = 20000
dt = 1
checkFreq = 100
iterativeType = 0 
iterative_x = np.array([startTime, endTime, dt, checkFreq, iterativeType])

Ubg = np.ones(Nz) * 0.05 # Standard value
Tbg = np.ones(Nz) * 2
Sbg = np.ones(Nz) * 34

background_x = np.zeros((3, Nz))
background_x[0,:] = Ubg
background_x[1,:] = Tbg
background_x[2,:] = Sbg

m_init = np.ones(Nz) * 5e-5
init_Sb = Sbg
init_Tb = Tbg
    
init_b = np.ones(Nz) * 5
init_Sn = Sbg
init_Us = 0.01
init_Up = 0.01
init_Tn = Tbg

init_B = np.ones(Nz) * B

init_Ti = Tbg
init_Si = Sbg
init_Uif = 0.01
init_Uin = 0.01

initial_x = np.zeros((13, Nz))

initial_x[0,:] = m_init
initial_x[2,:] = init_Sb
initial_x[1,:] = init_Tb
    
initial_x[3,:] = init_b
initial_x[5,:] = init_Sn
initial_x[4,:] = init_Tn
    
initial_x[6,:] = init_Us
initial_x[7,:] = init_Up

initial_x[8,:] = init_B
initial_x[9,:] = init_Ti
initial_x[10,:] = init_Si
initial_x[11,:] = init_Uif
initial_x[12,:] = init_Uin


lambs = np.linspace(5, 100, 100)

for each in lambs:

    dir = directory + f"/lamb_{each}"

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Created directory: {dir}")
    else:
        print(f"Directory already exists.")

    wave_H = 4
    lamb = each
    theta = 0 
    wave_x = np.array([wave_H, lamb, theta])

    modelType = 1
    model_x = np.array([modelType, meltType, printCheck])
    base = waveErosion(model_x, wave_x, domain_x, background_x, iterative_x, initial_x)
    m1, Tb1, Sb1 = waveErosion.getModelOutput(base)
    t = np.linspace(0, endTime, len(m1[:,-1]))

    np.savetxt(dir + '/m1.csv', m1, delimiter = ',')
    np.savetxt(dir + '/Tb1.csv', Tb1, delimiter = ',')
    np.savetxt(dir + '/Sb1.csv', Sb1, delimiter = ',')

    # ##---------------------------------------------------##
    modelType = 2
    model_x = np.array([modelType, meltType, printCheck])
    ocean = waveErosion(model_x, wave_x, domain_x, background_x, iterative_x, initial_x)
    m2, Tb2, Sb2, b2, Tn2, Sn2, rho_n2, Us2 = waveErosion.getModelOutput(ocean)
    t = np.linspace(0, endTime, len(m2[:,-1]))

    np.savetxt(dir + '/m2.csv', m2, delimiter = ',')
    np.savetxt(dir + '/Tb2.csv', Tb2, delimiter = ',')
    np.savetxt(dir + '/Sb2.csv', Sb2, delimiter = ',')

    np.savetxt(dir + '/b2.csv', b2, delimiter = ',')
    np.savetxt(dir + '/Tn2.csv', Tn2, delimiter = ',')
    np.savetxt(dir + '/Sn2.csv', Sn2, delimiter = ',')
    np.savetxt(dir + '/rho_n2.csv', rho_n2, delimiter = ',')
    np.savetxt(dir + '/Us2.csv', Us2, delimiter = ',')

    ##---------------------------------------------------##
    modelType = 3
    model_x = np.array([modelType, meltType, printCheck])
    plume = waveErosion(model_x, wave_x, domain_x, background_x, iterative_x, initial_x)
    m3, Tb3, Sb3, b3, Tn3, Sn3, rho_n3, Us3, Up3 = waveErosion.getModelOutput(plume)
    t = np.linspace(0, endTime, len(m3[:,-1]))

    np.savetxt(dir + '/m3.csv', m3, delimiter = ',')
    np.savetxt(dir + '/Tb3.csv', Tb3, delimiter = ',')
    np.savetxt(dir + '/Sb3.csv', Sb3, delimiter = ',')

    np.savetxt(dir + '/b3.csv', b3, delimiter = ',')
    np.savetxt(dir + '/Tn3.csv', Tn3, delimiter = ',')
    np.savetxt(dir + '/Sn3.csv', Sn3, delimiter = ',')
    np.savetxt(dir + '/rho_n3.csv', rho_n3, delimiter = ',')
    np.savetxt(dir + '/Us3.csv', Us3, delimiter = ',')
    np.savetxt(dir + '/Up3.csv', Up3, delimiter = ',')

    ##---------------------------------------------------##
    modelType = 4
    model_x = np.array([modelType, meltType, printCheck])
    horizontal = waveErosion(model_x, wave_x, domain_x, background_x, iterative_x, initial_x)
    m4, Tb4, Sb4, b4, Tn4, Sn4, rho_n4, Us4, Up4, B4, Ti4, Si4, rho_i4, Uif4, Uin4 = waveErosion.getModelOutput(horizontal)
    t = np.linspace(0, endTime, len(m4[:,-1]))

    
    np.savetxt(dir + '/m4.csv', m4, delimiter = ',')
    np.savetxt(dir + '/Tb4.csv', Tb4, delimiter = ',')
    np.savetxt(dir + '/Sb4.csv', Sb4, delimiter = ',')

    np.savetxt(dir + '/b4.csv', b4, delimiter = ',')
    np.savetxt(dir + '/Tn4.csv', Tn4, delimiter = ',')
    np.savetxt(dir + '/Sn4.csv', Sn4, delimiter = ',')
    np.savetxt(dir + '/rho_n4.csv', rho_n4, delimiter = ',')
    np.savetxt(dir + '/Us4.csv', Us4, delimiter = ',')
    np.savetxt(dir + '/Up4.csv', Up4, delimiter = ',')

    np.savetxt(dir + '/B4.csv', B4, delimiter = ',')
    np.savetxt(dir + '/Ti4.csv', Ti4, delimiter = ',')
    np.savetxt(dir + '/Si4.csv', Si4, delimiter = ',')
    np.savetxt(dir + '/rho_i4.csv', rho_i4, delimiter = ',')
    np.savetxt(dir + '/Uif4.csv', Uif4, delimiter = ',')
    np.savetxt(dir + '/Uin4.csv', Uin4, delimiter = ',')

    ##---------------------------------------------------##
    Ss = beaufort_scale(wave_H)

    crawford_mB = waveErosion.crawford2024_v0(base) ## with Tbg - Tm instead of Tb
    silva_mB = waveErosion.silva2006_v0(base, Ss, 0.5) ## with + 2
    Tm = (-1.80) * np.exp(-0.19 * (Tbg + 1.80))

    np.savetxt(dir + '/m_crawford.csv', crawford_mB, delimiter = ',')
    np.savetxt(dir + '/m_silva.csv', silva_mB, delimiter = ',')
    np.savetxt(dir + '/Tm_crawford.csv', Tm, delimiter = ',')

