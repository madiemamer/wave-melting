import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from waveErosion_classV3 import waveErosion
import matplotlib as mpl
import cmocean.cm as cm

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.25
mpl.rcParams['legend.fontsize'] = 14

size = 25

d_ocean = 500
d = 250
d_ice = 100
B = 1
dz = 1
Nz = int(d/dz)
ice_idx = -int(d_ice/dz)
z = np.linspace(-d, 0, Nz)
domain_x = np.array([d_ocean, d, d_ice, dz, B])

wave_H = 2
lamb = 100
theta = 0 
Tbg = 2

def beaufort_scale(wave_H):
        if wave_H < 0.1:
            Ss = 0  # Calm
        elif wave_H < 0.2:
            Ss = 1  # Light air
        elif wave_H < 0.3:
            Ss = 2  # Light breeze
        elif wave_H < 1:
            Ss = 3  # Gentle breeze
        elif wave_H < 1.5:
            Ss = 4  # Moderate breeze
        elif wave_H < 2.5:
            Ss = 5  # Fresh breeze
        elif wave_H < 4:
            Ss = 6  # Strong breeze
        elif wave_H < 5.5:
            Ss = 7  # Near gale
        elif wave_H < 7.5:
            Ss = 8  # Gale
        elif wave_H < 10:
            Ss = 9  # Strong gale
        elif wave_H < 12.5:
            Ss = 10  # Storm
        else:
            Ss = np.nan()
        return Ss


directory = '/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_HSensitivity'

wave_Hs = np.linspace(0.01, 7, 100)

m_here = np.zeros((len(wave_Hs)))
Res = np.zeros((len(wave_Hs)))
m_crawfords = np.zeros((len(wave_Hs)))
m_silvas = np.zeros((len(wave_Hs)))

m1s = np.zeros((len(wave_Hs)))
m2s = np.zeros((len(wave_Hs)))
m3s = np.zeros((len(wave_Hs)))
m4s = np.zeros((len(wave_Hs)))

m_crawfords_surf = np.zeros((len(wave_Hs)))
m_silvas_surf = np.zeros((len(wave_Hs)))

m1s_surf = np.zeros((len(wave_Hs)))
m2s_surf = np.zeros((len(wave_Hs)))
m3s_surf = np.zeros((len(wave_Hs)))
m4s_surf = np.zeros((len(wave_Hs)))

i = 0
j = 0

for wave_H in wave_Hs:

    print(f"Plotting for Wave Height {wave_H}m.")
    dir = directory + f"/waveH_{wave_H}"
    
    wave_x = np.array([wave_H, lamb, theta])

    nu = 1.9 * 10**(-6) # kinemtic viscosity [m^2/s]
    g = 9.81 # gravity m/s^2
    k = np.pi * 2 / lamb # wave number [m^-1]
    P = (( lamb * 2 * np.pi /( g * math.tanh(k * d_ocean)))**(1/2)) # wave period [s^-1]
    a = 0.5 * wave_H # wave amplitude [m]

    u = (np.pi * wave_H / P) * math.cos(theta) * np.cosh(k * (z + d_ocean)) / np.sinh(k * d_ocean) # wave horizontal orbital velocity [m/s]
    omega = (np.pi * wave_H / P) * (1+math.sin(theta)**2)**(1/2) * np.sinh(k * (z+d_ocean)) / np.sinh(k * d_ocean) # wave vertical orbital velocity [m/s]

    V = (u**2 + omega**2)**(1/2) # combined wave orbital velocity [m/s]
    Re_a = a * V / nu # wave abcissa reynolds number
    # Cf = 2 * (Re_a**(-0.5)) # wave skin friction - laminar version 
    Cf = 0.09 * Re_a**(-0.2) #

    
    ## Get values for White and Wagner
    Re_z = wave_H**2/(P * nu) * np.exp(-2 * k * z)

    # Set Ss based on Beaufort scale thresholds
    Ss = beaufort_scale(wave_H)

    m1 = np.loadtxt(dir + '/m1.csv', delimiter = ',') 
    Tb1 = np.loadtxt(dir + '/Tb1.csv', delimiter = ',')
    
    ##---------------------------------------------------##
    m2 = np.loadtxt(dir + '/m2.csv', delimiter = ',') 
    Tb2 = np.loadtxt(dir + '/Tb2.csv', delimiter = ',')
    Tn2 = np.loadtxt(dir + '/Tn2.csv', delimiter = ',') 

    ##---------------------------------------------------##
    m3 = np.loadtxt(dir + '/m3.csv', delimiter = ',') 
    Tb3 = np.loadtxt(dir + '/Tb4.csv', delimiter = ',')
    Tn3 = np.loadtxt(dir + '/Tn4.csv', delimiter = ',')

    ##---------------------------------------------------##
    m4 = np.loadtxt(dir + '/m4.csv', delimiter = ',')
    Tb4 = np.loadtxt(dir + '/Tb4.csv', delimiter = ',')
    Tn4 = np.loadtxt(dir + '/Tn4.csv', delimiter = ',') 

    m_crawford = np.loadtxt(dir + '/m_crawford.csv', delimiter = ',')
    Tm = np.loadtxt(dir + '/Tm_crawford.csv', delimiter = ',') 
    m_silva = np.loadtxt(dir + '/m_silva.csv', delimiter = ',')

    Tf1 = Tbg - Tb1
    Tf2 = Tn2 - Tb2
    Tf3 = Tn3 - Tb3
    Tf4 = Tn4 - Tb4
    Tf_crawford = Tbg - Tm
    Tf_silva = Tbg + np.ones(len(Tb1[-1,:])) * 2

    m_crawfords[j] = np.mean(m_crawford[ice_idx:] * P/(wave_H * Tf_crawford[ice_idx:]))
    m_silvas[j] = np.mean(m_silva[ice_idx:]/(24 * 3600)* P/(wave_H * Tf_silva[ice_idx:]))

    m1s[j] = np.mean(m1[-1,ice_idx:] * P/(wave_H * Tf1[-1,ice_idx:]))
    m2s[j] = np.mean(m2[-1,ice_idx:] * P/(wave_H * Tf2[-1,ice_idx:]))
    m3s[j] = np.mean(m3[-1,ice_idx:] * P/(wave_H * Tf3[-1,ice_idx:]))
    m4s[j] = np.mean(m4[-1,ice_idx:] * P/(wave_H * Tf4[-1,ice_idx:]))

    m_crawfords_surf[j] = (m_crawford[-1]) * P/(wave_H * Tf_crawford[-1])
    m_silvas_surf[j] = (m_silva[-1])/(24 * 3600)* P/(wave_H * Tf_silva[-1])

    m1s_surf[j] = (m1[-1,-1]) * P/(wave_H * Tf1[-1,-1])
    m2s_surf[j] = (m2[-1,-1]) * P/(wave_H * Tf2[-1,-1])
    m3s_surf[j] = (m3[-1,-1]) * P/(wave_H * Tf3[-1,-1])
    m4s_surf[j] = (m4[-1,-1]) * P/(wave_H * Tf4[-1,-1])

    Res[j] = Re_z[-1]
    
    j = j + 1


# Define the exponential model: y = a * exp(b * x)
def exp_func(x, a, b):
    return a * x **b

def func_fit(x, y):
    # Fit the curve
    popt, pcov = curve_fit(exp_func, x, y)

    # Extract parameters
    a_fit, b_fit = popt
    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}")

    return a_fit, b_fit

cmap = cm.haline(np.linspace(0, 1, 6))

fig, ax = plt.subplots(1,2, figsize = (15,5))

a_silva, b_silva = func_fit(Res[:], m_silvas[:])
ax[0].scatter(Res[::5], m_silvas[::5], color = 'darkgrey', marker = 'D', s = size, zorder = 2)
ax[0].plot(Res[:], exp_func(Res[:], a_silva, b_silva), color = 'darkgrey', linestyle = '--', zorder = 1)

a_crawford, b_crawford = func_fit(Res[:], m_crawfords[:])
ax[0].scatter(Res[::5], m_crawfords[::5], color = 'darkgrey', s = size,  zorder = 2)
ax[0].plot(Res[:], exp_func(Res[:], a_crawford, b_crawford), color = 'darkgrey', linestyle = '-', zorder = 1)

# ax[0].plot(Res[:], exp_func(Res[:], 0.00015, -0.12), color = 'k')

a_1, b_1 = func_fit(Res[:], m1s[:])
ax[0].scatter(Res[::5], m1s[::5], s = size, color = cmap[0], zorder = 2)
ax[0].plot(Res[:], exp_func(Res[:], a_1, b_1), color = cmap[0], linestyle = '-', zorder = 1)

a_2, b_2 = func_fit(Res[:], m2s[:])
ax[0].scatter(Res[::5], m2s[::5], s = size, color = cmap[1], zorder = 2)
ax[0].plot(Res[:], exp_func(Res[:], a_2, b_2), color = cmap[1], linestyle = '-', zorder = 1)

a_3, b_3 = func_fit(Res[:], m3s[:])
ax[0].scatter(Res[::5], m3s[::5], s = size, color = cmap[2], zorder = 2)
ax[0].plot(Res[:], exp_func(Res[:], a_3, b_3), color = cmap[2], linestyle = '-', zorder = 1)

a_4, b_4 = func_fit(Res[:], m4s[:])
ax[0].scatter(Res[::5], m4s[::5], s = size, color = cmap[4], zorder = 2)
ax[0].plot(Res[:], exp_func(Res[:], a_4, b_4), color = cmap[4], linestyle = '-', zorder = 1)

ax[0].set_xlabel('Re$_H$')
ax[0].set_ylabel('Depth-Averaged Scaled Melt Rate')

ax[0].set_xlim(0,3e6)
ax[0].set_ylim(0,4e-5)
ax[0].grid(alpha = 0.25)

###

# Extract parameters

a_silva, b_silva = func_fit(Res[:], m_silvas_surf[:])
sScat = ax[1].scatter(Res[::5], m_silvas_surf[::5], color = 'darkgrey', marker = 'D', s = size, zorder = 2)
s, = ax[1].plot(Res[:], exp_func(Res[:], a_silva, b_silva),  color = 'darkgrey', linestyle = '--', zorder = 1)

a_crawford, b_crawford = func_fit(Res[:], m_crawfords_surf[:])
wScat = ax[1].scatter(Res[::5], m_crawfords_surf[::5],  color = 'darkgrey', s = size, zorder = 2)
# c, = ax[1].plot(Res[:], exp_func(Res[:], a_wagner, b_wagner), color = 'darkgrey', linestyle = '--')

w, = ax[1].plot(Res[:], exp_func(Res[:], 0.00015, -0.12), color = 'k', zorder = 1)

a_1, b_1 = func_fit(Res[:], m1s_surf[:])
m1_scat = ax[1].scatter(Res[::5], m1s_surf[::5], color = cmap[0], s = size, zorder = 2)
mp1, = ax[1].plot(Res[:], exp_func(Res[:], a_1, b_1), color = cmap[0], linestyle = '-', zorder = 1)

a_2, b_2 = func_fit(Res[:], m2s_surf[:])
m2_scat = ax[1].scatter(Res[::5], m2s_surf[::5], color = cmap[1], s = size, zorder = 2)
mp2, = ax[1].plot(Res[:], exp_func(Res[:], a_2, b_2), color = cmap[1], linestyle = '-', zorder = 1)

a_3, b_3 = func_fit(Res[:], m3s_surf[:])
m3_scat = ax[1].scatter(Res[::5], m3s_surf[::5], color = cmap[2], s = size, zorder = 2)
mp3, = ax[1].plot(Res[:], exp_func(Res[:], a_3, b_3), color = cmap[2], linestyle = '-', zorder = 1)

a_4, b_4 = func_fit(Res[:], m4s_surf[:])
m4_scat = ax[1].scatter(Res[::5], m4s_surf[::5], color = cmap[4], s = size, zorder = 2)
mp4, = ax[1].plot(Res[:], exp_func(Res[:], a_4, b_4), color = cmap[4], linestyle = '-', zorder = 1)

ax[1].set_xlabel('Re$_H$')
ax[1].set_ylabel('Waterline Scaled Melt Rate')

ax[1].set_xlim(0,3e6)
ax[1].set_ylim(0,0.00008)
ax[1].grid(alpha = 0.25)

plt.tight_layout()

# Get lowercase letters a-i
letters = ['a', 'b', 'c']

# Loop over each subplot and add the letter in the lower-left corner
for i, axis in enumerate(ax.flat):
    axis.text(0.01, 0.92, letters[i], transform=axis.transAxes,
            fontsize=16, fontweight = 'bold',  verticalalignment='bottom', horizontalalignment='left')

ax[0].legend([w,sScat,wScat,m1_scat,m2_scat,m3_scat,m4_scat],['White et. al (1980)', 'S06', 'C24', 'BM', 'OFM', 'PEM', 'HTM'])
# ax[0].text(0.2e6,1.63e-5,'A', fontsize = 14)
# ax[1].text(0.2e6,1.7e-4,'B', fontsize = 14)
plt.savefig('/storage/home/hcoda1/1/mmamer3/scratch/waveProj/powerLaw_fit.jpg', dpi = 300)
