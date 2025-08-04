import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
from scipy.optimize import curve_fit
import matplotlib as mpl


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.25
mpl.rcParams['legend.fontsize'] = 14

def exp_func(x, a, b):
    return a * x **b

def func_fit(x, y):
    # Fit the curve
    popt, pcov = curve_fit(exp_func, x, y)

    # Extract parameters
    a_fit, b_fit = popt
    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}")

    return a_fit, b_fit

t = np.arange(0,20000,100)
size = 15
cmap = cm.haline(np.linspace(0, 1, 6))
d_ocean = 500
d = 250
d_ice = 100
B = 1
dz = 1
Nz = int(d/dz)
z = np.linspace(-d,0,Nz)

ice_idx = -int(d_ice/dz)
Tbg = 2

fig,ax = plt.subplots(1,3, figsize = (15,5))

ax[0].set_ylabel('Cooling Fraction')
ax[0].set_xlabel('Depth Averaged Melt Rate [md$^{-1}$]')
ax[0].grid(alpha = 0.25)

ax[1].set_ylabel('Freshening Fraction')
ax[1].set_xlabel('Depth Averaged Melt Rate [md$^{-1}$]')
ax[1].grid(alpha = 0.25)

ax[2].set_ylabel('Insulating Fraction')
ax[2].set_xlabel('Depth Averaged Melt Rate [md$^{-1}$]')
ax[2].grid(alpha = 0.25)


figureName = '/insulating_fraction'

i = 0

Tbgs = np.linspace(-1.8, 5, 100)
lambs = np.linspace(5, 100, 100)
wave_Hs = np.linspace(0.01, 7, 100)

directories = ['/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_HSensitivity',
               '/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_TbgSensitivity']
styles = ['-',
          '-.']

vars = [wave_Hs, Tbgs]
varNames = ['waveH', 'Tbg']
j = 0

for directory in directories:
    
    var = vars[j]
    varName = varNames[j]
    
    n = int(len(var))

    ms = np.zeros(n)
    Ss = np.zeros(n)
    Ts = np.zeros(n)

    ms3_relDiff = np.zeros(n)
    Ss3_relDiff = np.zeros(n)
    Ts3_relDiff = np.zeros(n)

    ms4_relDiff = np.zeros(n)
    Ss4_relDiff = np.zeros(n)
    Ts4_relDiff = np.zeros(n)

    i = 0
    for each in var:

        print(f"Plotting for {varName} {each}.")
        dir = directory + f"/{varName}_{each}"

        m1 = np.loadtxt(dir + '/m1.csv', delimiter = ',') * 24 * 3600
        Tb1 = np.loadtxt(dir + '/Tb1.csv', delimiter = ',')
        Sb1 = np.loadtxt(dir + '/Sb1.csv', delimiter = ',')

        ##---------------------------------------------------##
        m2 = np.loadtxt(dir + '/m2.csv', delimiter = ',') * 24 * 3600
        Tb2 = np.loadtxt(dir + '/Tb2.csv', delimiter = ',')
        Sb2 = np.loadtxt(dir + '/Sb2.csv',  delimiter = ',')
        Tn2 = np.loadtxt(dir + '/Tn2.csv', delimiter = ',')
        Sn2 = np.loadtxt(dir + '/Sn2.csv',  delimiter = ',')

        ##---------------------------------------------------##
        m3 = np.loadtxt(dir + '/m3.csv', delimiter = ',') * 24 * 3600
        Tb3 = np.loadtxt(dir + '/Tb3.csv', delimiter = ',')
        Sb3 = np.loadtxt(dir + '/Sb3.csv', delimiter = ',')
        Tn3 = np.loadtxt(dir + '/Tn3.csv', delimiter = ',')
        Sn3 = np.loadtxt(dir + '/Sn3.csv', delimiter = ',')

        ##---------------------------------------------------##
        m4 = np.loadtxt(dir + '/m4.csv', delimiter = ',') * 24 * 3600
        Tb4 = np.loadtxt(dir + '/Tb4.csv', delimiter = ',')
        Sb4 = np.loadtxt(dir + '/Sb4.csv', delimiter = ',')
        Tn4 = np.loadtxt(dir + '/Tn4.csv', delimiter = ',')
        Sn4 = np.loadtxt(dir + '/Sn4.csv', delimiter = ',')


        ms[i] = np.nanmean(m2[-2,ice_idx:])
        tf2 = (Tn2[-1,:] - Tb2[-1,:])
        tf3 = (Tn3[-1,:] - Tb3[-1,:])
        tf4 = (Tn4[-1,:] - Tb4[-1,:])

        Ts3_relDiff[i] = (tf2[-1] - tf3[-1])/tf2[-1]
        Ts4_relDiff[i] = (tf2[-1] - tf4[-1])/tf2[-1] 

        Ss3_relDiff[i] = (Sn2[-1,-1] - Sn3[-1,-1])/Sn2[-1,-1] 
        Ss4_relDiff[i] = (Sn2[-1,-1] - Sn4[-1,-1])/Sn2[-1,-1] 

        ms3_relDiff[i] = (m2[-1,-1] - m3[-1,-1])/m2[-1,-1] 
        ms4_relDiff[i] = (m2[-1,-1] - m4[-1,-1])/m2[-1,-1] 

        i = i + 1

    ax[0].plot(ms[:], Ts3_relDiff[:], color = cmap[2], linestyle = styles[j], zorder = 2)
    ax[0].plot(ms[:], Ts4_relDiff[:], color = cmap[4], linestyle = styles[j], zorder = 2)

    ax[1].plot(ms[:], Ss3_relDiff[:], color = cmap[2], linestyle = styles[j], zorder = 2)
    ax[1].plot(ms[:], Ss4_relDiff[:], color = cmap[4], linestyle = styles[j], zorder = 2)

    ax[2].plot(ms[:], ms3_relDiff[:], color = cmap[2], linestyle = styles[j], zorder = 2)
    ax[2].plot(ms[:], ms4_relDiff[:], color = cmap[4], linestyle = styles[j], zorder = 2)


#    a3, b3 = func_fit(ms[:], Ts3_relDiff[:])
#    ax[0].plot(ms[:], exp_func(ms[:], a3, b3), color = cmap[2], linestyle = '-', zorder = 1)

#    a4, b4 = func_fit(ms[:], Ts4_relDiff[:])
#    ax[0].plot(ms[:], exp_func(ms[:], a4, b4), color = cmap[4], linestyle = '-', zorder = 1)

#    a3, b3 = func_fit(ms[:], Ss3_relDiff[:])
#    ax[1].plot(ms[:], exp_func(ms[:], a3, b3), color = cmap[2], linestyle = '-', zorder = 1)

#    a4, b4 = func_fit(ms[:], Ss4_relDiff[:])
#    ax[1].plot(ms[:], exp_func(ms[:], a4, b4), color = cmap[4], linestyle = '-', zorder = 1)

#    a3, b3 = func_fit(ms[:], ms3_relDiff[:])
#    pem, = ax[2].plot(ms[:], exp_func(ms[:], a3, b3), color = cmap[2], linestyle = '-', zorder = 1)

#    a4, b4 = func_fit(ms[:], ms4_relDiff[:])
#    htm, = ax[2].plot(ms[:], exp_func(ms[:], a4, b4), color = cmap[4], linestyle = '-', zorder = 1)


    j = j + 1



ax[0].set_xlim(0,1)
ax[1].set_xlim(0,1)
ax[2].set_xlim(0,1)
ax[2].legend(['PEM, $H_w$ = 0.01-7m', 'HTM, $H_w$ = 0.01-7m', 'PEM, $\Delta T$ = -1.8-5$^{\circ}$C', 'HTM, $\Delta T$ = -1.8-5$^{\circ}$C'])

letters = ['a','b','c','d','e','f','g','h','i','j']


# Loop over each subplot and add the letter in the lower-left corner
for i, axis in enumerate(ax.flat):
    axis.text(0.01, 0.92, letters[i], transform=axis.transAxes,
            fontsize=16, fontweight = 'bold',  verticalalignment='bottom', horizontalalignment='left')
    
fig.tight_layout()
fig.savefig('/storage/home/hcoda1/1/mmamer3/scratch/waveProj/' + figureName + '.jpg', dpi = 300)
