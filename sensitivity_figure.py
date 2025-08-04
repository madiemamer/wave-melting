import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cmocean.cm as cm
import matplotlib as mpl

def exp_func(x, a, b):
    return a * x **b

def func_fit(x, y):
    # Fit the curve
    popt, pcov = curve_fit(exp_func, x, y)

    # Extract parameters
    a_fit, b_fit = popt
    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}")

    return a_fit, b_fit

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.25
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.labelweight'] = 'regular'

t = np.arange(0,20000,100)
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
lamb = 100
waveH = 4

fig,ax = plt.subplots(3,3, figsize = (15,15))

axis_labels = [
               'Wave Height [m]', 
               'Wave Length [m]', 
               'Far Field Temperature [C]'
               ]

directories = [
               '/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_HSensitivity',
               '/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_LambSensitivity',
               '/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_TbgSensitivity'
               ]
varNames = [
               'waveH',
               'lamb',
               'Tbg'
            ]
figureNames = [
               '/wave_HSensitivity',
               '/wave_LambSensitivity',
               '/wave_TbgSensitivity'
               ]

wave_Hs = np.linspace(0.01, 7, 100)
lambs = np.linspace(5, 100, 100)
Tbgs = np.linspace(-1.8, 5, 100)

i = 0
for var in [wave_Hs, lambs, Tbgs]:

    varName = varNames[i]
    print(f'Now Plotting for {varName}')
    ##---------------------------------------------------##
    fig1, ax1 = plt.subplots(4,3, figsize = (15,20))
    fig2, ax2 = plt.subplots(4,3, figsize = (15,20))
    ##---------------------------------------------------##

    ax[i,0].set_ylabel('Depth-Averaged Thermal Forcing [C]')
    ax[i,0].set_xlabel(axis_labels[i])
    ax[i,0].grid(alpha = 0.25)

    ax[i,1].set_ylabel('Depth-Averaged Salinity [ppt]')
    ax[i,1].set_xlabel(axis_labels[i])
    ax[i,1].grid(alpha = 0.25)

    ax[i,2].set_ylabel('Depth-Averaged Melt Rate [md$^{-1}$]')
    ax[i,2].set_xlabel(axis_labels[i])
    ax[i,2].grid(alpha = 0.25)

    ##---------------------------------------------------##
    ax1[3,0].set_xlabel('Thermal Forcing [C]')
    ax1[0,0].set_ylabel('Depth [m]')

    ax1[3,1].set_xlabel('Salinity [ppt]')
    ax1[1,0].set_ylabel('Depth [m]')

    ax1[3,2].set_ylabel('Melt Rate [md$^{-1}$]')
    ax1[2,0].set_xlabel('Depth [m]')

    ##---------------------------------------------------##

    ax2[0,0].set_ylabel('Thermal Forcing [C]')
    ax2[1,0].set_ylabel('Thermal Forcing [C]')
    ax2[2,0].set_ylabel('Thermal Forcing [C]')
    ax2[3,0].set_ylabel('Thermal Forcing [C]')

    ax2[0,1].set_ylabel('Salinity [ppt]')
    ax2[1,1].set_ylabel('Salinity [ppt]')
    ax2[2,1].set_ylabel('Salinity [ppt]')
    ax2[3,1].set_ylabel('Salinity [ppt]')

    ax2[0,2].set_ylabel('Melt Rate [md$^{-1}$]')
    ax2[2,2].set_ylabel('Melt Rate [md$^{-1}$]')
    ax2[3,2].set_ylabel('Melt Rate [md$^{-1}$]')
    ax2[3,2].set_ylabel('Melt Rate [md$^{-1}$]')

    ax2[3,0].set_xlabel('Time [s]')
    ax2[3,1].set_xlabel('Time [s]')
    ax2[3,2].set_xlabel('Time [s]')


    ## Plotting Figure ##

    directory = directories[i]

    figureName = figureNames[i]

    alphas = np.linspace(0.2,1,len(var))
    tf1 = np.zeros(len(var))
    tf2 = np.zeros(len(var))
    tf3 = np.zeros(len(var))
    tf4 = np.zeros(len(var))
    tf_silva = np.zeros(len(var))
    tf_crawford = np.zeros(len(var))

    s1 = np.zeros(len(var))
    s2 = np.zeros(len(var))
    s3 = np.zeros(len(var))
    s4 = np.zeros(len(var))
    sn2 = np.zeros(len(var))
    sn3 = np.zeros(len(var))
    sn4 = np.zeros(len(var))

    ms1 = np.zeros(len(var))
    ms2 = np.zeros(len(var))
    ms3 = np.zeros(len(var))
    ms4 = np.zeros(len(var))
    ms_silva = np.zeros(len(var))
    ms_crawford = np.zeros(len(var))

    j = 0
    for each in var:

        ## Loading Data ##
        print(f"Plotting for {varName} {each}.")
        dir = directory + f"/{varName}_{each}"

        ##---------------------------------------------------##
        m1 = np.loadtxt(dir + '/m1.csv', delimiter = ',')
        Tb1 = np.loadtxt(dir + '/Tb1.csv', delimiter = ',')
        Sb1 = np.loadtxt(dir + '/Sb1.csv', delimiter = ',')

        ##---------------------------------------------------##
        m2 = np.loadtxt(dir + '/m2.csv', delimiter = ',')
        Tb2 = np.loadtxt(dir + '/Tb2.csv', delimiter = ',')
        Sb2 = np.loadtxt(dir + '/Sb2.csv',  delimiter = ',')

        Tn2 = np.loadtxt(dir + '/Tn2.csv', delimiter = ',')
        Sn2 = np.loadtxt(dir + '/Sn2.csv', delimiter = ',')

        ##---------------------------------------------------##
        m3 = np.loadtxt(dir + '/m3.csv', delimiter = ',')
        Tb3 = np.loadtxt(dir + '/Tb3.csv', delimiter = ',')
        Sb3 = np.loadtxt(dir + '/Sb3.csv', delimiter = ',')

        Tn3 = np.loadtxt(dir + '/Tn3.csv', delimiter = ',')
        Sn3 = np.loadtxt(dir + '/Sn3.csv', delimiter = ',')

        ##---------------------------------------------------##
        m4 = np.loadtxt(dir + '/m4.csv', delimiter = ',')
        Tb4 = np.loadtxt(dir + '/Tb4.csv', delimiter = ',')
        Sb4 = np.loadtxt(dir + '/Sb4.csv', delimiter = ',')

        Tn4 = np.loadtxt(dir + '/Tn4.csv', delimiter = ',')
        Sn4 = np.loadtxt(dir + '/Sn4.csv', delimiter = ',')

        ##---------------------------------------------------##

        crawford_m = np.loadtxt(dir + '/m_crawford.csv', delimiter = ',')
        silva_m = np.loadtxt(dir + '/m_silva.csv', delimiter = ',')
        Tm = np.loadtxt(dir + '/Tm_crawford.csv', delimiter = ',')

        ##---------------------------------------------------##
        ##Getting Surface Values##
        if i == 2:
            Tbg = each
        else:
            Tbg = 2

        tf1[j] = np.mean(Tbg - Tb1[-1,ice_idx:])
        tf2[j] = np.mean(Tn2[-1,ice_idx:] - Tb2[-1,ice_idx:])
        tf3[j] = np.mean(Tn3[-1,ice_idx:] - Tb3[-1,ice_idx:])
        tf4[j] = np.mean(Tn4[-1,ice_idx:] - Tb4[-1,ice_idx:])

        tf_silva[j] = np.mean(Tbg + 2)
        tf_crawford[j] = np.mean(Tbg - Tm)

        ##---------------------------------------------------##

        s1[j] = np.mean(Sb1[-1,ice_idx:])
        s2[j] = np.mean(Sb2[-1,ice_idx:])
        s3[j] = np.mean(Sb3[-1,ice_idx:])
        s4[j] = np.mean(Sb4[-1,ice_idx:])

        sn2[j] = np.mean(Sn2[-1,ice_idx:])
        sn3[j] = np.mean(Sn3[-1,ice_idx:])
        sn4[j] = np.mean(Sn4[-1,ice_idx:])

        ##---------------------------------------------------##

        ms1[j] = np.mean(m1[-1,ice_idx:] * 24 * 3600)
        ms2[j] = np.mean(m2[-1,ice_idx:] * 24 * 3600)
        ms3[j] = np.mean(m3[-1,ice_idx:] * 24 * 3600)
        ms4[j] = np.mean(m4[-1,ice_idx:] * 24 * 3600)

        ms_silva[j] = np.mean(silva_m[ice_idx:])
        ms_crawford[j] = np.mean(crawford_m[ice_idx:]* 24 * 3600)

        ##---------------------------------------------------##

        ## Plotting the last time step ##

        ax1[0,0].plot(Tbg - Tb1[-1,:], z, color = cmap[0], alpha = alphas[j])
        ax1[1,0].plot(Tn2[-1,:] - Tb2[-1,:], z, color = cmap[1], alpha = alphas[j])
        ax1[2,0].plot(Tn3[-1,:] - Tb3[-1,:], z, color = cmap[2], alpha = alphas[j])
        ax1[3,0].plot(Tn4[-1,:] - Tb4[-1,:], z, color = cmap[4], alpha = alphas[j])

        ##---------------------------------------------------##

        ax1[0,1].plot(Sb1[-1,:], z, color = cmap[0], alpha = alphas[j])
        ax1[1,1].plot(Sb2[-1,:], z, color = cmap[1], alpha = alphas[j])
        ax1[2,1].plot(Sb3[-1,:], z, color = cmap[2], alpha = alphas[j])
        ax1[3,1].plot(Sb4[-1,:], z, color = cmap[4], alpha = alphas[j])

        ##---------------------------------------------------##

        ax1[0,2].plot(m1[-1,:] * 24 * 3600, z, color = cmap[0], alpha = alphas[i])
        ax1[1,2].plot(m2[-1,:] * 24 * 3600, z, color = cmap[1], alpha = alphas[i])
        ax1[2,2].plot(m3[-1,:] * 24 * 3600, z, color = cmap[2], alpha = alphas[i])
        ax1[3,2].plot(m4[-1,:] * 24 * 3600, z, color = cmap[4], alpha = alphas[i])

        ## Plotting the Surface at all time ##

        ax2[0,0].plot(t, Tbg - Tb1[:,-1], color = cmap[0], alpha = alphas[j])
        ax2[1,0].plot(t, Tn2[:,-1] - Tb2[:,-1], color = cmap[1], alpha = alphas[j])
        ax2[2,0].plot(t, Tn3[:, -1] - Tb3[:, -1], color = cmap[2], alpha = alphas[j])
        ax2[3,0].plot(t, Tn4[:, -1] - Tb4[:, -1], color = cmap[4], alpha = alphas[j])

        ##---------------------------------------------------##

        ax2[0,1].plot(t, Sb1[:, -1], color = cmap[0], alpha = alphas[j])
        ax2[1,1].plot(t, Sb2[:, -1], color = cmap[1], alpha = alphas[j])
        ax2[2,1].plot(t, Sb3[:, -1], color = cmap[2], alpha = alphas[j])
        ax2[3,1].plot(t, Sb4[:, -1], color = cmap[4], alpha = alphas[j])

        ##---------------------------------------------------##

        ax2[0,2].plot(t, m1[:,-1] * 24 * 3600, color = cmap[0], alpha = alphas[j])
        ax2[1,2].plot(t, m2[:,-1] * 24 * 3600, color = cmap[1], alpha = alphas[j])
        ax2[2,2].plot(t, m3[:,-1] * 24 * 3600, color = cmap[2], alpha = alphas[j])
        ax2[3,2].plot(t, m4[:,-1] * 24 * 3600, color = cmap[4], alpha = alphas[j])


        j = j + 1

    ax[i,0].plot(var, tf1 , color = cmap[0], linestyle = '-')
    ax[i,0].plot(var, tf2, color = cmap[1], linestyle = '-')
    ax[i,0].plot(var, tf3, color = cmap[2], linestyle = '-')
    ax[i,0].plot(var, tf4, color = cmap[4], linestyle = '-')

    ax[i,0].plot(var, tf_crawford, color = 'k', alpha = 0.5, linestyle = '-')
    ax[i,0].plot(var, tf_silva, color = 'k', alpha = 0.5, linestyle = '--')
    ##---------------------------------------------------##

    ax[i,1].plot(var, s1, color = cmap[0], linestyle = '-')
    ax[i,1].plot(var, s2, color = cmap[1], linestyle = '-')
    ax[i,1].plot(var, s3, color = cmap[2], linestyle = '-')
    ax[i,1].plot(var, s4, color = cmap[4], linestyle = '-')

    ax[i,1].plot(var, sn2, color = cmap[1], linestyle = ':')
    ax[i,1].plot(var, sn3, color = cmap[2], linestyle = ':')
    ax[i,1].plot(var, sn4, color = cmap[4], linestyle = ':')

    ##---------------------------------------------------##

    ax[i,2].plot(var, ms1, color = cmap[0], linestyle = '-', linewidth = 1.5)
    ax[i,2].plot(var, ms2, color = cmap[1], linestyle = '-', linewidth = 1.5)
    ax[i,2].plot(var, ms3, color = cmap[2], linestyle = '-', linewidth = 1.5)
    ax[i,2].plot(var, ms4, color = cmap[4], linestyle = '-', linewidth = 1.5)

    ax[i,2].plot(var, ms_crawford, color = 'k', alpha = 0.5, linestyle = '-')
    ax[i,2].plot(var, ms_silva, color = 'k', alpha = 0.5, linestyle = '--')

    ax[i,0].set_xlim(var[0],var[-1]); 
    ax[i,1].set_xlim(var[0],var[-1]); 
    ax[i,2].set_xlim(var[0],var[-1]); 

    print(f'The average ratio of Depth Averaged Values HTM to Silva for {varName} exp ={np.mean(ms_silva/ms4)}')
    print(f'The average ratio of Depth Averaged Values HTM to white for {varName} exp ={np.mean(ms_crawford/ms4)}')

    for tf in [tf1, tf2, tf3, tf4]:
        a, b  = func_fit(each, tf)
        print(f'For {varName} a = {a}, and b = {b}')

    fig1.savefig(directory + figureNames[i] + '_vertProf.png', dpi = 300)
    fig2.savefig(directory + figureNames[i] + '_timeEvo.png', dpi = 300)

    i = i + 1


letters = ['a','b','c','d','e','f','g','h','i','j']


# Loop over each subplot and add the letter in the lower-left corner
for i, axis in enumerate(ax.flat):
    axis.text(0.01, 0.92, letters[i], transform=axis.transAxes,
            fontsize=16, fontweight = 'bold',  verticalalignment='bottom', horizontalalignment='left')

ax[2,0].legend(['BM','OFM', 'PEM', 'HTM', 'C24', 'S06'], loc = 4)
fig.tight_layout()
fig.savefig('/storage/home/hcoda1/1/mmamer3/scratch/waveProj/sensitivity_figure.jpg', dpi = 300)
