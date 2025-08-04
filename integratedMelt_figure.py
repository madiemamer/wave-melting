import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.25
mpl.rcParams['legend.fontsize'] = 14

size = 20

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

fig,ax = plt.subplots(1, figsize = (10,5))
fig2,ax2 = plt.subplots(1, figsize = (10,5))

ax.set_ylabel('Depth-Averaged Melt Rate [md$^{-1}$]')
ax.set_xlabel('Waterline Melt Rate [md$^{-1}$]')
ax.grid(alpha = 0.25, zorder = 1)

ax2.set_ylabel('Depth-Averaged Melt Rate [md$^{-1}$]')
ax2.set_xlabel('Waterline Melt Rate [md$^{-1}$]')
ax2.grid(alpha = 0.25, zorder = 1)

figureName = '/integratedMelt'
i = 0
Tbgs = np.linspace(-1.8, 5, 100)
lambs = np.linspace(5, 100, 100)
wave_Hs = np.linspace(0.01, 7, 100)
n = int(len(wave_Hs))# + len(lambs) + len(Tbgs))

ms = np.zeros(n)
Ss = np.zeros(n)
Ts = np.zeros(n)


directories = ['/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_HSensitivity',
               '/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_TbgSensitivity']

j = 0
for directory in directories:
    
    if j == 0:
        var = wave_Hs
        varName = 'waveH'
    else:
        var = Tbgs
        varName = 'Tbg'
    
    n = int(len(var))
    waveHs_dA1 = np.zeros((n))
    waveHs_dA2 = np.zeros((n))
    waveHs_dA3 = np.zeros((n))
    waveHs_dA4 = np.zeros((n))
    waveHs_dAcrawford = np.zeros((n))
    waveHs_dAsilva = np.zeros((n))

    waveHs1 = np.zeros((n))
    waveHs2 = np.zeros((n))
    waveHs3 = np.zeros((n))
    waveHs4 = np.zeros((n))
    waveHs_crawford = np.zeros((n))
    waveHs_silva = np.zeros((n))

    i = 0
    for each in var:

        print(f"Plotting for {varName} {each}.")
        dir = directory + f"/{varName}_{each}"

        m1 = np.loadtxt(dir + '/m1.csv', delimiter = ',') * 24 * 3600

        ##---------------------------------------------------##
        m2 = np.loadtxt(dir + '/m2.csv', delimiter = ',') * 24 * 3600

        ##---------------------------------------------------##
        m3 = np.loadtxt(dir + '/m3.csv', delimiter = ',') * 24 * 3600

        ##---------------------------------------------------##
        m4 = np.loadtxt(dir + '/m4.csv', delimiter = ',') * 24 * 3600

        m_crawford = np.loadtxt(dir + '/m_crawford.csv', delimiter = ',') * 24 * 3600
        m_silva = np.loadtxt(dir + '/m_silva.csv', delimiter = ',')

        waveHs_dA1[i] = np.mean(m1[-1,ice_idx:])
        waveHs_dA2[i] = np.mean(m2[-1,ice_idx:])
        waveHs_dA3[i] = np.mean(m3[-1,ice_idx:])
        waveHs_dA4[i] = np.mean(m4[-1,ice_idx:])
        waveHs_dAcrawford[i] = np.mean(m_crawford[ice_idx:])
        waveHs_dAsilva[i] = np.mean(m_silva[ice_idx:])

        waveHs1[i] = m1[-1,-1]
        waveHs2[i] = m2[-1,-1]
        waveHs3[i] = m3[-1,-1]
        waveHs4[i] = m4[-1,-1]
        waveHs_crawford[i] = m_crawford[-1]
        waveHs_silva[i] = m_silva[-1]

        i = i + 1

    ax.scatter(waveHs1[::5], waveHs_dA1[::5], color = cmap[0], s = size, zorder = 2)
    ax.scatter(waveHs2[::5], waveHs_dA2[::5], color = cmap[1], s = size, zorder = 2)
    ax.scatter(waveHs3[::5], waveHs_dA3[::5], color = cmap[2], s = size, zorder = 2)
    ax.scatter(waveHs4[::5], waveHs_dA4[::5], color = cmap[4], s = size, zorder = 2)
    ax.scatter(waveHs_crawford[::5], waveHs_dAcrawford[::5], color = 'gray', s = size,  zorder = 2)
    ax.scatter(waveHs_silva[::5], waveHs_dAsilva[::5], color = 'gray', s = size,  marker = 'D', zorder = 2)

    ax.plot(waveHs1, waveHs_dA1, color = cmap[0], linestyle = '-', zorder = 1)
    ax.plot(waveHs2, waveHs_dA2, color = cmap[1], linestyle = '-', zorder = 1)
    ax.plot(waveHs3, waveHs_dA3, color = cmap[2], linestyle = '-', zorder = 1)
    ax.plot(waveHs4, waveHs_dA4, color = cmap[4], linestyle = '-', zorder = 1)
    ax.plot(waveHs_crawford, waveHs_dAcrawford, color = 'gray', linestyle = '-', zorder = 1)
    ax.plot(waveHs_silva, waveHs_dAsilva, color = 'gray', linestyle = '-', zorder = 1)

    j = j + 1 

ax.legend(['BM','OFM','PEM','HTM','C24','S06'])
ax.set_xlim(0,7)
fig.tight_layout()
fig.savefig('/storage/home/hcoda1/1/mmamer3/scratch/waveProj' + figureName + '.jpg', dpi = 300)

directory = '/storage/home/hcoda1/1/mmamer3/scratch/waveProj/wave_LambSensitivity'
i = 0 
n = int(len(lambs))

ms = np.zeros(n)
Ss = np.zeros(n)
Ts = np.zeros(n)

alphas = np.linspace(0.2,1,len(lambs))
for each in lambs:

    dir = directory + f"/lamb_{each}"

    m1 = np.loadtxt(dir + '/m1.csv', delimiter = ',') * 24 * 3600

    ##---------------------------------------------------##
    m2 = np.loadtxt(dir + '/m2.csv', delimiter = ',') * 24 * 3600

    ##---------------------------------------------------##
    m3 = np.loadtxt(dir + '/m3.csv', delimiter = ',') * 24 * 3600

    ##---------------------------------------------------##
    m4 = np.loadtxt(dir + '/m4.csv', delimiter = ',') * 24 * 3600

    m_crawford = np.loadtxt(dir + '/m_crawford.csv', delimiter = ',') * 24 * 3600
    m_silva = np.loadtxt(dir + '/m_silva.csv', delimiter = ',')

    ax2.scatter(m1[-1,-1], np.mean(m1[-1,ice_idx:]), color = cmap[0], alpha = alphas[i],  zorder = 2)
    ax2.scatter(m2[-1,-1], np.mean(m2[-1,ice_idx:]), color = cmap[1], alpha = alphas[i],  zorder = 2)
    ax2.scatter(m3[-1,-1], np.mean(m3[-1,ice_idx:]), color = cmap[2], alpha = alphas[i],  zorder = 2)
    ax2.scatter(m4[-1,-1], np.mean(m4[-1,ice_idx:]), color = cmap[4], alpha = alphas[i],  zorder = 2)
    ax2.scatter(m_crawford[-1], np.mean(m_crawford[ice_idx:]), color = 'gray', alpha = alphas[i],  zorder = 2)
    ax2.scatter(m_silva[-1], np.mean(m_silva[ice_idx:]), color = 'gray', marker = 'D', zorder = 2)


    i = i + 1


ax2.legend(['BM','OFM','PEM','HTM','C24','S06'])
fig2.tight_layout()
fig2.savefig('/storage/home/hcoda1/1/mmamer3/scratch/waveProj' + figureName + '_lamb.png', dpi = 300)
