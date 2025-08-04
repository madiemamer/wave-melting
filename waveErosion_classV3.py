import numpy as np
import time
import matplotlib.pyplot as plt
import math
import cmocean.cm as cm
from scipy.ndimage import gaussian_filter1d

class waveErosion:

    def __init__(self, model_x, wave_x, domain_x, background_x, iterative_x, initial_x):
        
        ## Model Conditions
        self.modelType = model_x[0]         ## Model Hierarchy
        self.meltType = model_x[1]          ## Melt formulation
        self.printCheck = model_x[2]        ## Print checking

        ## Wave Parameters
        self.wave_H = wave_x[0]              ## Wave Height [m]
        self.lamb = wave_x[1]                ## Wave length [m]
        self.theta = math.radians(wave_x[2]) ## Ice Angle [deg]

        ## Background Conditions 
        self.Ubg = background_x[0]          ## Background velocity [m/s]
        self.Tbg = background_x[1]          ## Background Temperature [C]
        self.Sbg = background_x[2]          ## Background Salinity [ppt]
        self.rho_bg = self.rho_eq(self.Sbg, self.Tbg)

        ## Domain
        self.d_ocean = domain_x[0]          ## Total ocean depth [m]
        self.d = domain_x[1]                ## Total model domain depth [m]
        self.d_ice = domain_x[2]            ## Total ice depth [m]
        self.dz = domain_x[3]               ## Height resolution [m]
        self.Nz = len(self.Ubg)             ## Number of layers

        if self.modelType == 4:
            self.B = domain_x[4]            ## Width of Intermediate box [m]
        
        ## Iterative Conditions
        self.startTime = iterative_x[0]     ## Iteration Start [s]
        self.endTime = iterative_x[1]       ## Iteration End [s]
        self.dt = iterative_x[2]            ## Time step [s]
        self.checkFreq = iterative_x[3]     ## Export Frequency [s]
        self.iterativeType = iterative_x[4] ## Variable or constant dt

        ## Plume Constants
        self.E = 0.1                       ## Entrainment Coefficient [] from Cowton et. al 2015, Schulz et. al 2022
        self.gammaT = 10**(-3)              ## Thermal Turbulent Transfer Velocity [m/s] from Schulz et. al 2022
        self.gammaS = 0.07 * self.gammaT    ## Haline Turbulent Transfer Velocity [m/s] from Schulz et. al 2022

        ## Liquidus Constants
        self.lam1 = -0.0573
        self.lam2 = 0.0832
        self.lam3 = -7.53 * 10**(-4)        ## Negative because we abs(z) on line 122

        ## Ice Constants
        self.S_ice = 0                      ## Ice Salinity
        self.cp_ice = 2093                  ## Ice Heat capacity [J/kgK]
        self.L_ice = 334000                 ## Ice Latent Heat [J/kg]
        self.rho_ice = 917                  ## Ice Density [kg/m^3]

        ## Ocean Constants
        self.cp_ocean = 4019.0              ## Seawater heat capacity [J/kg]
        self.Pr = 13.3                      ## Seawater Prandtl Number
        self.rho_bg = self.rho_eq(self.Sbg, self.Tbg)
        self.nu = 1.9 * 10**(-6)            ## Kinemtic viscosity [m^2/s]

        ## Constants
        self.g = 9.81                       ## Gravity m/s^2

        ##---------------------------------------------------------------------------------------##
        ## Running the Model ##

        self.makeDomain()
        self.V, self.Cf = self.makeWaveParameters()
        self.makeIterative()
        self.makeInitial(initial_x)

        if self.modelType == 4:

            print(f"Preparing to Run the Horizontal Transport Model. \
                  Total Layers = {self.Nz}, \
                  Total iterations = {self.Nt}. \
                  dt = {self.dt}")
            
            self.horizontalEnabled(self.m, self.Tb, self.Sb, 
                                   self.b, self.Tn, self.Sn, self.rho_n, self.Us, self.Up, 
                                   self.B, self.Ti, self.Si, self.rho_i, self.Uif, self.Uin)
            
        elif self.modelType == 3:
            print(f"Preparing to Run the Plume Transport Model. \
                  Total Layers = {self.Nz}, \
                  Total iterations = {self.Nt}. \
                  dt = {self.dt}")
            
            self.plumeEnabled(self.m, self.Tb, self.Sb, 
                                   self.b, self.Tn, self.Sn, self.rho_n, self.Us, self.Up)
            
        elif self.modelType == 2:
            print(f"Preparing to Run the Ocean Transport Model. \
                  Total Layers = {self.Nz}, \
                  Total iterations = {self.Nt}. \
                  dt = {self.dt}")
            
            self.oceanEnabled(self.m, self.Tb, self.Sb, 
                                   self.b, self.Tn, self.Sn, self.rho_n, self.Us)
            
        
        elif self.modelType == 1:
            print(f"Preparing to Run the Base Model. \
                  Total Layers = {self.Nz}, \
                  Total iterations = {self.Nt}. \
                  dt = {self.dt}")
            
            self.boundaryModel_analytical()
            
        else:
            print('No model given or model type does not match the available models.')
        
        return
            
    ## Liquidus condition ##
    ## Inputs: Boundary salinity in ppt (Sb), depth in m (z). ##
    ## Returns: Freezing point temperature [C] same length as Sb and z. ##
    def Tb_eq(self, Sb, z):

        lam1 = np.copy(self.lam1)
        lam2 = np.copy(self.lam2)
        lam3 = np.copy(self.lam3)

        return Sb * lam1 + lam2 + abs(z) * lam3

    ## Density Equation of State ##
    ## Inputs: Salinity in ppt (S), Temperature in C (T) ##
    ## Outputs: Density [kg/m^3] same length as S and T ##
    def rho_eq(self, S, T):
        return 1000 + S * (0.7718) - T * (0.17765)
    
    ## Build Layer Array ##
    ## Outputs: define global array z, variable H, and indexing position of ice depth ##
    def makeDomain(self):
        print("Making Domain.")
        self.z = np.linspace(-self.d, 0, self.Nz)               ## Creates array for vertical layers
        self.H = self.dz                                        ## Sets the layer thickness [m]
        self.ice_idx = np.argmin(abs(self.d_ice + self.z))      ## Finds index of ice depth

        return 
    
    ## Build Wave Orbital Velocities ##
    ## Uses equations from White et. al 1980 chapter 8 to calculate ##
    ## orbital velocities based on given wave height and wave length. ##
    ## Outputs: Combined orbital wave velocity (V [m/s]) and friction factor (Cf). ##
    ## Defines global variables wave number and wave period. ##
    def makeWaveParameters(self): # z is vector of domain in vertical, d_ocean is total water depth
        print("Calculating Wave Orbital Velocities.")

        lamb = np.copy(self.lamb)
        wave_H = np.copy(self.wave_H)
        d_ocean = np.copy(self.d_ocean)
        theta = np.copy(self.theta)
        z = np.copy(self.z)
        nu = np.copy(self.nu)
        g = np.copy(self.g)  

        k = np.pi * 2 / lamb # wave number [m^-1]
        P = ((lamb * 2 * np.pi /( g * math.tanh(k * d_ocean)))**(1/2)) # wave period [s^-1]
        a = 0.5 * wave_H # wave amplitude [m]
        
        u = (np.pi * wave_H / P) * math.cos(theta) * np.cosh(k * (z + d_ocean)) / np.sinh(k * d_ocean) # wave horizontal orbital velocity [m/s]
        omega = (np.pi * wave_H / P) * (1+math.sin(theta)**2)**(1/2) * np.sinh(k * (z+d_ocean)) / np.sinh(k * d_ocean) # wave vertical orbital velocity [m/s]
        
        V = (u**2 + omega**2)**(1/2) # combined wave orbital velocity [m/s]
        Re_a = a * V / nu # wave abcissa reynolds number
        Cf = 0.09 * Re_a**(-0.2) #2 * (Re_a**(-0.5)) # wave skin friction - laminar version

        self.k = k
        self.P = P

        return V, Cf

    
    ## Create export time array ##
    ## Outputs: define global array t and variable Nt ##
    def makeIterative(self):
        print("Building Iteration arrays")

        startTime = np.copy(self.startTime)
        endTime = np.copy(self.endTime)
        checkFreq = np.copy(self.checkFreq)

        t = np.arange(startTime, endTime, checkFreq)
        Nt = len(t)

        self.t = t
        self.Nt = Nt

        return 
    
    ## Make initial conditions and solver matrices ##
    ## Inputs: matrix of initial conditions with shape # of variables x Nz ##
    ## Outputs: global matrices for each solver variable. ##
    def makeInitial(self, initial_x):
        print("Setting initial conditions and building solver arrays.")
        ## Boundary ##
        self.m = np.ones((self.Nt, self.Nz)) * np.nan
        self.Tb = np.ones((self.Nt, self.Nz)) * np.nan
        self.Sb = np.ones((self.Nt, self.Nz)) * np.nan

        self.m[0,:] = initial_x[0,:]
        self.Tb[0,:] = initial_x[1,:]
        self.Sb[0,:] = initial_x[2,:]

        if self.modelType > 1:
            ## Near Field ##
            self.b = np.zeros((self.Nt, self.Nz)) * np.nan
            self.Tn = np.zeros((self.Nt, self.Nz)) * np.nan
            self.Sn = np.zeros((self.Nt, self.Nz)) * np.nan
            self.rho_n = np.zeros((self.Nt, self.Nz)) * np.nan
            self.Us = np.ones((self.Nt,self.Nz)) * np.nan
            

            self.b[0,:] = initial_x[3, :]
            self.Tn[0,:] = initial_x[4, :]
            self.Sn[0,:] = initial_x[5, :]
            self.rho_n[0,:] = self.rho_eq(self.Sn[0,:], self.Tn[0,:])

            self.Us[0,:] = initial_x[6,:]

            if self.modelType > 2:
                ## Plume ##
                self.Up = np.ones((self.Nt, self.Nz)) * np.nan
                self.Up[0,:] = initial_x[7,:]

                ## Intermediate Field ##
                if self.modelType > 3:
                    self.B = np.zeros((self.Nt, self.Nz)) * np.nan 
                    self.Ti = np.zeros((self.Nt, self.Nz)) * np.nan
                    self.Si = np.zeros((self.Nt, self.Nz)) * np.nan
                    self.rho_i = np.zeros((self.Nt, self.Nz)) * np.nan
                    self.Uif = np.zeros((self.Nt, self.Nz)) * np.nan
                    self.Uin = np.zeros((self.Nt, self.Nz)) * np.nan 

                    self.B[0,:] = initial_x[8,:]
                    self.Ti[0,:] = initial_x[9,:]
                    self.Si[0,:] = initial_x[10,:]
                    self.rho_i[0,:] = self.rho_eq(self.Sn[0,:], self.Tn[0,:])

                    self.Uif[0,:] = initial_x[11,:]
                    self.Uin[0,:] = initial_x[12,:]    
        
        self.dts = np.zeros(self.Nt)

        return

    def horizontalEnabled(self, 
                          m, Tb, Sb, 
                          b, Tn, Sn, rho_n, Us, Up, 
                          B, Ti, Si, rho_i, Uif, Uin):

        ## Iterative Values ##
        t_running = 0
        dt = np.copy(self.dt)
        Nt = np.copy(self.Nt)
        
        ## Domain Values ##
        ice_idx = np.copy(self.ice_idx)
        H = np.copy(self.H)

        ## Background Conditions ##
        Ubg = np.copy(self.Ubg)
        Tbg = np.copy(self.Tbg)
        Sbg = np.copy(self.Sbg)

        ## Ice and Plume Values ##
        E = np.copy(self.E)
        S_ice = np.copy(self.S_ice)

        ## Wave Values ##
        V = np.copy(self.V)
        Cf = np.copy(self.Cf)

        ## Ocean Properties ##
        L_ice = np.copy(self.L_ice)
        cp_ocean = np.copy(self.cp_ocean)
        gammaS = np.copy(self.gammaS)
        g = np.copy(self.g)

        ## Initializing iterative before arrays ##
        m_before = np.copy(m[0,:])
        Tb_before = np.copy(Tb[0,:])
        Sb_before = np.copy(Sb[0,:])

        b_before = np.copy(b[0,:])
        Tn_before = np.copy(Tn[0,:])
        Sn_before = np.copy(Sn[0,:])
        rho_n_before = np.copy(rho_n[0,:])
        Us_before = np.copy(Us[0,:])
        Up_before = np.copy(Up[0,:])

        B_before = np.copy(B[0,:])
        Ti_before = np.copy(Ti[0,:])
        Si_before = np.copy(Si[0,:])
        rho_i_before = np.copy(rho_i[0,:])
        Uif_before = np.copy(Uif[0,:])
        Uin_before = np.copy(Uin[0,:])

        ## Initializing iterative next arrays ##
        m_next = np.zeros(self.Nz) * np.nan
        Tb_next = np.zeros(self.Nz) * np.nan
        Sb_next = np.zeros(self.Nz) * np.nan

        b_next = np.zeros(self.Nz) * np.nan
        Tn_next = np.zeros(self.Nz) * np.nan
        Sn_next = np.zeros(self.Nz) * np.nan
        rho_n_next = np.zeros(self.Nz) * np.nan
        Us_next = np.zeros(self.Nz) * np.nan
        Up_next = np.zeros(self.Nz) * np.nan

        B_next = np.zeros(self.Nz) * np.nan
        Ti_next = np.zeros(self.Nz) * np.nan
        Si_next = np.zeros(self.Nz) * np.nan
        rho_i_next = np.zeros(self.Nz) * np.nan
        Uif_next = np.zeros(self.Nz) * np.nan
        Uin_next = np.zeros(self.Nz) * np.nan

        print("Running the Horizontal Transport Model.")

        ## Iterating ##
        time_start = time.time()
        dts = np.zeros(Nt) * np.nan
        t_last_export = 0
        
        i = 0
        j = 0


        while t_running < (self.endTime - 1):

            ## Checking CFL Criteria ##
            u = np.array([
                            np.nanmax(abs(Us_before[:])), 
                            np.nanmax(abs(Uif_before[:])), 
                            np.nanmax(abs(Uin_before[:])), 
                            np.nanmax(abs(Ubg[:]))
                        ])

            w = np.array([
                            np.nanmax(abs(Up_before[:]))
                        ])

            dt = self.check_CFL(u, w, b_before[:])
            
            if dt > self.dt:
                dt = self.dt
            
            dts[i] = dt

            ## Discretized ODEs for Mass and scalar conservation ##

            b_next[0] = b_before[0] + (dt/H) * (
                                                  m_before[0] * H
                                                + Uin_before[0] * H
                                                - Us_before[0] * H
                                                - Up_before[0] * b_before[0]
                                                )
            
            b_next[1:] = b_before[1:] + (dt/H) * (
                                                     m_before[1:] * H
                                                    + Uin_before[1:] * H
                                                    - Us_before[1:] * H
                                                    + Up_before[:-1] * b_before[:-1]
                                                    - Up_before[1:] * b_before[1:]
                                                    )
            
            Tn_next[0] = Tn_before[0] + (dt/(H * b_before[0])) * (   
                                                                    - m_before[0] * H * (L_ice/cp_ocean)   
                                                                    + m_before[0] * H * (Tb_before[0]) ## melt flux
                                                                    + Uin_before[0] * H * (Ti_before[0]) ## Entrainment
                                                                    - Us_before[0] * H * (Tn_before[0]) ## Spreading 
                                                                    - Up_before[0] * b_before[0] * (Tn_before[0]) ## Plume leaving
                                                                )

            Tn_next[1:] = Tn_before[1:] + (dt/(H * b_before[1:])) * (
                                                                    - m_before[1:] * H * (L_ice/cp_ocean)
                                                                    + m_before[1:] * H * (Tb_before[1:]) ## melt flux
                                                                    + Uin_before[1:] * H * (Ti_before[1:]) ## Entrainment
                                                                    - Us_before[1:] * H * (Tn_before[1:]) ## Spreading 
                                                                    + Up_before[:-1] * b_before[:-1] * (Tn_before[:-1])
                                                                    - Up_before[1:] * b_before[1:] * (Tn_before[1:]) ## Plume leaving
                                                                )
            
            Sn_next[0] = Sn_before[0] + (dt/(H * b_before[0])) * (
                                                                      m_before[0] * H * (S_ice - Sn_before[0]) ## melt flux
                                                                    + Uin_before[0] * H * (Si_before[0]) ## Entrainment
                                                                    - Us_before[0] * H * (Sn_before[0]) ## Spreading 
                                                                    - Up_before[0] * b_before[0] * (Sn_before[0]) ## Plume leaving
                                                                )

            Sn_next[1:] = Sn_before[1:] + (dt/(H * b_before[1:])) * (
                                                                      m_before[1:] * H * (S_ice - Sn_before[1:]) ## melt flux
                                                                    + Uin_before[1:] * H * (Si_before[1:]) ## Entrainment
                                                                    - Us_before[1:] * H * (Sn_before[1:]) ## Spreading 
                                                                    + Up_before[:-1] * b_before[:-1] * (Sn_before[:-1])
                                                                    - Up_before[1:] * b_before[1:] * (Sn_before[1:]) ## Plume leaving
                                                                )
            
            B_next[:] = B_before[:] + (dt/H) * (
                                                Ubg * H  
                                                + Us_before[:] * H
                                                - Uif_before[:] * H
                                                - Uin_before[:] * H
                                                )
            
            Ti_next[:] = Ti_before + (dt/(B_before[:] * H)) * (
                                                                  Ubg * H * (Tbg) 
                                                                - Uif_before[:] * H * (Ti_before[:]) 
                                                                + Us_before[:] * H * (Tn_before[:]) 
                                                                - Uin_before[:] * H * (Ti_before[:])
                                                            )
            
            Si_next[:] = Si_before + (dt/(B_before[:] * H)) * (
                                                                  Ubg * H * (Sbg) 
                                                                - Uif_before[:] * H * (Si_before[:]) 
                                                                + Us_before[:] * H * (Sn_before[:]) 
                                                                - Uin_before[:] * H * (Si_before[:])
                                                            )
            Sb_next[ice_idx:] = gammaS * Sn_before[ice_idx:] / (gammaS + m_before[ice_idx:])
            Sb_next[:ice_idx] = Sn_before[:ice_idx]


            ## Diagnostic Properties ##
            
            ## Density ##
            rho_n_next[:] = self.rho_eq(Sn_next, Tn_next) 
            rho_i_next[:] = self.rho_eq(Si_next, Ti_next)

            ## Plume Dynamics ##
            gPrime = g * (rho_i_next[:] - rho_n_next[:]) / rho_i_next[:]
            gPrime[ice_idx:] = gaussian_filter1d(gPrime[ice_idx:], sigma = 2)

            gPrime = np.maximum(gPrime, 0)
            gPrime[:ice_idx] = 0

            Up_next[:ice_idx] = 0
            Up_next[ice_idx:] = (b_next[ice_idx:] * gPrime[ice_idx:])**(1/2)
            Up_next[:] = gaussian_filter1d(Up_next[:], sigma = 2)
            Up_next[-1] = 0
                
            ## Melting Dynamics ##
            Tb_next[:] = self.Tb_eq(Sb_next, self.z)
            Tn_next[:] = np.maximum(Tn_next, Tb_next)
            m_next[ice_idx:] = self.melt_eq(V[ice_idx:], Cf[ice_idx:], 
                                            Tn_next[ice_idx:], Tb_next[ice_idx:], 
                                            rho_n_next[ice_idx:], Up_next[ice_idx:])
            m_next[:ice_idx] = 0

            ## Transport from Intermediate to Near Field ##
            Uin_next[ice_idx:] = ((Up_next[ice_idx:] + Up_next[ice_idx-1:-1]) / 2) * E
            Uin_next[:ice_idx] = Ubg[:ice_idx]

            ## Transport from Near Field to Intermediate Field ##
            Us_next[1:] = (1/H) * (m_next[1:] * H + Up_next[0:-1] * b_next[0:-1] - Up_next[1:] * b_next[1:] + Uin_next[1:] * H)
            Us_next[0] = (1/H) * (m_next[0] * H - Up_next[0] * b_next[0] + Uin_next[0] * H)
            Us_next[:] = np.maximum(Us_next[:], 0)

            ## Transport from Intermediate to Far Field, exiting model domain ##
            Uif_next[:] = Ubg + Us_next[:] - Uin_next[:]
          
            ## Updating the time counter ##
            t_running = t_running + dt

            ## Evaluating Checking Frequency to save Data ##
            if t_running >= t_last_export + self.checkFreq - 1e-6:

                m[i+1, :] = m_next
                Tb[i+1, :] = Tb_next
                Sb[i+1, :] = Sb_next

                b[i+1, :] = b_next
                Tn[i+1, :] = Tn_next
                Sn[i+1, :] = Sn_next
                rho_n[i+1, :] = rho_n_next
                Us[i+1, :] = Us_next
                Up[i+1, :] = Up_next
                
                B[i+1, :] = B_next
                Ti[i+1, :] = Ti_next
                Si[i+1, :] = Si_next
                rho_i[i+1,:] = rho_i_next
                Uin[i+1, :] = Uin_next
                Uif[i+1,:] = Uif_next
                
                print(f"Iteration {j}, time = {t_running.round(3)}s, index = {i}, dt = {dt}s")
                
                t_last_export = t_last_export + self.checkFreq
                i = i + 1
            
            ## Updating dummy variables for next iteration ##

            m_before = m_next
            Tb_before = Tb_next
            Sb_before = Sb_next

            b_before = b_next
            Tn_before = Tn_next
            Sn_before = Sn_next
            rho_n_before = rho_n_next
            Us_before = Us_next
            Up_before = Up_next

            B_before = B_next
            Ti_before = Ti_next
            Si_before = Si_next
            rho_i_before = rho_i_next
            Uin_before = Uin_next
            Uif_before = Uif_next

            j = j + 1

            # if j > 1000000:
            #     break
            ## Break iterative loop if Nan ##
            # if np.isnan(res_norm):
            #     break

        ## Exporting Model solution to global variables ##

        self.m = m
        self.Tb = Tb
        self.Sb = Sb
        
        self.b = b
        self.Tn = Tn
        self.Sn = Sn
        self.rho_n = rho_n
        self.Us = Us
        self.Up = Up

        self.B = B
        self.Ti = Ti
        self.Si = Si
        self.rho_i = rho_i
        self.Uin = Uin
        self.Uif = Uif

        self.dts = dts

        time_stop = time.time()

        print(f"Finished running. Total time = {time_stop - time_start:.2f}s")
        print('')

        return

    def plumeEnabled(self, 
                            m, Tb, Sb, 
                            b, Tn, Sn, rho_n, Us, Up):

            ## Iterative Values ##
            t_running = 0
            dt = np.copy(self.dt)
            Nt = np.copy(self.Nt)
            
            ## Domain Values ##
            ice_idx = np.copy(self.ice_idx)
            H = np.copy(self.H)
            z = np.copy(self.z)

            ## Background Conditions ##
            Ubg = np.copy(self.Ubg)
            Ubg[ice_idx:] = 0
            Tbg = np.copy(self.Tbg)
            Sbg = np.copy(self.Sbg)

            ## Ice and Plume Values ##
            S_ice = np.copy(self.S_ice)
            E = np.copy(self.E)

            ## Wave Values ##
            V = np.copy(self.V)
            Cf = np.copy(self.Cf)

            ## Ocean Values ##
            rho_bg = np.copy(self.rho_bg)
            L_ice = np.copy(self.L_ice)
            cp_ocean = np.copy(self.cp_ocean)
            gammaS = np.copy(self.gammaS)

            ## Constants ##
            g = np.copy(self.g)

            ## Initializing iterative before arrays ##
            m_before = np.copy(m[0,:])
            Tb_before = np.copy(Tb[0,:])
            Sb_before = np.copy(Sb[0,:])

            b_before = np.copy(b[0,:])
            Tn_before = np.copy(Tn[0,:])
            Sn_before = np.copy(Sn[0,:])
            rho_n_before = np.copy(rho_n[0,:])
            Us_before = np.copy(Us[0,:])
            Up_before = np.copy(Up[0,:])

            ## Initializing iterative next arrays ##
            m_next = np.zeros(self.Nz) * np.nan
            Tb_next = np.zeros(self.Nz) * np.nan
            Sb_next = np.zeros(self.Nz) * np.nan

            b_next = np.zeros(self.Nz) * np.nan
            Tn_next = np.zeros(self.Nz) * np.nan
            Sn_next = np.zeros(self.Nz) * np.nan
            rho_n_next = np.zeros(self.Nz) * np.nan
            Us_next = np.zeros(self.Nz) * np.nan
            Up_next = np.zeros(self.Nz) * np.nan


            print("Running the Plume Transport Model.")

            ## Iterating ##
            time_start = time.time()
            dts = np.zeros(Nt) * np.nan
            t_last_export = 0
            
            i = 0
            j = 0

            while t_running < (self.endTime - 1):

                ## Checking CFL Criteria ##
                u = np.array([
                                np.nanmax(abs(Us_before[:])), 
                                np.nanmax(abs(Ubg[:]))
                            ])

                w = np.array([
                                np.nanmax(abs(Up_before[:]))
                            ])

                dt = self.check_CFL(u, w, b_before[:])
                
                if dt > self.dt:
                    dt = self.dt
                
                dts[i] = dt

                ## Discretized ODEs for Mass and scalar conservation ##

                b_next[0] = b_before[0] + (dt/H) * (
                                                    m_before[0] * H
                                                    + Ubg[0] * H
                                                    + Up_before[0] * E * H
                                                    - Us_before[0] * H
                                                    - Up_before[0] * b_before[0]
                                                    )
                
                b_next[1:] = b_before[1:] + (dt/H) * (
                                                        m_before[1:] * H
                                                        + (Up_before[1:] + Up_before[0:-1])/2 * E * H
                                                        + Ubg[1:] * H
                                                        - Us_before[1:] * H
                                                        + Up_before[:-1] * b_before[:-1]
                                                        - Up_before[1:] * b_before[1:]
                                                        )
                
                Tn_next[0] = Tn_before[0] + (dt/(H * b_before[0])) * (
                                                                        - m_before[0] * H * (L_ice/cp_ocean)
                                                                        + m_before[0] * H * (Tb_before[0]) ## melt flux
                                                                        + Up_before[0] * E * H * (Tbg[0]) ## Entrainment
                                                                        + Ubg[0] * H * (Tbg[0])
                                                                        - Us_before[0] * H * (Tn_before[0]) ## Spreading 
                                                                        - Up_before[0] * b_before[0] * (Tn_before[0]) ## Plume leaving
                                                                    )

                Tn_next[1:] = Tn_before[1:] + (dt/(H * b_before[1:])) * (
                                                                        - m_before[1:] * H * (L_ice/cp_ocean)
                                                                        + m_before[1:] * H * (Tb_before[1:]) ## melt flux
                                                                        + (Up_before[1:] + Up_before[0:-1])/2 * E * H * (Tbg[1:]) ## Entrainment
                                                                        + Ubg[1:] * H * Tbg[1:]
                                                                        - Us_before[1:] * H * (Tn_before[1:]) ## Spreading 
                                                                        + Up_before[:-1] * b_before[:-1] * (Tn_before[:-1])
                                                                        - Up_before[1:] * b_before[1:] * (Tn_before[1:]) ## Plume leaving
                                                                    )
                
                Sn_next[0] = Sn_before[0] + (dt/(H * b_before[0])) * (
                                                                        m_before[0] * H * (S_ice - Sn_before[0]) ## melt flux
                                                                        + Up_before[0] * E *  H * (Sbg[0]) ## Entrainment
                                                                        + Ubg[0] * H * Sbg[0]
                                                                        - Us_before[0] * H * (Sn_before[0]) ## Spreading 
                                                                        - Up_before[0] * b_before[0] * (Sn_before[0]) ## Plume leaving
                                                                    )

                Sn_next[1:] = Sn_before[1:] + (dt/(H * b_before[1:])) * (
                                                                        m_before[1:] * H * (S_ice - Sn_before[1:]) ## melt flux
                                                                        + (Up_before[1:] + Up_before[0:-1])/2 * E * H * (Sbg[1:]) ## Entrainment
                                                                        + Ubg[1:] * H * Sbg[1:]
                                                                        - Us_before[1:] * H * (Sn_before[1:]) ## Spreading 
                                                                        + Up_before[:-1] * b_before[:-1] * (Sn_before[:-1])
                                                                        - Up_before[1:] * b_before[1:] * (Sn_before[1:]) ## Plume leaving
                                                                    )
                
        

                Sb_next[ice_idx:] = gammaS * Sn_before[ice_idx:] / (gammaS + m_before[ice_idx:])
                Sb_next[:ice_idx] = Sn_before[:ice_idx]

                ## Diagnostic Properties ##
                
                ## Density ##
                rho_n_next[:] = self.rho_eq(Sn_next, Tn_next)

                ## Plume Dynamics ##
                gPrime = g * (rho_bg[:] - rho_n_next[:]) / rho_bg[:]
                gPrime[ice_idx:] =  gaussian_filter1d(gPrime[ice_idx:], sigma = 2)
                gPrime = np.maximum(gPrime, 0)
                gPrime[:ice_idx] = 0

                Up_next[:ice_idx] = 0
                Up_next[ice_idx:] = (b_next[ice_idx:] * gPrime[ice_idx:])**(1/2)
                Up_next[:] = gaussian_filter1d(Up_next[:], sigma = 2)
                Up_next[-1] = 0

                ## Melting Dynamics ##
                Tb_next[:] = self.Tb_eq(Sb_next, z)
                Tn_next[:] = np.maximum(Tn_next, Tb_next)
                m_next[ice_idx:] = self.melt_eq(V[ice_idx:], Cf[ice_idx:], 
                                                Tn_next[ice_idx:], Tb_next[ice_idx:], 
                                                rho_n_next[ice_idx:], Up_next[ice_idx:])
                m_next[:ice_idx] = 0

                ## Transport from Near Field to Intermediate Field ##
                Us_next[1:] = (1/H) * (m_next[1:] * H + Up_next[0:-1] * b_next[0:-1] - Up_next[1:] * b_next[1:] + (Up_before[1:] + Up_before[0:-1])/2 * E * H + Ubg[1:] * H)
                Us_next[0] = (1/H) * (m_next[0] * H - Up_next[0] * b_next[0] + Up_next[0] * E * H + Ubg[0] * H)
                Us_next = np.maximum(Us_next[:], 0)
                
                ## Updating the time counter ##
                t_running = t_running + dt

                ## Evaluating Checking Frequency to save Data ##
                if t_running >= t_last_export + self.checkFreq - 1e-6:

                    m[i+1, :] = m_next
                    Tb[i+1, :] = Tb_next
                    Sb[i+1, :] = Sb_next

                    b[i+1, :] = b_next
                    Tn[i+1, :] = Tn_next
                    Sn[i+1, :] = Sn_next
                    rho_n[i+1, :] = rho_n_next
                    Us[i+1, :] = Us_next
                    Up[i+1, :] = Up_next
                    
                    print(f"Iteration {j}, time = {t_running.round(3)}s, index = {i}, dt = {dt}s")
                    
                    t_last_export = t_last_export + self.checkFreq
                    i = i + 1
                
                ## Updating dummy variables for next iteration ##

                m_before = m_next
                Tb_before = Tb_next
                Sb_before = Sb_next

                b_before = b_next
                Tn_before = Tn_next
                Sn_before = Sn_next
                rho_n_before = rho_n_next
                Us_before = Us_next
                Up_before = Up_next

                j = j + 1

                # if j > 1000000:
                #     break
                ## Break iterative loop if Nan ##
                # if np.isnan(res_norm):
                #     break

            ## Updating Model Solution to Global Variables ##
            self.m = m
            self.Tb = Tb
            self.Sb = Sb
            
            self.b = b
            self.Tn = Tn
            self.Sn = Sn
            self.rho_n = rho_n
            self.Us = Us
            self.Up = Up

            self.dts = dts

            time_stop = time.time()

            print(f"Finished running. Total time = {time_stop - time_start:.2f}s")
            print('')

            return

    def oceanEnabled(self, 
                            m, Tb, Sb, 
                            b, Tn, Sn, rho_n, Us):

            ## Iterative Values ##
            t_running = 0
            dt = np.copy(self.dt)
            Nt = np.copy(self.Nt)
            
            ## Domain Values ##
            ice_idx = np.copy(self.ice_idx)
            H = np.copy(self.H)

            ## Background Conditions ##
            Ubg = np.copy(self.Ubg)
            Tbg = np.copy(self.Tbg)
            Sbg = np.copy(self.Sbg)

            ## Ice and Plume Values ##
            L_ice = np.copy(self.L_ice)
            S_ice = np.copy(self.S_ice)
            E = np.copy(self.E)

            ## Wave Values ##
            V = np.copy(self.V)
            Cf = np.copy(self.Cf)

            ## Ocean Values ##
            cp_ocean = np.copy(self.cp_ocean)
            gammaS = np.copy(self.gammaS)

            ## Initializing iterative before arrays ##
            m_before = np.copy(m[0,:])
            Tb_before = np.copy(Tb[0,:])
            Sb_before = np.copy(Sb[0,:])

            b_before = np.copy(b[0,:])
            Tn_before = np.copy(Tn[0,:])
            Sn_before = np.copy(Sn[0,:])
            rho_n_before = np.copy(rho_n[0,:])
            Us_before = np.copy(Us[0,:])

            ## Initializing iterative next arrays ##
            m_next = np.zeros(self.Nz) * np.nan
            Tb_next = np.zeros(self.Nz) * np.nan
            Sb_next = np.zeros(self.Nz) * np.nan

            b_next = np.zeros(self.Nz) * np.nan
            Tn_next = np.zeros(self.Nz) * np.nan
            Sn_next = np.zeros(self.Nz) * np.nan
            rho_n_next = np.zeros(self.Nz) * np.nan
            Us_next = np.zeros(self.Nz) * np.nan


            print("Running the Ocean Transport Model.")

            ## Iterating ##
            time_start = time.time()
            dts = np.zeros(Nt) * np.nan
            t_last_export = 0
            
            i = 0
            j = 0

            while t_running < (self.endTime - 1):

                ## Checking CFL Criteria ##
                u = np.array([
                                np.nanmax(abs(Us_before[:])), 
                                np.nanmax(abs(Ubg[:]))
                            ])
                
                w = np.array([0])

                dt = self.check_CFL(u, w, b_before[:])
                
                if dt > self.dt:
                    dt = self.dt
                
                dts[i] = dt

                ## Discretized ODEs for Mass and scalar conservation ##

                b_next[:] = b_before[:] + (dt/H) * (
                                                    m_before[:] * H
                                                    + Ubg[:] * H
                                                    - Us_before[:] * H
                                                    )
                
                Tn_next[:] = Tn_before[:] + (dt/(H * b_before[:])) * (
                                                                        - m_before[:] * H * (L_ice/cp_ocean)
                                                                        + m_before[:] * H * (Tb_before[:]) ## melt flux
                                                                        + Ubg[:] * H * (Tbg[:])
                                                                        - Us_before[:] * H * (Tn_before[:]) ## Spreading 
                                                                    )
                
                Sn_next[:] = Sn_before[:] + (dt/(H * b_before[:])) * (
                                                                        m_before[0] * H * (S_ice - Sn_before[:]) ## melt flux
                                                                        + Ubg[:] * H * Sbg[:]
                                                                        - Us_before[:] * H * (Sn_before[:]) ## Spreading g
                                                                    )

                Sb_next[ice_idx:] = gammaS * Sn_before[ice_idx:] / (gammaS + m_before[ice_idx:])
                Sb_next[:ice_idx] = Sn_before[:ice_idx]

                ## Diagnostic Properties ##
                
                ## Density ##
                rho_n_next[:] = self.rho_eq(Sn_next, Tn_next)

                ## Melting Dynamics ##
                Tb_next[:] = self.Tb_eq(Sb_next, self.z)
                Tn_next[:] = np.maximum(Tn_next, Tb_next)
                m_next[ice_idx:] = self.melt_eq(V[ice_idx:], Cf[ice_idx:], 
                                                Tn_next[ice_idx:], Tb_next[ice_idx:], 
                                                rho_n_next[ice_idx:], 0)
                m_next[:ice_idx] = 0

                ## Transport from Near Field to Intermediate Field ##
                Us_next[1:] = (1/H) * (m_next[1:] * H  + Ubg[1:] * H)
                Us_next[0] = (1/H) * (m_next[0] * H + Ubg[0] * H)

                ## Updating the time counter ##
                t_running = t_running + dt

                ## Evaluating Checking Frequency to save Data ##
                if t_running >= t_last_export + self.checkFreq - 1e-6:

                    m[i+1, :] = m_next
                    Tb[i+1, :] = Tb_next
                    Sb[i+1, :] = Sb_next

                    b[i+1, :] = b_next
                    Tn[i+1, :] = Tn_next
                    Sn[i+1, :] = Sn_next
                    rho_n[i+1, :] = rho_n_next
                    Us[i+1, :] = Us_next
                    
                    print(f"Iteration {j}, time = {t_running.round(3)}s, index = {i}, dt = {dt}s")
                    
                    t_last_export = t_last_export + self.checkFreq
                    i = i + 1
                
                ## Updating the dummy variables ##
                m_before = m_next
                Tb_before = Tb_next
                Sb_before = Sb_next

                b_before = b_next
                Tn_before = Tn_next
                Sn_before = Sn_next
                rho_n_before = rho_n_next
                Us_before = Us_next

                j = j + 1

                # if j > 1000000:
                #     break
                ## Break iterative loop if Nan ##
                # if np.isnan(res_norm):
                #     break

            ## Updating the Model Solution to the global variables ##
            self.m = m
            self.Tb = Tb
            self.Sb = Sb
            
            self.b = b
            self.Tn = Tn
            self.Sn = Sn
            self.rho_n = rho_n
            self.Us = Us

            time_stop = time.time()

            self.dts = dts

            print(f"Finished running. Total time = {time_stop - time_start:.2f}s")
            print('')

            return

    def boundaryModel_analytical(self):
        time_start = time.time()

        ## Background Conditions ##
        Tbg = np.copy(self.Tbg)
        Sbg = np.copy(self.Sbg)
        rho_bg = np.copy(self.rho_bg)
        cp_ocean = np.copy(self.cp_ocean)
        Pr = np.copy(self.Pr)
        V = np.copy(self.V)
        L_ice = np.copy(self.L_ice)
        rho_ice = np.copy(self.rho_ice)
        Cf = np.copy(self.Cf)
        gammaS = np.copy(self.gammaS)
        z = np.copy(self.z)
        lam1 = np.copy(self.lam1)
        lam2 = np.copy(self.lam2)
        lam3 = np.copy(self.lam3)
        ice_idx = np.copy(self.ice_idx)

        chi = (rho_bg * cp_ocean) / (L_ice * rho_ice)
        beta = V * (1/2 * Cf) / (1 + 12.8 * (Pr**(0.68) - 1)*(1/2 * Cf)**(1/2))
        C1 = Tbg - lam2 - lam3 * z

        a = rho_ice
        b = gammaS * rho_bg - chi * beta * (C1 * rho_ice)
        c = -chi * beta * gammaS * rho_bg * (C1 - Sbg * lam1)

        m1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        # m2 = -b - np.sqrt(b**2 - 4 * a * c) / (2 * a)
        m = m1#np.maximum(m1, m2)
        m[:ice_idx] = 0

        A = chi * beta * C1 /2 - (gammaS * rho_bg)/(2 * rho_ice)
        B = gammaS * rho_bg * chi * beta / (2 * rho_ice)
        C = np.sqrt(1/(chi**2 * beta**2) + (C1 * rho_ice)/(gammaS**2 * rho_bg**2) + rho_ice / (chi * beta * gammaS * rho_bg) * (2 * C1 - 4 * Sbg * lam1))

        m1 = A + B * C
        m = m1#np.maximum(m1, m2)
        m[:ice_idx] = 0
        
        Sb = gammaS * Sbg * rho_bg / (gammaS * rho_bg + m * rho_ice)
        Sb[:ice_idx] = Sbg[:ice_idx]
        Tb = Sb * lam1 + lam2 + z * lam3

        self.m[:, :] = m
        self.Tb[:, :] = Tb
        self.Sb[:, :] = Sb

        time_stop = time.time()

        print(f"Finished running. Total time = {time_stop - time_start:.2f}s")
        print('')

        return
    
    ## Equation 1 from Crawford et. al 2024, ##
    ## adaption from White et. al 1980, ##
    ## with thermal forcing = Tbg - Tm. ##
    ## Tm is the "melting temperature" ##
    def crawford2024_v0(self):

        H = np.copy(self.wave_H)
        P = np.copy(self.P)
        k = np.copy(self.k)

        Sbg = np.copy(self.Sbg)
        Tbg = np.copy(self.Tbg)
        z = np.copy(self.z)
        Tb = self.Tb_eq(Sbg, z)

        nu = np.copy(self.nu)

        a = 1.5 * 10**(-4) ## Model Constant
        alpha = 0.19 ## Model Constant

        ReH = (H**2/(P * nu)) * np.exp(-2 * k * z)
        
        Tm = -1.80 * np.exp(-alpha * (Tbg - (-1.80)))

        return (H/P) * a * ReH**(-0.12) * (Tbg - Tm)
    
    ## Equation 1 from Crawford et. al 2024, ##
    ## adaption from White et. al 1980, ##
    ## with thermal forcing = Tbg - Tb. ##
    def crawford2024_v1(self):

        H = np.copy(self.wave_H)
        P = np.copy(self.P)
        k = np.copy(self.k)

        Sbg = np.copy(self.Sbg)
        Tbg = np.copy(self.Tbg)
        z = np.copy(self.z)
        Tb = self.Tb_eq(Sbg, z)

        nu = np.copy(self.nu)

        a = 1.5 * 10**(-4) ## Model Constant

        ReH = (H**2/(P * nu)) * np.exp(-2 * k * z)

        return (H/P) * a * ReH**(-0.12) * (Tbg - Tb)
    
    ## Reynolds Analogy derived in White 1974. ##
    ## thermal forcing = Tbg - Tb ##
    def white1974(self):

        V = np.copy(self.V)
        Cf = np.copy(self.Cf)
        Pr = np.copy(self.Pr)
        rho_bg = np.copy(self.rho_bg)
        L_ice = np.copy(self.L_ice)
        rho_ice = np.copy(self.rho_ice)
        Tbg = np.copy(self.Tbg)
        Sbg = np.copy(self.Sbg)
        z = np.copy(self.z)

        Tb = self.Tb_eq(Sbg, z)

        m = ((rho_bg * 4019.2 * V) / (L_ice * rho_ice)) * (Tbg-Tb) * (1/2 * Cf) / (1 + 12.8*(Pr**(0.68) - 1)*(1/2*Cf)**(1/2))

        return m
    
    ## Parameterization used in Silva et. al 2006 ##
    ## thermal forcing = Tbg + 2##
    def silva2006_v0(self,Ss,C):

        Tbg = np.copy(self.Tbg)

        return (1/12) * Ss * (1) * (Tbg + 2)
    
    ## Parameterization used in Silva et. al 2006 ##
    ## thermal forcing = Tbg - Tb##
    def silva2006_v1(self, Ss, C):

        Sbg = np.copy(self.Sbg)
        Tbg = np.copy(self.Tbg)
        z = np.copy(self.z)

        Tb = self.Tb_eq(Sbg, z)

        return (1/12) * Ss * (1) * (Tbg - Tb)
    
    ## Function with different forms of the Melt Rate. ##
    ## meltType = 0 returns a melt rate based on the reynolds analogy function from White 1974 ##
    ## meltType = 1 returns a melt rate based on ambient plume melting and wave-induced melting ##
    ## meltType = 2 returns a melt rate based on ambient plume melting only ##
    ## returns melt rate = 0 if meltType != 0, 1, or 2 ##
    def melt_eq(self, V, Cf, Tn, Tb, rho_n, Up):

        meltType = np.copy(self.meltType)

        cp_ocean = np.copy(self.cp_ocean)
        L_ice = np.copy(self.L_ice)
        rho_ice = np.copy(self.rho_ice)
        Pr = np.copy(self.Pr)

        gammaT = 0.005 * Up
        
        if meltType == 0: ## Wave

            return ((rho_n * cp_ocean * V) / (L_ice * rho_ice)) * (Tn - Tb) * (1/2 * Cf) / (1 + 12.8*(Pr**(0.68) - 1)*(1/2*Cf)**(1/2))
        
        elif meltType == 1: ## Ambient and Wave

            wave = ((rho_n * cp_ocean * V) / (L_ice * rho_ice)) * ((1/2 * Cf) / (1 + 12.8*(Pr**(0.68) - 1)*(1/2*Cf)**(1/2))) 

            return  (Tn - Tb) * (wave + (rho_n * gammaT * cp_ocean) / (rho_ice * L_ice))
        
        elif meltType == 2: ## Ambient
            return (Tn - Tb) * (rho_n * gammaT * cp_ocean) / (rho_ice * L_ice)
        
        else:
            return np.zeros(len(Tb))

    ## Function to check Courant number ##
    def check_CFL(self, u, w, b):

        u_max = np.max(u)
        w_max = np.max(w)

        dt1 = 0.5 * self.dz/w_max
        dt2 = 0.5 * np.nanmin(b)/u_max

        if self.iterativeType == 'constant_dt':
            dt = np.array([self.dt])
        else:
            dt = np.nanmin([dt1, dt2])

        return dt 
    
    def getModelOutput(self):

        if self.modelType == 4:
            return self.m, self.Tb, self.Sb, self.b, self.Tn, self.Sn, self.rho_n, self.Us, self.Up, self.B, self.Ti, self.Si, self.rho_i, self.Uif, self.Uin
        
        elif self.modelType == 3:
            return self.m, self.Tb, self.Sb, self.b, self.Tn, self.Sn, self.rho_n, self.Us, self.Up
        
        elif self.modelType == 2:
            return self.m, self.Tb, self.Sb, self.b, self.Tn, self.Sn, self.rho_n, self.Us
        
        elif self.modelType == 1:
            return self.m, self.Tb, self.Sb

        else:
            print("No Model matches that value.")
            return
        
    def getDomain(self):
        return self.z, self.ice_idx, self.H, self.dz, self.dts