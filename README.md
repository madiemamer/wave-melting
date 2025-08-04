# wave-melting
Repo associated with manuscript titled, "Buoyancy Feedbacks on Wave-Induced Melting of Icebergs," submitted to JGR: Oceans. The repo includes the models developed for the paper as well as the scripts needed to reproduce model output and figures.

'waveErosion_V3.py' contains a class with the four models developed in this study along with those from Crawford et. al (2024) and Silva et. al (2006).

'waveErosion_demo.ipynb' gives a demo of how to use the model class, along with the scripts necessary to reproduce Figures 2 and 6 from the main text.

To get the model data for creating the rest of the figures, the scripts 'waveTbg_Sensitivity.py', 'waveH_Sensitivity.py', and waveLamb_Sensitivity.py' must be run. 

Code for generating Figure 3 is provided in 'insulatingPower_figure.py'. 

For generating Figure 4, the code is provided in 'integratedMelt_figure.py'. 

The code for Figure 5 is in 'sensitivity_figure.py' and for Figure 7, the code is in 'fitted_functions.py'.
