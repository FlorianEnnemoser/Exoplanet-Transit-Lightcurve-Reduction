# This script was created by Florian Ennemoser
# April 2018
# Karl-Franzens University Graz, Austria
# Institute of Geophysics, Astrophysics and Meteorology

import logging

############# INPUT VALUES #############

## uncomment to activate features. may only be one active at the time.


############# LOG LEVEL #############
loggername = 'exolog'
logname = 'exo_console.log'
loglevel = logging.ERROR
#loglevel = logging.DEBUG
#loglevel = logging.INFO
#loglevel = logging.WARN


############# FILE PATHS #############

lights_filepath = '_WASP52b/WASP52b'
darks_filepath = '_WASP52b/Dark'
bias_filepath = '_WASP52b/Bias'
save_images_filepath = '_WASP52b/images'

############# STAR INFO #############
#sci_coordinates = [419,759]
#cal1_coordinates = [485,1436]
#cal2_coordinates = [677,586]

sci_name = 'WASP52'
cal1_name = 'Calibrator_1'
cal2_name = 'Calibrator_2'

# FIGURES
casename = 'WASP52' #added info-text to all(!) figures
## FIGSIZE
# xy should be equal
figsize = [15,10]
# GRAPHS

## YLIMS
#ylim_scical1 = [0.45,0.55]
#ylim_scical2 = [-2,2]
ylim_delta = [-0.15,0.15]
ylim_delta_single = [-0.15,0.15]

ylim_scical1_csv = [0.86,0.96]

## CMAP
colormap = 'gray_r'
#colormap = 'inferno'

#######################
# CALCULATION OF MASTER DARK AND MASTER BIAS
#Use 1 for average, use 2 for median
combo_master_dark = 2
combo_master_bias = 2
#######################

## IMAGE ATTRIBUTES
pix_around_star = 25 #pixels around designated stars to be cut (square)
## LIGHT IMAGE
create_lights = 0  # 0 = false, 1 = true

## REDUCED IMAGE
create_reduced = 0 # 0 = false, 1 = true
add_local_peaks = 1 # 0 = false, 1 = true

# REDUCTION METHODS LIGHTS
# only reduced images are evaluated!
##SIMPLE CUT #every activation subtracts the lights flux-count
median_cut = 0      # 0 = false, 1 = true
average_cut = 0     # 0 = false, 1 = true
min_cut = 0         # 0 = false, 1 = true

# INTIAL REDUCTION METHODS LIGHTS
# at least one has to be true (1)
no_red = 0 # 0 = false, 1 = true ,no reduction
## DARK + BIAS
dark_bias_red = 0   # 0 = false, 1 = true
## DARK
dark_red = 1        # 0 = false, 1 = true
## BIAS
bias_red = 0        # 0 = false, 1 = true
## BIAS + SIGMACLIP
bias_sigma = [0,3,1]  # 0 = false, 1 = true ; sigmas ; iters
## BIAS + MIN
bias_min = 0        # 0 = false, 1 = true
## SIGMA
sigma_red = [0,1,1]   # 0 = false, 1 = true  ; sigmas ; iters


# BACKGROUND ESTIMATION VALUES
background_sigma = [3,5] #sigmas , iters

# DAOFIND
fwhm = 3.
threshold = [20.,2] #times ; 0 = mean, 1 = median, 2 = stddev #20,2.

# PHOTOMETRY
aperture = 4.
annulus = [6.,8.] #inner, outer radius
methods = ['subpixel']

#COORDS
# since tracking is still not working because i'm too stupid to get it to work,
# put image-number and delta coords where to shift here
#i = [49,90,120,189,240]
#shift_x = [20,40,45,65,95]
#shift_y = [-12,-20,-30,-40,-50]
#x_ax_loc = [0,55,110,165,220] # max. 5 values
starttransit_pred = 0
endtransit_pred = 90

#COMPARE AND WRITE CSV
write_csv = 1       # 0 = false, 1 = true
write_csv_name = 'WASP52b-std.csv'

option_compare = 0  # 0 = false, 1 = true
compare_csv_filename = '_HATP19_light-dark.csv'


# STAR SYSTEM DATA
# URL = http://exoplanet.eu/catalog/tres-5_/
pred_start = '21:48:00'
pred_end = '23:36:00'
rstar = 0.79
e_rstar = 0.02
a = 0.0272
P = 1.74978
m_planet = 0.46
e_m_planet = 0.02
trandur = 110

########################################
####### DO NOT EDIT BEYOND HERE ########
########################################
rjup = 69911000
rastron = 695508000
au = 149597870700
den_jup = 1333
m_jup = 1.898*10**27

#WASP52:

sci_coordinates = [595,705]
cal1_coordinates = [425,210]
cal2_coordinates = [240,437]
ylim_scical1 = [0.74,0.85]
ylim_scical2 = [1.8,2]
ylim_delta = [-0.15,0.15]
ylim_delta_single = [-0.15,0.15]
i = [50,80,105,124,139,152]
shift_x = [30,55,80,100,120,145]
shift_y = [-5,-15,-25,-30,-30,-40]
x_ax_loc = [0,40,81,120,162]
#rstar = 0.79
#e_rstar = 0.02
#a = 0.0272
#P = 1.74978
#m_planet = 0.46
#e_m_planet = 0.02
#trandur = 110

#HATP19:
#sci_coordinates = [447,510]
#cal1_coordinates = [357,425]
#cal2_coordinates = [72,850]
#ylim_scical1 = [0.86,0.96]
#ylim_scical2 = [1.1,1.24]
#ylim_delta = [-0.15,0.15]
#ylim_delta_single = [-0.15,0.15]
#
#ylim_scical1_csv = [0.86,0.96]

#i = [63,109,137,156]
#shift_x = [0,35,70,100]
#shift_y = [-30,-50,-70,-70]
#x_ax_loc = [0,40,81,120,162]
#rstar = 0.82
#e_rstar = 0.048
#a = 0.0466
#P = 4.00878
#m_planet = 0.292
#e_m_planet = 0.018
#trandur = 180

###TRES5
#sci_coordinates = [349,511]
#cal1_coordinates = [195,411]
#cal2_coordinates = [434,946]
#
#ylim_scical1 = [0.95,1.02]
#ylim_scical2 = [1.18,1.28]
#ylim_delta = [-0.15,0.15]
#ylim_delta_single = [-0.15,0.15]
#
#ylim_scical1_csv = [0.86,0.96]
#
#i = [21,44,76,99,500]
#shift_x = [30,60,90,120,145]
#shift_y = [-20,-20,-50,-50,-40]
#x_ax_loc = [0,25,50,75,103] 

# URL = http://exoplanet.eu/catalog/tres-5_/

#rstar = 0.866
#e_rstar = 0.013
#a = 0.02446
#P = 1.48224
#m_planet = 1.778
#e_m_planet = 0.063
#trandur = 115
###

#KELT16
#i = [11,25,37,51,63,69,89,90,107,127,146,170]
#shift_x = [35,-25,5,35,50,65,95,30,60,90,125,160]
#shift_y = [-10,-10,-10,-10,-10,-20,-20,-20,-30,-20,-30,-30]
#sci_coordinates = [426,541]
#cal1_coordinates = [474,538]
#cal2_coordinates = [450,786]
#ylim_scical1 = [1.35,1.5]
#ylim_scical2 = [1.15,1.3]
