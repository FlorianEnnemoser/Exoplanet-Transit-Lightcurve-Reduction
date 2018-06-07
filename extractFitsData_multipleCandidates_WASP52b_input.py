# This script was created by Florian Ennemoser
# April 2018
# Karl-Franzens University Graz, Austria
# Department for geophysics, astrophysics and meterology
 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedFormatter, FixedLocator
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats, sigma_clip
from photutils import (DAOStarFinder, aperture_photometry,CircularAperture)
from photutils import find_peaks
from photutils import CircularAnnulus
import logging
import os
import csv

import exo_input_values

############# INITIALIZE LOGGER #############

logger = logging.getLogger(exo_input_values.loggername)
hdlr = logging.FileHandler(exo_input_values.logname)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(exo_input_values.loglevel)
logger.info('SET LOGLEVEL' + '\n' + 'created Logfile: ' + str(hdlr))

############# LOAD IMAGE #############
logger.info('SECTION LOAD IMAGE')

images_array = []

filepath = exo_input_values.lights_filepath

if not os.path.exists(filepath):
    os.makedirs(filepath)

for filename in os.listdir(filepath):
    images_array.append(filepath + '/' + filename)
    logger.info('LOADING IMAGE ' + filename)

print(images_array)

############# CREATE DARK #############
dark_array = []
dark_list = []
dark_path = exo_input_values.darks_filepath

if not os.path.exists(dark_path):
    os.makedirs(dark_path)

for filename in os.listdir(dark_path):
    dark_array.append(dark_path + '/' + filename)
    logger.info('LOADING IMAGE ' + filename)

print(dark_array)

for i in range(np.size(dark_array)):
    fits.info(dark_array[i])
    dark_list.append(fits.getdata(dark_array[i]))
print(dark_list)
dark_cube = np.stack(dark_list,axis=0)

if exo_input_values.combo_master_dark == 1:
    dark_master = np.average(dark_cube,axis=0)

elif exo_input_values.combo_master_dark == 2:
    dark_master = np.median(dark_cube,axis=0)
    
plt.figure(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1])) 
plt.imshow(dark_master, origin='lower', cmap=exo_input_values.colormap);
plt.title('$Dark$ $Master$')
plt.savefig(exo_input_values.save_images_filepath + '/' + "fig_" + 'MasterDark' + '_' + exo_input_values.casename + '.png')

#######################################
############# CREATE BIAS #############
bias_array = []
bias_list = []
bias_path = exo_input_values.bias_filepath

if not os.path.exists(bias_path):
    os.makedirs(bias_path)

for filename in os.listdir(bias_path):
    bias_array.append(bias_path + '/' + filename)
    logger.info('LOADING IMAGE ' + filename)

print(bias_array)

for i in range(np.size(bias_array)):
    fits.info(bias_array[i])
    bias_list.append(fits.getdata(bias_array[i]))
print(bias_list)
bias_cube = np.stack(bias_list,axis=0)

bias_master = np.average(bias_cube,axis=0)

if not os.path.exists(exo_input_values.save_images_filepath):
    os.makedirs(exo_input_values.save_images_filepath)

plt.figure(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1])) 
plt.imshow(np.log10(bias_master), origin='lower', cmap=exo_input_values.colormap);
plt.title('$Bias$ $Master$')
plt.savefig(exo_input_values.save_images_filepath + '/' + "fig_" + 'MasterBias' + '_' + exo_input_values.casename + '.png')

#######################################
logger.info('FINISHED STOREING ' + str(np.size(images_array)) + ' IMAGES IN ARRAY')


############# READ IMAGE #############
logger.info('SECTION READ IMAGE')


def fluxtarget(coord_x_min,coord_x_max,coord_y_min,coord_y_max,name):
    flux = []
    intial_coord_x_min = coord_x_min
    intial_coord_x_max = coord_x_max
    intial_coord_y_min = coord_y_min
    intial_coord_y_max = coord_y_max
    count = 0
    
    for i in range(np.size(images_array)):
        fits.info(images_array[i])
            
        if exo_input_values.dark_red == 1:    
            image_data = fits.getdata(images_array[i])- dark_master
            
        elif exo_input_values.dark_bias_red == 1:
            image_data = fits.getdata(images_array[i])- bias_master - dark_master
            
        elif exo_input_values.no_red == 1: 
            image_data = fits.getdata(images_array[i])

        elif exo_input_values.bias_red == 1 or exo_input_values.bias_sigma[0] == 1 or exo_input_values.bias_min == 1: 
            image_data = fits.getdata(images_array[i]) - bias_master
        
        act_coords = [coord_x_min,coord_x_max,coord_y_min,coord_y_max]
        print('using coordinates: ' + str(act_coords))   
        image_data_cropped = image_data[coord_x_min:coord_x_max,coord_y_min:coord_y_max]
        print(image_data_cropped)
        logger.info('COORDINATES SET AT: ' + str(image_data_cropped))
            
        ############# ENTER DATA INTO LOGFILE #############
        logger.info('SECTION ENTER DATA INTO LOGFILE')
    
        print('Min:', np.min(image_data_cropped))
        print('Max:', np.max(image_data_cropped))
        print('Mean:', np.mean(image_data_cropped))
        print('Stdev:', np.std(image_data_cropped))
    
        logger.info('Min:', np.min(image_data_cropped))
        logger.info('Max:', np.max(image_data_cropped))
        logger.info('Mean:', np.mean(image_data_cropped))
        logger.info('Stdev:', np.std(image_data_cropped))
    
        ####################################################
        
        ############# PRINT FIGURE #############
        logger.info('PRINT FIGURE')

        if exo_input_values.create_lights == 0:
            cmap = exo_input_values.colormap
            
        elif exo_input_values.create_lights == 1:
            plt.style.use(astropy_mpl_style)
            fig, ax = plt.subplots(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1]))
            cmap = exo_input_values.colormap
            ax.imshow(image_data_cropped, cmap=exo_input_values.colormap)   
            ax.autoscale(False)
            plt.title('$original$ $image$' + '\n' + '$'+images_array[i]+'$' + '\n' + '$'+name+'$' ,fontsize=10)
            plt.show()
            fig.savefig(exo_input_values.save_images_filepath + '/' + "fig_" + cmap + "_" + name + '_' + exo_input_values.casename + '_' + str(i+1) + ".png")
            logger.info('SAVED FIGURE ' + str(i+1))
          
        ############# START PHOTOMETRY #############
        logger.info('SECTION REDUCE IMAGE')
        
        image = image_data_cropped.astype(float)
        ############# PERFORM CLIPPING #############
           
        if exo_input_values.median_cut == 1:
            image -= np.median(image)

        elif exo_input_values.average_cut == 1:
            image -= np.average(image)

        elif exo_input_values.min_cut == 1:
            image -= np.min(image)
            
        elif exo_input_values.bias_sigma[0] == 1: 
            image = sigma_clip(image,sigma=exo_input_values.bias_sigma[1], iters=exo_input_values.bias_sigma[2])

        elif exo_input_values.bias_min == 1:
            image -= np.min(image)

        elif exo_input_values.sigma_red == 1:
            image = sigma_clip(image,sigma=exo_input_values.bias_sigma[1], iters=exo_input_values.bias_sigma[2])
        
        bkg_sigma = sigma_clipped_stats(image, sigma=exo_input_values.background_sigma[0], iters=exo_input_values.background_sigma[1]) #3,5
        daofind = DAOStarFinder(fwhm=exo_input_values.fwhm, threshold= exo_input_values.threshold[0] * bkg_sigma[exo_input_values.threshold[1]])
        sources = daofind(image)
        
        logger.info(sources[0])
           
        ############# PERFORM APERTURE PHOTOMETRY #############
        
        positions = (sources['xcentroid'], sources['ycentroid'])
        
        apertures = CircularAperture(positions, r= exo_input_values.aperture)
        annulus_apertures = CircularAnnulus(positions, r_in = exo_input_values.annulus[0] , r_out = exo_input_values.annulus[1] )
        
        apers = [apertures, annulus_apertures]
        
        phot_table = aperture_photometry(image, apers, method = exo_input_values.methods[0] ,subpixels=5)
        
        print(phot_table)
        
        ############# PERFORM LOCAL BACKGROUND SUBSTRACTION ON APERTURE #############
        
        bkg_mean = phot_table['aperture_sum_1']/ annulus_apertures.area()
        bkg_sum = bkg_mean * apertures.area()
        final_sum = phot_table['aperture_sum_0'] - bkg_sum
        phot_table['residual_aperture_sum'] = final_sum
        print(phot_table['residual_aperture_sum'])
        
        flux.append(phot_table['residual_aperture_sum'][0])
        
        print ('FLUX TARGET: ' + str(flux[i]))
        
        ############# PERFORM LOCAL PEAK DETECTION #############
        if exo_input_values.add_local_peaks == 1:
            threshold = bkg_sigma[1] + (10*bkg_sigma[2])
            peaks = find_peaks(image, threshold, box_size = 5)
        
            print(peaks)
        
        ############# RESET COORDINATES IF NECESSARY #############
        print(positions)
        
        
        if i in exo_input_values.i:
            coord_x_min = intial_coord_x_min + exo_input_values.shift_x[count]
            coord_x_max = intial_coord_x_max + exo_input_values.shift_x[count]
            coord_y_min = intial_coord_y_min + exo_input_values.shift_y[count]
            coord_y_max = intial_coord_y_max + exo_input_values.shift_y[count]
            count += 1
         
        ############# PRINT REDUCED IMAGE #############
        if exo_input_values.create_reduced == 1:
            fig, ax = plt.subplots(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1]))
            plt.imshow(image, cmap=exo_input_values.colormap)
            plt.plot(peaks['x_peak'],peaks['y_peak'],ls='none', color='blue',marker='+', ms=10, lw=1.5)
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
            plt.title('$reduced$ $and$ $normalized$ $image$ $with$ $aperture$ $and$ $local$ $peaks$' + '\n' + '$'+images_array[i]+'$' + '\n' + '$'+name+'$',fontsize=10)
            plt.savefig(exo_input_values.save_images_filepath + '/' + "fig_" + cmap + "_" + "reduced_" + name + '_' + exo_input_values.casename + '_' + str(i+1) + ".png")
        
        elif exo_input_values.create_reduced == 0:
            logger.debug('create_reduced = 0 >> do not create reduced image')
            
        image_data = []
        
    plt.style.use(astropy_mpl_style)
    fig, ax = plt.subplots(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1]))
    plt.plot(flux,'b+')
    plt.title('$'+name+'$' + ' $Flux$')
    plt.savefig(exo_input_values.save_images_filepath + '/' + 'graph_' + name + '_flux_' + exo_input_values.casename  + '.png')
    
    return flux
################################
############# MAIN #############

sci_coords = [exo_input_values.sci_coordinates[0],exo_input_values.sci_coordinates[1]]
sci_x_min = sci_coords[0]-exo_input_values.pix_around_star
sci_x_max = sci_coords[0]+exo_input_values.pix_around_star
sci_y_min = sci_coords[1]-exo_input_values.pix_around_star
sci_y_max = sci_coords[1]+exo_input_values.pix_around_star
sci_name = exo_input_values.sci_name

cal1_coords = [exo_input_values.cal1_coordinates[0],exo_input_values.cal1_coordinates[1]]
cal1_x_min = cal1_coords[0]-exo_input_values.pix_around_star
cal1_x_max = cal1_coords[0]+exo_input_values.pix_around_star
cal1_y_min = cal1_coords[1]-exo_input_values.pix_around_star
cal1_y_max = cal1_coords[1]+exo_input_values.pix_around_star
cal1_name = exo_input_values.cal1_name

cal2_coords = [exo_input_values.cal2_coordinates[0],exo_input_values.cal2_coordinates[1]]
cal2_x_min = cal2_coords[0]-exo_input_values.pix_around_star
cal2_x_max = cal2_coords[0]+exo_input_values.pix_around_star
cal2_y_min = cal2_coords[1]-exo_input_values.pix_around_star
cal2_y_max = cal2_coords[1]+exo_input_values.pix_around_star
cal2_name = exo_input_values.cal2_name

############# READ IN CSV FOR COMPARISON #############
if exo_input_values.option_compare == 1:
    flux_sci_2 = []
    flux_cal1_2 = []
    flux_cal2_2 = []
    with open(exo_input_values.compare_csv_filename,'r') as readcsv:
        reader = csv.reader(readcsv)
        
        rownumber = 0
        for row in reader:
            if rownumber >= 1:
                flux_sci_2.append(float(row[2]))
                flux_cal1_2.append(float(row[3]))
                flux_cal2_2.append(float(row[4]))
                
            rownumber += 1
    print(flux_sci_2)
    print(flux_cal1_2) 
    print(flux_cal2_2) 
elif exo_input_values.option_compare == 0:
    logger.debug('option_compare = 0 >> do not compare given csv')


flux_sci = fluxtarget(sci_x_min,sci_x_max,sci_y_min,sci_y_max,sci_name)
flux_cal1 = fluxtarget(cal1_x_min,cal1_x_max,cal1_y_min,cal1_y_max,cal1_name)

flux_cal2 = fluxtarget(cal2_x_min,cal2_x_max,cal2_y_min,cal2_y_max,cal2_name)

if exo_input_values.option_compare == 1:
    flux_sci_2 = fluxtarget(sci_x_min,sci_x_max,sci_y_min,sci_y_max,sci_name)
    flux_cal1_2 = fluxtarget(cal1_x_min,cal1_x_max,cal1_y_min,cal1_y_max,cal1_name)
    flux_cal2_2 = fluxtarget(cal2_x_min,cal2_x_max,cal2_y_min,cal2_y_max,cal2_name)

    flux_sci_cal1_2 = []
    flux_sci_cal2_2 = []
 
flux_sci
flux_cal1
flux_cal2
flux_sci_cal1 = []
flux_sci_cal2 = []



delta_options_sci_cal1 = []
delta_options_sci_cal2 = []

flux_delta_sci = []
flux_delta_cal1 = []
flux_delta_cal2 = []

obs = []
numberobs = []
############# SAVE DATE FROM FITS HEADER TO ARRAY #############
for j in range(np.size(flux_sci)):
    
    image_time = fits.open(images_array[j])
    appobs = image_time[0].header['TIME-OBS']
    obs.append(appobs)
    
############# CREATE CSV #############
if exo_input_values.write_csv == 1:
    with open(exo_input_values.write_csv_name,'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['NUMBER','TIME-OBS','FLUX-SCI','FLUX-CAL1','FLUX-CAL2'])
        for i in range(np.size(flux_sci)):
            numberobs.append(i)
            filewriter.writerow([i,obs[i],flux_sci[i],flux_cal1[i],flux_cal2[i]])

elif exo_input_values.write_csv == 0:
    logger.debug('write_csv = 0 >> do not create csv')

######################################

for x in range(np.size(flux_sci)):
    
    flux_sci_cal1.append((flux_sci[x]/flux_cal1[x]))
    flux_sci_cal2.append((flux_sci[x]/flux_cal2[x]))
    if exo_input_values.option_compare == 1:    
        flux_sci_cal1_2.append((flux_sci_2[x]/flux_cal1_2[x]))
        flux_sci_cal2_2.append((flux_sci_2[x]/flux_cal2_2[x]))
    
        delta_options_sci_cal1.append((flux_sci_cal1[x]-flux_sci_cal1_2[x]))
        delta_options_sci_cal2.append((flux_sci_cal2[x]-flux_sci_cal2_2[x]))
    
        flux_delta_sci.append((flux_sci[x]/flux_sci_2[x]))
        flux_delta_cal1.append((flux_cal1[x]/flux_cal1_2[x]))
        flux_delta_cal2.append((flux_cal2[x]/flux_cal2_2[x]))


############# MEDIAN LINE FLUX #############
        
g1 = []
g_avg1 = []
g2 = []
g_avg2 = []

for j in range(0,20):
    g1.append(flux_sci_cal1[j])
    g2.append(flux_sci_cal2[j])
    
for x in range(np.size(flux_sci)):      
    g_avg1.append(np.median(g1))
    g_avg2.append(np.median(g2))

m1 = []
m_avg1 = []
m2 = []
m_avg2 = []

for j in range(np.size(flux_sci)-20,np.size(flux_sci)):
    m1.append(flux_sci_cal1[j])
    m2.append(flux_sci_cal2[j])

for x in range(np.size(flux_sci)):     
    m_avg1.append(np.median(m1))
    m_avg2.append(np.median(m2))

o1 = []
o_avg1 = []
o2 = []
o_avg2 = []
start_size = int((np.size(flux_sci)/2)-10)
end_size = int((np.size(flux_sci)/2)+10)

for j in range(start_size,end_size):
    o1.append(flux_sci_cal1[j])
    o2.append(flux_sci_cal2[j])
    
for x in range(np.size(flux_sci)):
    o_avg1.append(np.median(o1))
    o_avg2.append(np.median(o2))

print(obs)

############# CALCULATE PLANETARY VALUES #############

delt_gm = (g_avg1[0]+m_avg1[0])/2
delta_gmo_mag = -2.5*np.log10(o_avg1[0]/delt_gm)
print ("mitte med:  " + str(o_avg1[0]))
print ("mwfl:  " + str(delt_gm))
print ("DELTA FLUX MAG:   " + str(delta_gmo_mag))

delt_gmo = delt_gm - o_avg1[0]
rp = np.sqrt(delt_gmo*(exo_input_values.rstar*exo_input_values.rastron)**2)
print ("Radius Planet:   " + str(rp))
print ("Radius Planet (rjups):   " + str(rp/exo_input_values.rjup))

e_rp = np.sqrt(delt_gmo)*(exo_input_values.e_rstar*exo_input_values.rastron)
print ("Error Radius Planet (rjups):   " + str(e_rp/exo_input_values.rastron))

density = (3*exo_input_values.m_planet*exo_input_values.m_jup)/(4*np.pi*rp**3)
print ("Density:    " + str(density))

e_density = np.sqrt(  (abs((-9*exo_input_values.m_planet*exo_input_values.m_jup)/(4*np.pi*rp**4))*e_rp)**2 + abs(3/(4*np.pi*rp**3)*exo_input_values.e_m_planet*exo_input_values.m_jup)      )
print ("Error Density:     " + str(e_density))

rprs = (exo_input_values.rstar*exo_input_values.rastron+rp)**2
sin = (np.sin( (np.pi*exo_input_values.trandur*60)/(exo_input_values.P*86400))*exo_input_values.a*exo_input_values.au)**2
inclination = np.degrees(np.arccos(np.sqrt((rprs-sin)/(exo_input_values.a*exo_input_values.au*10**10))))
print (rprs)
print (sin)
print ("\n")
print("Inclination:    " + str(inclination))

max_rprs = (rp-e_rp+(exo_input_values.rstar-exo_input_values.e_rstar)*exo_input_values.rastron)**2
max_inc = np.degrees(np.arccos(np.sqrt((max_rprs-sin)/(exo_input_values.a*exo_input_values.au*10**10))))
print("Max. Inclination:    " + str(max_inc))

print("Median Flux Sci:    " + str(np.median(flux_sci)))
print("Std Flux Sci/Cal1:    " + str(np.std(flux_sci_cal1)))
#######################################

############# SCI-CAL1 IMAGE #############
plt.style.use(astropy_mpl_style)
fig, ax = plt.subplots(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1]))
plt.plot(obs,flux_sci_cal1,'b+', label='Data Points')
plt.plot(obs,g_avg1,'r--', label='Median Start', alpha=0.8, linewidth=2)
plt.plot(obs,m_avg1,'g--', label='Median End', alpha=0.8, linewidth=2)
plt.plot(obs,o_avg1,'k--', label='Median Middle', alpha=0.8, linewidth=2)
plt.axvline(x=obs[exo_input_values.starttransit_pred],color='grey', linestyle='--')
plt.axvline(x=obs[exo_input_values.endtransit_pred],color='grey', linestyle='--')

 
plt.minorticks_off()

majorLocator = FixedLocator(exo_input_values.x_ax_loc)
majorFormatter = FixedFormatter([obs[exo_input_values.x_ax_loc[0]],obs[exo_input_values.x_ax_loc[1]],obs[exo_input_values.x_ax_loc[2]],obs[exo_input_values.x_ax_loc[3]],obs[exo_input_values.x_ax_loc[4]]])
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)

plt.ylim((exo_input_values.ylim_scical1[0],exo_input_values.ylim_scical1[1]))
plt.title('$Flux$ $Target$ $-$ $Cal1$')
plt.xlabel('$Local$ $Time$ $[UTC]$')
plt.ylabel('$Target/Calibrator$ $[-]$')
plt.legend(loc=2)
plt.savefig(exo_input_values.save_images_filepath + '/' + 'graph_' + 'sci-cal1' + '_' + exo_input_values.casename + '.png')

############# SCI-CAL2 IMAGE #############
plt.style.use(astropy_mpl_style)
fig, ax = plt.subplots(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1]))
plt.plot(obs,flux_sci_cal2,'b.',label='Data Points')
plt.plot(obs,g_avg2,'r-',label='Median', alpha=0.8, linewidth=2)
plt.minorticks_off()
plt.ylim((exo_input_values.ylim_scical2[0],exo_input_values.ylim_scical2[1]))
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)

plt.title('$Flux$ $Target$ $-$ $Cal2$')
plt.xlabel('$Local$ $Time$ $[UTC]$')
plt.ylabel('$Target/Calibrator$ $[-]$')
plt.legend(loc=2)
plt.savefig(exo_input_values.save_images_filepath + '/' + 'graph_' + 'sci-cal2' + '_' + exo_input_values.casename + '.png')

############# DELTA IMAGE #############
if exo_input_values.option_compare == 1:
    plt.style.use(astropy_mpl_style)
    fig, ax = plt.subplots(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1]))
    plt.plot(obs,delta_options_sci_cal1,'b.',label='Delta Option 1-2 Sci Cal1')
    plt.plot(obs,delta_options_sci_cal1,'r-',label='Convolve Option 1-2 Sci Cal1', alpha=0.8, linewidth=2)
    plt.plot(obs,delta_options_sci_cal2,'g.',label='Delta Option 1-2 Sci Cal2')
    plt.plot(obs,delta_options_sci_cal2,'y-',label='Convolve Option 1-2 Sci Cal2', alpha=0.8, linewidth=2)
    plt.minorticks_off()
    plt.ylim((exo_input_values.ylim_delta[0],exo_input_values.ylim_delta[1]))
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    
    plt.title('$Delta$ $Flux$')
    plt.xlabel('$Local$ $Time$ $[UTC]$')
    plt.ylabel('$Delta$ $Sci-Cal$ $[-]$')
    plt.legend(loc=2)
    plt.savefig(exo_input_values.save_images_filepath + '/' + 'graph_' + 'delta' + '_' + exo_input_values.casename + '.png')
    
    ############# DELTA SINGLE IMAGE #############
    
    plt.style.use(astropy_mpl_style)
    fig, ax = plt.subplots(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1]))
    plt.plot(obs,flux_delta_sci,'b.',label='Flux Delta Option 1-2 Sci')
    
    plt.plot(obs,flux_delta_cal1,'r.',label='Flux Delta Option 1-2 Cal1')
    
    plt.plot(obs,flux_delta_cal2,'g.',label='Flux Delta Option 1-2 Cal2')
    
    plt.minorticks_off()
    plt.ylim((exo_input_values.ylim_delta_single[0],exo_input_values.ylim_delta_single[1]))
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    
    plt.title('$Delta$ $Flux$')
    plt.xlabel('$Local$ $Time$ $[UTC]$')
    plt.ylabel('$Delta$ $Sci$ $[-]$')
    plt.legend(loc=2)
    plt.savefig(exo_input_values.save_images_filepath + '/' + 'graph_' + 'delta' + '_sci' + '_' + exo_input_values.casename + '.png')
    
    ############# SCI-CAL1 OPTION 2 IMAGE #############
    plt.style.use(astropy_mpl_style)
    fig, ax = plt.subplots(figsize=(exo_input_values.figsize[0],exo_input_values.figsize[1]))
    plt.plot(obs,flux_sci_cal1_2,'b.', label='Data Points')
    plt.minorticks_off()
    
    majorLocator = MultipleLocator(50)
    majorFormatter = FormatStrFormatter('%d')
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    plt.ylim((exo_input_values.ylim_scical1_csv[0],exo_input_values.ylim_scical1_csv[1]))
    plt.title('$Flux$ $Target$ $-$ $Cal1_2$')
    plt.xlabel('$Local$ $Time$ $[UTC]$')
    plt.ylabel('$Target/Calibrator$ $[-]$')
    plt.legend(loc=2)
    plt.savefig(exo_input_values.save_images_filepath + '/' + 'graph_' + 'sci-cal1_2' + '_' + exo_input_values.casename + '.png')
