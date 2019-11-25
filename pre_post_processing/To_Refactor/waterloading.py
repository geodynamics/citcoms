#!/usr/bin/pyth on2.6

# encoding: utf-8
"""
extract_lith.py

combines cap and opt files to extract the depth to Lithosphere

Created by Nicolas Flament on 2013-02-06.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import os.path
import subprocess

#---- read the data file from user input
#-- exit if file doesn't exist
# inFile = raw_input("Tell me the path to your input file:")

cmd="gmtset COLOR_MODEL RGB BASEMAP_TYPE fancy ANNOT_FONT_SIZE_PRIMARY 12 LABEL_FONT_SIZE 12 HEADER_FONT_SIZE 18 ANNOT_FONT_PRIMARY Helvetica PLOT_DEGREE_FORMAT ddd"
os.system(cmd)

cmd="mkdir PS PNG grids XYZ GIF"
os.system(cmd)

pref1="gld93"
prefix="%(pref1)s" % vars()
topo_dir="/Volumes/Data2/Citcoms-runs/Global/Results/Lith_Def/%(pref1)s/Topo/check" % vars()
grd_pref="%(prefix)sct.topo_corr" % vars()
timefile="/Volumes/Data2/Citcoms-runs/Global/Results/Lith_Def/%(pref1)s/Topo/check/%(prefix)sct.timese" % vars()

topology_dir="/Volumes/Data1/Citcoms-runs/Global/Input/Topologies/20140225_svn252"

filevol="%(prefix)s.check_volume.dat" % vars()
fileSL="%(prefix)s.age_SL.dat" % vars()
VOL=open(filevol,"w")
SL=open(fileSL,"w")

steps = 0
for line in open(timefile): steps += 1
#steps=1
print steps

TF=open(timefile,"r")

for k in range(steps):
    print k
    dum1,dum2=TF.readline().split()
    l=int(dum1)
    age=int(dum2)
    #l=1
    #age=230
    topo_grid="%(topo_dir)s/%(grd_pref)s.%(l)i.grd" % vars()
    final_grid="%(prefix)s.topo_waterloaded.%(l)i.%(age)iMa.grd" % vars()
    if k < 10:
        ps_pref="%(prefix)s.topo_waterloaded.0%(k)i.%(l)i.%(age)iMa" % vars()
    else:
        ps_pref="%(prefix)s.topo_waterloaded.%(k)i.%(l)i.%(age)iMa" % vars()

    psfile="%(ps_pref)s.ps" % vars()

    #psfile="%(prefix)s.topo_final_DEMcpt.%(l)i.%(age)iMa.ps" % vars()

    coasts="/Volumes/Data1/CitcomS-runs/Global/Input/GlobalGrids/20140225/Global_EarthByte_GPlates_Coastlines_2013_2_LowRes/reconstructed_%(age)i.00Ma.xy" % vars()
    print coasts

    subduction_left="%(topology_dir)s/topology_subduction_boundaries_sL_%(age)i.00Ma.xy" % vars()
    subduction_right="%(topology_dir)s/topology_subduction_boundaries_sR_%(age)i.00Ma.xy" % vars()

    #transect_north="/Volumes/Data1/Citcoms-runs/Global/Results/Lith_Def/%(pref1)s/Topo/check/XYZ/transect5S_%(age)i.xy" % vars()
    #transect_south="/Volumes/Data1/Citcoms-runs/Global/Results/Lith_Def/%(pref1)s/Topo/check/XYZ/transect45S_%(age)i.xy" % vars()

    print topo_grid

    if os.path.isfile(topo_grid):
        print "Opening %s" % topo_grid 

    else:
        print "File does not exist - Should exit now"
        #sys.exit(0)

    # multiply the air-loaded topography grid by -1 (presumably used for some
    # kind of correction?)  Does 'topo_corr_inv' mean topography_correction_inverse?
    # Nico - is topo_grid the air-loaded topography in meters?
    ## Yes, topo_grid is the air-loaded topography. If I remember correctly I had to
    ## do this step to find the volume of the ocean basin (the grdvolume commands
    ## I used calculate a volume above rather than below a contour)
    cmd="grdmath %(topo_grid)s -1 MUL = %(prefix)s.topo_corr_inv.%(l)i.grd" % vars()
    #print cmd
    os.system(cmd)
	
    # volume of all ocean basins
    # Nico - what's the reference / resource for this exact value?
    ## Good point, Dan: the ocean volume is not that well constrained
    ## I think I calculated the volume of the oceans based on etopo1
    ## The value is reasonable: In Flament et al. (2008) I had 1.36e18+/-2e17 m^3
    VoTrue=1.33702586599e+18 # units of m^3

    # loading factor is a constant that quantifies the ratio of
    # water-loaded topography to air-loaded topography for a given
    # (radial) normal stress
    # derived from dh_air = dP / (drho_a * g)
    # where drho_a = 3300 - 0
    # dh_water = dP / (drho_w * g)
    # where drho_w = 3300 - 1025 (1025 approx density of seawater)
    # Nico - what's the reference / resource for this exact value?
    ## this loading factor comes out of the assumed densities for water and mantle
    ## I suppose there is no reason this should be hard-coded.
    ## In Flament et al. (2014) I used rho_m=3340 kg m^-3 and rho_water=1030 kg m^-3 
    ## so that Loading_fact=3340/(3340-1030)
    Loading_fact=1.44588744588744588744

    # volume of all ocean basins with water removed
    # i.e. this air-loads the oceans
    VoTarget=VoTrue/Loading_fact
	#print VoTarget, float(VoTarget)

    lim=VoTarget/1e7
	#print lim, float(lim)
    # alpha has not been defined - unsure what this is (so far)
    ## sometimes the iteration process does not converge; adding an aribtrary DC shift (alpha)
    ## usually resolves this
    alpha=0

    # iterative approach will not exactly locate VoTrue,
    # so provide a constrained solution range here
    # using VoMax and VoMin
    VoMax=VoTarget+lim
    VoMin=VoTarget-lim
	#print VoMax, VoMin

	# iteration counters
    # first iteration
    it=0
    # maximum number of iterations
    iterations=100

    inf1="grdinfo %(prefix)s.topo_corr_inv.%(l)i.grd | grep z_min | awk '{print $3}' " % vars()
    inf2="grdinfo %(prefix)s.topo_corr_inv.%(l)i.grd | grep z_min | awk '{print $5}' " % vars()
    # minimum topography value is used as a contour?
    ## that's right, the iteration process needs two values to start off, the further apart the better
    ## cont1 and cont2 are the minimum and maximum bounds for sea level (deepest abyss and highest mountains)
    cont1=float(os.popen(inf1).read())
    # maximum topography value is used as a contour?
    cont2=float(os.popen(inf2).read())
    #print cont1, float(cont1)

    cmd1="grdvolume -C%(cont1)f -S -Rg %(prefix)s.topo_corr_inv.%(l)i.grd | awk '{print $3}' " % vars()
	#print cmd
    # volume within (or above? - check convention) minimum contour level (probably above, always would
    # be zero)
    ## I think this is above and this is why I had to invert the grid in the first place
    v1=float(os.popen(cmd1).read())
	#print v1
    cmd2="grdvolume -C%(cont2)f -S -Rg %(prefix)s.topo_corr_inv.%(l)i.grd | awk '{print $3}' " % vars()
	#print cmd
    # volume contained within maximum contour?
    v2=float(os.popen(cmd2).read())
	#print v2

    contf=0.0
    ## contf will be the returned value. Our first guess is 0.

    # DJB - added comments to this point.  Also made tabs to 4 spaces
    # to stick to python convention

	while it < iterations and (float(contf) == 0.0):
    ## ensuring that contf is within bounds throughout iterations
		if (float(v1) < float(VoMax)) and (float(v1) > float(VoMin)):
			contf=float(cont1)
		elif (float(v2) < float(VoMax)) and (float(v2) > float(VoMin)):
			contf=float(cont2)
    ## below: linear variation of contf between cont1 and cont2
		else:
			cont3=float(cont2)+(1.0+float(alpha))*(float(cont1)-float(cont2))*(float(VoTarget)-float(v2))/(float(v1)-float(v2))
			#print cont3
			if (float(cont3) == float(cont2)):
                                contf=float(cont2)
    ## ensuring that contf is within bounds throughout iterations
			else:
				if (float(cont3) > float(cont2)):
					cont3=float(cont2)
				elif (float(cont3) < float(cont1)):
					cont3=float(cont1)
    ## Calculating the volume of oceans above cont3
				cmd3="grdvolume -C%(cont3)f -S -Rg %(prefix)s.topo_corr_inv.%(l)i.grd | awk '{print $3}' " % vars()
        			v3=float(os.popen(cmd3).read())
        			#print v3,VoMax,VoMin
    ## Checking if that volume v3 meets convergence criterion
				if (float(v3) < float(VoMax)) and (float(v3) > float(VoMin)):
                        		contf=float(cont3)
    ## If not updating either upper bound or lower bound
				elif (float(v3) > float(VoMax)):
					v1=float(v3)
					cont1=float(cont3)
				elif (float(v3) < float(VoMin)):
					v2=float(v3)
					cont2=float(cont3)
				it+=1
				if it == iterations:
					print "Contour loop does not converge - changing alpha"
					alpha=alpha+0.2
					print v3,VoMax,VoMin
                			it=0
					if alpha == 1.2:
						"Contour loop does not converge - exiting now"
						sys.exit(0)
	SL.write("%f %f\n" % (age,contf))
	#print contf

	#cmd="grdtrack %(transect_north)s -G%(final_grid)s -m -Rg -Qn -V | awk '{print $1, $4 }' > transect_north_%(age)i.xyz" % vars()
	#cmd="grdtrack %(transect_south)s -G%(final_grid)s -m -Rg -Qn -V | awk '{print $1, $4 }' > transect_south_%(age)i.xyz" % vars()

	## Adding contf to original, air-loaded topography
	cmd="grdmath -V %(topo_grid)s %(contf)f ADD = %(prefix)s.topo_corr_AL.%(l)i.grd" % vars()
	#print cmd
        os.system(cmd)

	## removing positive values (that should be air-loaded) from that grid
	cmd="grdclip -V %(prefix)s.topo_corr_AL.%(l)i.grd -Sa0/NaN -G%(prefix)s.topo_corr_AL_oceans.%(l)i.grd" % vars()
        #print cmd
        os.system(cmd)
	
	## water-loading of the negative values (oceans)
	cmd="grdmath -V %(prefix)s.topo_corr_AL_oceans.%(l)i.grd %(Loading_fact)f MUL = %(prefix)s.topo_corr_AL_oceans_WL.%(l)i.grd" % vars()
        #print cmd
        os.system(cmd)

	## stitching air-loaded continents and water-loaded oceans together
	cmd="grdmath -V %(prefix)s.topo_corr_AL_oceans_WL.%(l)i.grd %(prefix)s.topo_corr_AL.%(l)i.grd AND = %(final_grid)s" % vars()
        #print cmd
        os.system(cmd)

        ## inverting the stitched grid
	cmd="grdmath -V %(final_grid)s -1 MUL = %(prefix)s.topo_final_inv.%(l)i.%(age)iMa.grd" % vars()
        #print cmd
        os.system(cmd)

        ## calculating the ocean volume of the stitched grid
	cmdf="grdvolume -C0 -S -Rg %(prefix)s.topo_final_inv.%(l)i.%(age)iMa.grd | awk '{print $3}' " % vars()
        #print cmd
        vf=(float(os.popen(cmdf).read())-VoTrue)/VoTrue
        print vf

	## saving ocean volume through time
	VOL.write("%f %f\n" % (age,vf))

	## cleaning up
	cmd="rm -v %(prefix)s.topo_corr_inv.%(l)i.grd %(prefix)s.topo_corr_AL.%(l)i.grd %(prefix)s.topo_corr_AL_oceans.%(l)i.grd %(prefix)s.topo_corr_AL_oceans_WL.%(l)i.grd %(prefix)s.topo_final_inv.%(l)i.%(age)iMa.grd" % vars()
	#print cmd
	os.system(cmd) 

	## below is standard plotting
	## colour palette
	cmd="makecpt -Cpolar -T-6000/6000/500 -D > topo.cpt"
        #print cmd
        os.system(cmd)
	
	##plotting grid
	cmd="grdimage %(final_grid)s -Rg -JW0/20c -Ba30f15:.%(age)i\ Ma: -Ctopo.cpt -X0.75 -Y5.0 -V -K > %(psfile)s" % vars()
	#print cmd
	os.system(cmd)

	## coastlines
	cmd="psxy %(coasts)s -m -W1/black -J -R -K -O -V >> %(psfile)s" % vars()
        #print cmd
        os.system(cmd)

	## SZs
	cmd="psxy -R -J -W2.0p,darkgrey -Sf6p/2plt -K -O -m %(subduction_left)s -Gdarkgrey -V >> %(psfile)s" % vars()
	#print cmd
        os.system(cmd)

	## SZs
	cmd="psxy -R -J -W2.0p,darkgrey -Sf6p/2prt -K -O -m %(subduction_right)s -Gdarkgrey -V >> %(psfile)s" % vars()
	#print cmd
        os.system(cmd)
	
	## Colour scale
	cmd="psscale -Ctopo.cpt -Ba1500f500:Model\ Elevation\ [m]: -D10c/-1c/12c/0.75ch -K -V -O >> %(psfile)s" % vars()
        #print cmd
        os.system(cmd)

	## basemap
	cmd="psbasemap -JX16/9 -R-60/50/-7000/6000 -Ba20f10:Degrees\ of\ longitude:/a1000f500:Model\ total\ elevation\ [m]::.Transects:WSne -X2c -Y-15c -K -O -V >> %(psfile)s" % vars()
        #print cmd
        os.system(cmd)

	## conversion to raster
	cmd="ps2raster %(psfile)s -E300 -A -TG -P -V" % vars()
        #print cmd
        os.system(cmd)

	cmd="convert -verbose %(ps_pref)s.png %(ps_pref)s.gif" %vars()
	os.system(cmd)
	#sys.exit(0)
		
TF.close()
SL.close()
VOL.close()

## GIF animation
cmd="gifsicle -V --delay=80 --loop --colors 256 %(prefix)s.topo_waterloaded*Ma.gif > %(prefix)s.topo_waterloaded_all-times.gif" % vars()
os.system(cmd)

## clean up
cmd="mv *.ps PS/"
os.system(cmd)
cmd="mv *.png PNG/"
os.system(cmd)
cmd="mv *.grd grids/"
os.system(cmd)
cmd="mv *.xyz XYZ/"
os.system(cmd)
cmd="mv *.gif GIF/"
os.system(cmd)

## legacy junk from sample script by someone who actually knows how to use python
def my_function():
	pass
	
def main():
	# comment
    pass

if __name__ == '__main__':
	main()

