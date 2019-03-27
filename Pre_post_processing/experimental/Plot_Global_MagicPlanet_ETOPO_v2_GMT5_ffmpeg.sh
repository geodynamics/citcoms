#!/bin/bash
#gmtset COLOR_MODEL RGB LABEL_FONT_SIZE 10p ANNOT_FONT_SIZE 8p MEASURE_UNIT cm BASEMAP_TYPE plain PLOT_DEGREE_FORMAT dddF

rm aaa bbb ccc *.png *.ps *.eps
# gmtswitch GMT4.5.8 # Switch to GMT5

gmtset COLOR_MODEL RGB LABEL_FONT_SIZE 10p ANNOT_FONT_SIZE 8p MEASURE_UNIT cm BASEMAP_TYPE plain PLOT_DEGREE_FORMAT dddF

# ---- Initiates basic parameters that do not depend on the $age variable 
mkdir PNG

central_meridian=180
cpt=age.cpt

# Projection
frame=d
width=15c
#proj=G0/-90/90/$width
#proj=X$width
proj=X15c/11.25c
scale_1=${width}
scalepar="-D7.5/-0.4/15.0/0.5h"
mesh=33mesh
vel_cpt=plate_velocity_sz.cpt

plate_colour=new_plateID.cpt

# ---- Set recontime loop
age=0
max_age=200

  while (( $age <= $max_age ))
      do
		
	 #  ### Make velocity grids 
	 #  rm ${age}_vel.dat
	 #  touch ${age}_vel.dat
     # 
	 #  		cap=0
	 #  		while (( $cap <= 11 ))
     # 
	 #  		do
     # 
	 #  		cat ${mesh}/bvel${age}.${cap}.xy >> ${age}_vel.dat
     # 
	 #  		echo "working on " ${mesh}/bvel${age}.${cap}.xy
     # 
     # 
	 #  		cap=$(($cap + 1))	
	 #  		done
     # 
	 #  awk '{print $2, $1, ($4)*10}' ${age}_vel.dat > ${age}_grid_vels.dat
	 #  
	 #  blockmedian ${age}_grid_vels.dat -Rd -I5m -V > ${age}_grid_vels.median
     # 
	 #  surface ${age}_grid_vels.median -Rd -I5m -V -G${age}_vels.grd
	 #  
	 #  ### Make velocity grids 

vel_grd=${age}_vels.grd

# ---- Input parameters dependant on the $age variable

# Input grid file located on Ebyte4
grdfile=/Volumes/LaCie/AgeGrid/20111110_Seton_etal_ESR/agegrid_final_mask_${age}.grd
hotspots=HS/reconstructed_0.00Ma.xy
LIPs=LIPs/reconstructed_${age}.00Ma.xy

# grdfile=/Volumes/izanagi/Agegrids/20120621/Mask/agegrid_final_mask_${age}.grd

# Input coastline directory generated using GPlates "Export Animation" tool
csfilexy=Coastlines/reconstructed_${age}.00Ma.xy
coastlines_polygons=Coastlines_Polygons/reconstructed_${age}.00Ma.gmt
FZs=FZ/reconstructed_${age}.00Ma.xy
subduction_left=Polygons/topology_subduction_boundaries_sL_${age}.00Ma.xy
subduction_right=Polygons/topology_subduction_boundaries_sR_${age}.00Ma.xy

# Input polygon and velocities directories with relevant date-stamps (see above)
input_polygons=Polygons/topology_platepolygons_${age}.00Ma.xy
input_velocity=6mesh/bvel${age}.*.xy

etopo=etopo_reconstructed/raster_data_ETOPO_Bedrock_NETCDF_${age}.00Ma.grd
continents=continents/reconstructed_${age}.00Ma.gmt
etopo_cpt=DEM_poster.cpt

# Output filenames
timestep=$( echo " 200 - $age" | bc )
echo "Age is " $age " Ma"
echo "Frame is " $timestep
timestamp=$(printf "%04d" $timestep)

echo "Timestamp is " $timestamp

name=EarthByte_MagicPlanet_ETOPO_v2_Seton_etal_2012_${timestamp}
outfile=${name}.ps

	  #   ### AWK STUFF - This convert a GMT5 file so that the segment delimiter (">") is followed by a -Z$colour so that psxy knows what colour to use to shade your polygons by Plate ID
	  #   
	  #   infile=Coastlines/reconstructed_${age}.00Ma.gmt
	  #   
	  #   awk '{if ($1 != ">" ) print $0 }' $infile > aaa 
	  #   
	  #   awk -F"[ |]+" '{if ($1 == "#" && $2 == "@D0") print ">-Z"$5
	  #   else print $1, $2 
	  #   }' aaa > bbb  
	  #   
	  #   awk '{ if ( $1 != "#") print $0}' bbb > ccc
	  #   
	  #   ### Extract all Plate IDs used 
	  #   
	  #   ## awk -F"[ |]+" '{if ($1 == "#" && $2 == "@D0") print $5 }' aaa > ddd
	  #   ## 
	  #   ## sort -n ddd > eee
	  #   ## uniq eee > fff
	  #   
	  #   ### END OF AWK STUFF

# Plots basemap 
# psbasemap -R$frame -J$proj -Ba30f15 -Y5c -P -K > $outfile

# grdimage -R$frame -J$proj $vel_grd -C$vel_cpt -K -O -V >> $outfile

#gmt grdgradient $grdfile -A45 -fg -Ne0.1 -Gage_${age}.grad -V
# gmt grdgradient $etopo -A45 -Getopo_${age}.grad -V

gmt grdimage -R$frame -J$proj $grdfile -C$cpt -Y5c -P -K -V > $outfile

gmt psclip $continents -R${frame} -J${proj} -O -V -K >> $outfile
gmt grdimage -R -J $etopo -C$etopo_cpt -K -O -V >> $outfile 
gmt psclip $continents -R${frame} -J${proj} -O -V -K -C >> $outfile

gmt psclip clip.txt -R${frame} -J${proj} -O -V -K >> $outfile

# Plots coastline outlines (GMT xy) which have been generated using GPlates "Export Animation" feature

# psxy -R$frame -J$proj -W0.5p,black -K -O  $GI -V >> $outfile

# gmt psxy -R${frame} -J${proj} $coastlines_polygons -K -O -V -Gnavajowhite4 >> $outfile

# gmt psxy -R${frame} -J${proj} $csfilexy -K -O -V -W0.5p,black >> $outfile

gmt psxy -R${frame} -J${proj} $FZs  -K -O -V -W0.2p,black >> $outfile

gmt psxy -R${frame} -J${proj} $LIPs  -K -O -V -G40 >> $outfile

# psxy -R${frame} -J${proj} ccc -Ba30g30 -C${plate_colour} -K -O -L -V >> $outfile

#  # Plots polygon outlines
#  gmt psxy -R$frame -J$proj  -W2.0p  $input_polygons -K -O -V >> $outfile
#  
#  # Plot subduction zones
#  
#  gmt psxy -R$frame -J$proj -W2.0p,magenta  -Sf8p/1.5plt -K -O  ${subduction_left} -V >> $outfile
#  gmt psxy -R$frame -J$proj -W2.0p,magenta  -Sf8p/1.5prt -K -O  ${subduction_right} -V >> $outfile

# gmt psxy -R${frame} -J${proj} HS/reconstructed_0.00Ma.xy -Sa10.0p -Gred -W0.5p,black  -K -O -V >> $outfile

# Plots velocity vectors
gmt psxy -R$frame -J$proj -W0.3p $input_velocity -SV0.1c+e+g -: -G0 -K -O -V >> $outfile

gmt psclip clip.txt -R${frame} -J${proj} -O -V -K -C >> $outfile

echo 10 10 $age Ma | gmt pstext -R -J -F+f12,Helvetica,black -W1.0p,black -Gwhite -O >> $outfile

# Converts the PS file to raster
gmt ps2raster $outfile -A -E300 -Tj -P

convert -resize 1024 ${name}.jpg MagicPlanet_${name}.jpg 

age=$(($age + 1))
done
#mv *.png PNG
# ffmpeg -y -r 5 -i PNG/EarthByte_Seton_etal_2012_%04d.png -b 4800k -vcodec mpeg4 EarthByte_Seton_etal_2012.mp4
