#!/bin/sh

gmtset COLOR_MODEL RGB BASEMAP_TYPE fancy ANNOT_FONT_SIZE_PRIMARY 12 LABEL_FONT_SIZE 12 HEADER_FONT_SIZE 18 ANNOT_FONT_PRIMARY Helvetica PLOT_DEGREE_FORMAT ddd

mkdir TZ PNG PS GIF

pref1=gld28
pref2=gld27
prefix=${pref1}NLt

folder_topo=/Volumes/Data3/Citcoms-runs/Global/Results/Lith_Def/${pref1}/Topo/nolith
Topology_dir=/Volumes/Data3/Citcoms-runs/Global/Input/Topologies/20130524_svn191
#FALK=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Jones2004/Falkland14051A_WD/reconstructed_0.00Ma.xy

#rm FALK_DT.tz
#touch FALK_DT.tz

time_file=${folder_topo}/${prefix}-all.timese
time_files=${folder_topo}/${prefix}-all.times

grdspace=0.1

infile3=/Volumes/Data3/Citcoms-runs/Global/Results/Lith_Def/${pref2}/Topo/nolith/PlateFrame/StaticPolygons/2013.1_svn196/Static_Polygons_2012.1_AUS/reconstructed_0.00Ma.xy

### ---- Calculate extent of final grids ----###

xminb=$(minmax xmin_all.dat | awk '{ print $NF }' | awk 'BEGIN { FS = "/"} ; {print $1}' | sed 's/<//g')
yminb=$(minmax ymin_all.dat | awk '{ print $NF }' | awk 'BEGIN { FS = "/"} ; {print $1}' | sed 's/<//g')
xmaxb=$(minmax xmax_all.dat | awk '{ print $NF }' | awk 'BEGIN { FS = "/"} ; {print $2}' | sed 's/>//g')
ymaxb=$(minmax ymax_all.dat | awk '{ print $NF }' | awk 'BEGIN { FS = "/"} ; {print $2}' | sed 's/>//g')

regionb=${xminb}/${xmaxb}/${yminb}/${ymaxb}

n=$(wc -l ${time_file} | awk '{ print $1 }')

i=1
#i=$n
while (($i <= $n))
#while (($i <= 1))
do

l=0
age=$(awk '{ if (NR == '$i') print $2 }' ${time_file})
fage=$(awk '{ if (NR == '$i') print $2 }' ${time_files})

if [ $age -lt 10 ];
        then
        age2=0${age}
        echo $age, $age2
else
        age2=$age
fi

# ---- Set paths and prefixes -----

blended_grid_pref=${prefix}.blended_plate_frame.$l.${age}Ma
blended_grid=${blended_grid_pref}.grd
final_grid=${prefix}.blended_resurfaced_masked_plate_frame.$l.${age}Ma.grd

maskallgrd=all-mask_${age}Ma.grd

# ---- Set up frame index

if [ $i -lt 10 ];
        then
        k=0${i}
        echo $i, $k
else
        k=${i}
        echo $i, $k
fi

ps_pref=${prefix}.blended_resurfaced_masked_plate_frame_${k}_${l}.${age}Ma
psfile=${ps_pref}.ps


### ---- Re-interpolate to get rid of edge effects ---- ##

if [ ! -f ${final_grid} ];
        then

	grd2xyz ${blended_grid} -R${regionb} -V > ${blended_grid_pref}.xyz

	blockmedian ${blended_grid_pref}.xyz -R${regionb} -I$grdspace -V > ${blended_grid_pref}.median
	surface ${blended_grid_pref}.median -I$grdspace -R${regionb} -T0.5 -V -G${blended_grid_pref}_resurfaced.grd

	rm ${blended_grid_pref}.xyz ${blended_grid_pref}.median

	### ---- Create masking grid for all plates ----###

awk '{
if ($1 !~ ">" && $1 < 0)
        print $1+360, $2;
else
        print $0;
}' $infile3 > tmp_mask.xyz

        	if [ $age -eq 0 ]
                	then
                	/usr/local/GMT/GMT4.3.1/bin/grdmask tmp_mask.xyz -R${regionb} -M -I$grdspace -NNaN/1/1 -V -G$maskallgrd
        	else
                	grdmask tmp_mask.xyz -R${regionb} -m -I$grdspace -NNaN/1/1 -V -G$maskallgrd
        	fi

	# ---- Create masked grid -----

	grdmath ${blended_grid_pref}_resurfaced.grd $maskallgrd OR = ${final_grid} -V

	rm ${blended_grid_pref}_resurfaced.grd $maskallgrd tmp_mask.xyz

fi

### ---- Extract point location ---- ###

#awk '{
#if ($1 !~ ">" && $1 < 0)
#        print $1+360, $2;
#else
#        print $0;
#}' ${FALK} > tmp_sample.xyz
#
#point1r=$(grdtrack tmp_sample.xyz -G${final_grid} -m -R${regionb} -Qn -V | awk '{if ($1 !~ ">") print $3}')
#echo $fage $point1r >> FALK_DT.tz
#
#rm tmp_sample.xyz

# ---- Plot blended topography grid -----

regionb2=${xminb}/${yminb}/${xmaxb}/${ymaxb}r
projb=M18c

proj=$(tail -1 proj.dat)

cpt=DT.cpt
makecpt -Cpolar -T-1200/1200/200 -D > $cpt

grdimage -C$cpt ${final_grid} -J${projb} -P -R${regionb} -Y4 -B5:."$age Ma": -V -K > $psfile
psxy ${infile3} -m -W2,darkgrey -J -R -K -O -V >> $psfile
pscoast -Dl -R -J -K -O -V -W4,black >> $psfile
psscale -C$cpt -Ba200f100:"Dynamic topography [m]": -D9/-1/18/0.5h -V -O >> $psfile

ps2raster $psfile -E300 -A -TG -P -V

convert -verbose ${ps_pref}.png ${ps_pref}.gif

# ----- End loop -----

i=$(($i + 1 ))
done


#FALK_pres_DT=$(awk 'END{print $2}' FALK_DT.tz)
#awk '{ print $1, $2-'${FALK_pres_DT}' }' FALK_DT.tz > FALK_DT.wrt.pres.tz

#awk '{ if ($1 < 150 ) print $0 }' FALK_DT.tz > FALK_DT_pj.tz
#awk '{ if ($1 < 150 ) print $0 }' FALK_DT.wrt.pres.tz > FALK_DT.wrt.pres_pj.tz

gifsicle -V --delay=80 --loop --colors 256 ${prefix}.blended_resurfaced_masked_plate_frame*Ma.gif > ${prefix}.blended_resurfaced_masked_plate_frame_all-times.gif


mv *.ps PS
mv *.png PNG
mv *.gif GIF
mv *.tz TZ/

./extract_trends.sh
