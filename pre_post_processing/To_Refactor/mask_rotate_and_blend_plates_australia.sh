#!/bin/sh

gmtset COLOR_MODEL RGB BASEMAP_TYPE fancy ANNOT_FONT_SIZE_PRIMARY 12 LABEL_FONT_SIZE 12 HEADER_FONT_SIZE 18 ANNOT_FONT_PRIMARY Helvetica PLOT_DEGREE_FORMAT ddd

# ------ This script extracts the mean topo from the palaeo-agegrids -------------
# ------ and writes it to the screen  ------------------------------------

mkdir PS PNG GIF

pref1=gld28
pref2=gld27
prefix=${pref1}NLt

folder_topo=/Volumes/Data3/Citcoms-runs/Global/Results/Lith_Def/${pref1}/Topo/nolith
Topology_dir=/Volumes/Data3/Citcoms-runs/Global/Input/Topologies/20130524_svn191
rotation_file=/Users/nflament/Documents/PostDoc/GPlates/SVN/models/Global_Model_WD_Internal_Release_2013.1/Global_EarthByte_TPW_CK95G94_2013.1.rot

time_file=${folder_topo}/${prefix}-all.timese
time_files=${folder_topo}/${prefix}-all.times

rm xmin_all.dat
touch xmin_all.dat
rm xmax_all.dat
touch xmax_all.dat
rm ymin_all.dat
touch ymin_all.dat
rm ymax_all.dat
touch ymax_all.dat
rm proj.dat
touch proj.dat

n=$(wc -l ${time_file} | awk '{ print $1 }')

i=1
#i=$n
while (($i <= $n))
#while (($i <= 1))
do

p=$(wc -l plate_list.dat | awk '{ print $1 }')

touch xmin.dat
touch xmax.dat
touch ymin.dat
touch ymax.dat
touch blend.dat

j=1
#j=$p
while (($j <= $p))
#while (($j <= 1))
do

plate=$(awk '{ if (NR == '$j') print $1 }' plate_list.dat)

echo $plate

folder_plate=/Volumes/Data3/Citcoms-runs/Global/Results/Lith_Def/${pref2}/Topo/nolith/PlateFrame/StaticPolygons/2013.1_svn196/Static_Polygons_2012.1_Plate_${plate}

l=0
age=$(awk '{ if (NR == '$i') print $2 }' ${time_file})
fage=$(awk '{ if (NR == '$i') print $2 }' ${time_files})

# ---- Set paths and prefixes -----

grdspace=0.1

infile=${folder_plate}/reconstructed_${age}.00Ma.xy
infile2=${folder_plate}/reconstructed_0.00Ma.xy
infile3=/Volumes/Data3/Citcoms-runs/Global/Results/Lith_Def/${pref2}/Topo/nolith/PlateFrame/StaticPolygons/2013.1_svn196/Static_Polygons_2012.1_AUS/reconstructed_0.00Ma.xy

maskgrd=${plate}-mask_${age}Ma.grd

rotated_grid_pref=${prefix}.${plate}_plate_frame.$l.${age}Ma
rotated_grid=${rotated_grid_pref}.grd

masked_rotated_grid=${rotated_grid_pref}_masked.grd

blended_grid_pref=${prefix}.blended_plate_frame.$l.${age}Ma
blended_grid=${blended_grid_pref}.grd
final_grid=${prefix}.blended_resurfaced_masked_plate_frame.$l.${age}Ma.grd

topogrd=${folder_topo}/${prefix}-${age}.topo_corr.$l.grd

Coasts=/Volumes/Data3/Citcoms-runs/Global/Input/GlobalGrids/20130524_svn191/Coastlines_2012.2/reconstructed_${age}.00Ma.xy
subduction_left=${Topology_dir}/topology_subduction_boundaries_sL_${age}.00Ma.xy
subduction_right=${Topology_dir}/topology_subduction_boundaries_sR_${age}.00Ma.xy

# ---- Set up frame index

if [ $i -lt 10 ];
        then
        k=0${i}
        echo $i, $k
else
        k=${i}
        echo $i, $k
fi

ps_pref=${prefix}.blended_plate_frame_${k}_${l}.${age}Ma
psfile=${ps_pref}.ps

### ---- Calculate and format rotation ---- ###

pole=$(gplates equivalent-total-rotation -r ${rotation_file} -t ${fage} -a ${plate} -p 001 | sed 's/,//g' | awk '{ print $2"/"substr($1,2)"/"$3 }' | sed 's/)*$//')

echo $pole

### ---- Mask and rotate topography for plate ---###

grdrotater ${topogrd} -T${pole} -F${infile} -V -Q -Rg -G${rotated_grid}

### ---- Calculate extent ----###

xmin=$(grdinfo ${rotated_grid} | grep x_min | awk '{print $3}')
xmine=$(grdinfo ${rotated_grid} | grep x_min | awk '{print int($3)}')
xmax=$(grdinfo ${rotated_grid} | grep x_min | awk '{print $5}')
ymin=$(grdinfo ${rotated_grid} | grep y_min | awk '{print $3}')
ymine=$(grdinfo ${rotated_grid} | grep y_min | awk '{print int($3)}')
ymax=$(grdinfo ${rotated_grid} | grep y_min | awk '{print $5}')

if [ $ymine -lt 0 ];
        then
        ymin2=$(grdinfo ${rotated_grid} | grep y_min | awk '{print -$3}')
        ycent=$(echo "${ymin}+(${ymax}+${ymin2})/2" | bc -l)
else
        ycent=$(echo "${ymin}+(${ymax}-${ymin})/2" | bc -l)
fi

if [ $xmine -lt 0 ];
        then
        xmin2=$(grdinfo ${rotated_grid} | grep x_min | awk '{print -$3}')
        xcent=$(echo "${xmin}+(${xmax}+${xmin2})/2" | bc -l)
        x1=$(echo "${xmin}+(${xmax}+${xmin2})/3" | bc -l)
        x2=$(echo "${xmin}+2*(${xmax}+${xmin2})/3" | bc -l)
else
        xcent=$(echo "${xmin}+(${xmax}-${xmin})/2" | bc -l)
        x1=$(echo "${xmin}+(${xmax}-${xmin})/3" | bc -l)
        x2=$(echo "${xmin}+2*(${xmax}-${xmin})/3" | bc -l)
fi

region=${xmin}/${ymin}/${xmax}/${ymax}r
proj=B${xcent}/${ycent}/${x1}/${x2}/18c

echo $proj >> proj.dat

echo ${masked_rotated_grid} -R${xmin}/${xmax}/${ymin}/${ymax} 1 >> blend.dat

echo ${xmin} >> xmin.dat
echo ${xmax} >> xmax.dat
echo ${ymin} >> ymin.dat
echo ${ymax} >> ymax.dat

echo ${xmin} >> xmin_all.dat
echo ${xmax} >> xmax_all.dat
echo ${ymin} >> ymin_all.dat
echo ${ymax} >> ymax_all.dat

echo ${region}

echo ${proj}

### ---- Create masking grid for plate ----###

awk '{
if ($1 !~ ">" && $1 < 0)
        print $1+360, $2;
else
        print $0;
}' $infile2 > tmp_mask.xyz

        if [ $age -eq 0 ]
                then
                /usr/local/GMT/GMT4.3.1/bin/grdmask tmp_mask.xyz -R${region} -M -I$grdspace -NNaN/1/1 -V -G$maskgrd
        else
                grdmask tmp_mask.xyz -R${region} -m -I$grdspace -NNaN/1/1 -V -G$maskgrd
        fi

# ---- Create masked grid -----

grdmath ${rotated_grid} $maskgrd OR = ${masked_rotated_grid} -V

rm $maskgrd ${rotated_grid}
rm tmp_mask.xyz

j=$(($j + 1 ))
done

### ---- Calculate extent of blended grid ----###

xminb=$(minmax xmin.dat | awk '{ print $NF }' | awk 'BEGIN { FS = "/"} ; {print $1}' | sed 's/<//g')
yminb=$(minmax ymin.dat | awk '{ print $NF }' | awk 'BEGIN { FS = "/"} ; {print $1}' | sed 's/<//g')
xmaxb=$(minmax xmax.dat | awk '{ print $NF }' | awk 'BEGIN { FS = "/"} ; {print $2}' | sed 's/>//g')
ymaxb=$(minmax ymax.dat | awk '{ print $NF }' | awk 'BEGIN { FS = "/"} ; {print $2}' | sed 's/>//g')

regionb=${xminb}/${xmaxb}/${yminb}/${ymaxb}

### ---- Blend grids ----###

grdblend blend.dat -G${blended_grid} -I$grdspace -R${regionb} -V

rm ${prefix}.8??_plate_frame.$l.${age}Ma_masked.grd

# ---- Plot blended topography grid -----

regionb2=${xminb}/${yminb}/${xmaxb}/${ymaxb}r
projb=M18c

cpt=DT.cpt
makecpt -Cpolar -T-1200/1200/200 -D > $cpt

grdimage -C$cpt ${blended_grid} -J${projb} -P -R${regionb} -Y4 -B5:."$age Ma": -V -K > $psfile
psxy ${infile3} -m -W2,darkgrey -J -R -K -O -V >> $psfile
pscoast -Dl -R -J -K -O -V -W4,black >> $psfile
psscale -C$cpt -Ba200f100:"Dynamic topography [m]": -D9/-1/18/0.5h -V -O >> $psfile

ps2raster $psfile -E300 -A -TG -P -V

convert -verbose ${ps_pref}.png ${ps_pref}.gif

# ----- Clean up -----

rm xmin.dat
rm xmax.dat
rm ymin.dat
rm ymax.dat
rm blend.dat

# ----- End loop -----

i=$(($i + 1 ))
done

gifsicle -V --delay=80 --loop --colors 256 ${prefix}.blended_plate_frame*Ma.gif > ${prefix}.blended_plate_frame_all-times.gif

mv *.ps PS
mv *.png PNG
mv *.gif GIF

./reinterpolate_and_mask.sh
