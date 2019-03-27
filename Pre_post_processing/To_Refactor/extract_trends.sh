#!/bin/sh

gmtset COLOR_MODEL RGB BASEMAP_TYPE fancy ANNOT_FONT_SIZE_PRIMARY 12 LABEL_FONT_SIZE 12 HEADER_FONT_SIZE 18 ANNOT_FONT_PRIMARY Helvetica PLOT_DEGREE_FORMAT ddd

mkdir PS PNG GIF TZ

pref1=gld28
pref2=gld27
prefix=${pref1}NLt

grdspace=0.1

folder_topo=/Volumes/Data3/Citcoms-runs/Global/Results/Lith_Def/${pref1}/Topo/nolith
infile3=/Volumes/Data3/Citcoms-runs/Global/Results/Lith_Def/${pref2}/Topo/nolith/PlateFrame/StaticPolygons/2013.1_svn196/Static_Polygons_2012.1_AUS/reconstructed_0.00Ma.xy

#FALK=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Jones2004/Falkland14051A_WD/reconstructed_0.00Ma.xy

#rm FALK_DTrate.tz
#touch FALK_DTrate.tz
#rm FALK_cummulativeDT.tz
#touch FALK_cummulativeDT.tz

time_file=${folder_topo}/${prefix}-all.timese
time_files=${folder_topo}/${prefix}-all.times

topo_230=${prefix}.blended_resurfaced_masked_plate_frame.0.230Ma.grd

ls -rtl ${topo_230}

### ---- Calculate extent ----###

xmin=$(grdinfo ${topo_230} | grep x_min | awk '{print $3}')
xmax=$(grdinfo ${topo_230} | grep x_min | awk '{print $5}')
ymin=$(grdinfo ${topo_230} | grep y_min | awk '{print $3}')
ymax=$(grdinfo ${topo_230} | grep y_min | awk '{print $5}')

region=${xmin}/${xmax}/${ymin}/${ymax}
regionb=${xmin}/${ymin}/${xmax}/${ymax}r

### ---- Create masking grid for all plates ----###

awk '{
if ($1 !~ ">" && $1 < 0)
        print $1+360, $2;
else
        print $0;
}' $infile3 > tmp_mask.xyz


/usr/local/GMT/GMT4.3.1/bin/grdmask tmp_mask.xyz -R${region} -M -I$grdspace -NNaN/1/1 -V -Gmaskall.grd

rm tmp_mask.xyz

### ---- Loop over time ---- ###

n=$(wc -l ${time_file} | awk '{ print $1 }')

#i=1
i=1
while (($i <= $n))
#while (($i <= 22))
do

j=$(($i + 1 ))

a=$(awk '{ if (NR == '$i') print $2 }' ${time_file})
a2=$(awk '{ if (NR == '$j') print $2 }' ${time_file})
a3=$(echo "(${a}+${a2})/2" | bc -l | awk '{print int($1)}')
age=$a

as=$(awk '{ if (NR == '$i') print $2 }' ${time_files})
as2=$(awk '{ if (NR == '$j') print $2 }' ${time_files})
as3=$(echo "(${as}+${as2})/2" | bc -l)
as4=$(echo "(${as}-${as2})" | bc -l)
fage=$as

topo_old=${prefix}.blended_resurfaced_masked_plate_frame.0.${a}Ma.grd
topo_young=${prefix}.blended_resurfaced_masked_plate_frame.0.${a2}Ma.grd

topo_rate_inc_pref=${prefix}.DT_Australia.rate_${a3}Ma
topo_rate_inc=${topo_rate_inc_pref}.grd
topo_diff_230_pref=${prefix}.DT_Australia.diff_${a}Ma-150Ma
topo_diff_230=${topo_diff_230_pref}.grd

# ---- Set up frame index

if [ $i -lt 10 ];
        then
        k=0${i}
        echo $i, $k
else
        k=${i}
        echo $i, $k
fi

psfile_inc_pref=${prefix}.DT_Australia.rate.filt.${k}.${a3}Ma
psfile_inc=${psfile_inc_pref}.ps
psfile_230_pref=${prefix}.DT_Australia.diff.filt.${k}.${a}Ma.wrt.230Ma
psfile_230=${psfile_230_pref}.ps

### ---- Compute, filter and remask incremental differential topography ----###

if [ ! -f ${topo_rate_inc} ] && [ $i -lt $n ];
        then
	
	grdmath ${topo_young} ${topo_old} SUB ${as4} DIV = tmp1.grd -V

	grdfilter tmp1.grd -D4 -Gtmp2.grd -Fg1000 -I$grdspace -R${region} -Ni -V

	grdmath tmp2.grd maskall.grd OR = ${topo_rate_inc} -V

	rm tmp1.grd tmp2.grd

fi

if [ ! -f ${topo_diff_230} ];
        then

	grdmath ${topo_old} ${topo_230} SUB = tmp3.grd -V

	grdfilter tmp3.grd -D4 -Gtmp4.grd -Fg500 -I$grdspace -R${region} -Ni -V

	grdmath tmp4.grd maskall.grd OR = ${topo_diff_230} -V

	rm tmp3.grd tmp4.grd
fi

### ---- Extract point location ---- ###

#awk '{
#if ($1 !~ ">" && $1 < 0)
#        print $1+360, $2;
#else
#        print $0;
#}' ${FALK} > tmp_sample.xyz
#
#point1r=$(grdtrack tmp_sample.xyz -G${topo_rate_inc} -m -R${region} -Qn -V | awk '{if ($1 !~ ">") print $3}')
#echo $as3 $point1r >> FALK_DTrate.tz
#
#point1c=$(grdtrack tmp_sample.xyz -G${topo_diff_230} -m -R${region} -Qn -V | awk '{if ($1 !~ ">") print $3}')
#echo $fage $point1c >> FALK_cummulativeDT.tz
#
#rm tmp_sample.xyz

projb=M18c

echo $region
echo $regionb
echo $projb

# ---- Plot incremental differential dynamic topography -----

colour1=trend_inc.cpt
makecpt -Cpolar -T-20/20/4 -D > $colour1

grdimage -C$colour1 ${topo_rate_inc} -J${projb} -P -R${region} -Y4 -B5:."${a3} Ma": -V -K > $psfile_inc
psxy ${infile3} -m -W2/black -J -R -K -O -V >> $psfile_inc
pscoast -Dl -R -J -K -O -V -W4,black >> $psfile_inc
#psxy ${FALK} -m -Sa0.5 -Gwhite -W4,purple -J -R -O -K -V >> $psfile_inc
psscale -C$colour1 -Ba8f4:"Rate of dynamic topography change [m Myr @+-1@+]": -D9/-1/18/0.5h -V -O >> $psfile_inc

ps2raster $psfile_inc -E300 -A -TG -P -V

convert -verbose ${psfile_inc_pref}.png ${psfile_inc_pref}.gif

# ---- Plot differential dynamic topography wrt present-day -----

colour2=trend_230.cpt
#makecpt -Cpolar -T-450/450/75 -D > $colour2
makecpt -Cpolar -T-1200/1200/200 -D > $colour2

grdimage -C$colour2 ${topo_diff_230} -J${projb} -P -R${region} -Y4 -B5:."${a} Ma - 230 Ma": -V -K > $psfile_230
psxy ${infile3} -m -W2/black -J -R -K -O -V >> $psfile_230
pscoast -Dl -R -J -K -O -V -W4,black >> $psfile_230
#psxy ${FALK} -m -Sa0.5 -Gwhite -W4,purple -J -R -O -K -V >> $psfile_230
psscale -C$colour2 -Ba200f100:"Dynamic topography change [m]": -D9/-1/18/0.5h -V -O >> $psfile_230
#psscale -C$colour2 -Ba800f400:"Change in model elevation [m]": -D9/-1/18/0.5h -V -O >> $psfile_230

ps2raster $psfile_230 -E300 -A -TG -P -V

convert -verbose ${psfile_230_pref}.png ${psfile_230_pref}.gif

i=$(($i + 1 ))
done

#awk '{ if ($1 < 30 ) print $0 }' FALK_DTrate.tz > FALK_DTrate_pj.tz
#awk '{ if ($1 < 30 ) print $0 }' FALK_cummulativeDT.tz > FALK_cummulativeDT_pj.tz

gifsicle -V --delay=80 --loop --colors 256 ${prefix}.DT_Australia.rate.*Ma.gif > ${prefix}.DT_Australia.rate_10Myr_increment.gif

gifsicle -V --delay=80 --loop --colors 256 ${prefix}.DT_Australia.diff.*wrt.230Ma.gif > ${prefix}.DT_Australia.diff.wrt.230Ma.gif

rm maskall.grd

mv *.png PNG/
mv *.ps PS/
mv *.gif GIF/
mv *.tz TZ/
