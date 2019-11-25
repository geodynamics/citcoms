gmtset COLOR_MODEL RGB BASEMAP_TYPE fancy ANNOT_FONT_SIZE_PRIMARY 12 LABEL_FONT_SIZE 12 HEADER_FONT_SIZE 18 ANNOT_FONT_PRIMARY Helvetica PLOT_DEGREE_FORMAT ddd

colour=DT.cpt
makecpt -Cpolar -T-2000/2000/200 -D > $colour

mkdir PS PNG

prefix=gld29NLt

wc -l $prefix-all.timese > aaa
n=$(awk '{ print $1 }' aaa)
echo $n
rm aaa

i=1
#i=$n
while (($i <= $n))
#while (($i <= 1))
do

l=0

a=$(awk '{ if (NR == '$i') print $2 }' $prefix-all.timese)
echo $a

file=$prefix-$a
topo=${file}.topo.$l
corr=${file}.topo_corr.$l
psfile=${corr}_selected_locations.ps
#psfile=${corr}.ps

region=g
age=$(awk '{ if (NR == '$i') print $2 }' $prefix-all.timese)

root=/Volumes/Data3/Citcoms-runs/Global/Input/Topologies/20130524_svn191

subduction_left=$root/topology_subduction_boundaries_sL_${age}.00Ma.xy
subduction_right=$root/topology_subduction_boundaries_sR_${age}.00Ma.xy
flatslabtrench=$root/topology_slab_edges_trench_${age}.00Ma.xy
flatslableading_left=$root/topology_slab_edges_leading_sL_${age}.00Ma.xy
flatslableading_right=$root/topology_slab_edges_leading_sR_${age}.00Ma.xy
flatslabside=$root/topology_slab_edges_side_${age}.00Ma.xy
coasts=/Volumes/Data3/Citcoms-runs/Global/Input/GlobalGrids/20130524_svn191/Coastlines_2012.2/reconstructed_${age}.00Ma.xy
all_COB=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Reference_points_xlight_Heine/reconstructed_${age}.00Ma.xy
just_COB=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Reference_points_xxlight_Heine/reconstructed_${age}.00Ma.xy

Falkland14051A=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Jones2004/Falkland14051A/reconstructed_${age}.00Ma.xy
AC3=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Hirsch2010/AC-3/reconstructed_${age}.00Ma.xy

Sergipe_C=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Chang1992/Sergipe_Chang/reconstructed_${age}.00Ma.xy
Bahia_C=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Chang1992/Bahia_Chang/reconstructed_${age}.00Ma.xy
Espirito_C=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Chang1992/Espirito_Santo_Chang/reconstructed_${age}.00Ma.xy
Campos_C=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Chang1992/Campos_Chang/reconstructed_${age}.00Ma.xy
Santos_C=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Chang1992/Santos_Chang/reconstructed_${age}.00Ma.xy
Pelotas_C=/Volumes/Data3/Citcoms-runs/Global/Postprocessing/Chang1992/Pelotas_Chang/reconstructed_${age}.00Ma.xy

echo $file
z_grid=${corr}.grd

        if [ ! -f ${z_grid} ];
        then

        blockmedian $topo -Rg -I0.1 -V > ${topo}.median
        surface ${topo}.median -I0.1 -Rg -T0.5 -V -G${topo}.grd

        bbb=$(grdinfo -L2 ${topo}.grd | grep mean | awk '{print $3}')

        echo $bbb

	rm ${topo}.median

        grdmath -V ${topo}.grd $bbb SUB = ${corr}.grd
        fi

grdimage ${corr}.grd -R${region} -JW0/20c -Ba30f15:."$age Ma": -C$colour -X0.75 -Y5.0 -P -V -K > $psfile

#psxy -R -J -W0.4p,grey -K -O -m $cob -V >> $psfile
#psxy $plates -m -V -O -K -Rd -J -W2 >> $psfile
psxy ${coasts} -m -W1/black -J -R -K -O -V >> $psfile
psxy -R -J -W2.0p,darkgrey -Sf6p/2plt -K -O -m ${subduction_left} -Gred -V >> $psfile
psxy -R -J -W2.0p,darkgrey -Sf6p/2prt -K -O -m ${subduction_right} -Gred -V >> $psfile

#psxy -R -J -W1.0p,blue -K -O -m ${flatslabtrench} -V >> $psfile
#psxy -R -J -W1.0p,blue -Sf4p/2plt -K -O -m ${flatslableading_left} -Gblue -V >> $psfile
#psxy -R -J -W1.0p,blue -Sf4p/2prt -K -O -m ${flatslableading_right} -Gblue -V >> $psfile
#psxy -R -J -W1.0p,blue -K -O -m ${flatslabside} -V >> $psfile
#psxy ${FLO} -m -Sa0.3 -Gwhite -W4 -J -R -O -K -V >> $psfile

#psscale -C$colour -Ba1250f625:"Dynamic topography [m]": -D10c/-1c/12c/1ch -V -O >> $psfile
psscale -C$colour -Ba400f200:"Dynamic topography [m]": -D10c/-1c/16c/0.75ch -V -O >> $psfile

ps2raster $psfile -E300 -A -TG -P -V

i=$(($i + 1 ))
done

mv *.png PNG/
mv *.ps PS/

