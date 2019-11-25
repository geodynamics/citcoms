pref=gld29
prefcoord=gld21
prefix=${pref}NLt
prefct=${pref}ct

wc -l $prefix-all.timese > aa
n=$(awk '{ print $1 }' aa)

echo $n

i=1
#i=3
while (($i <= $n))
#while (( $i <= 2 ))
do

#l=$(awk '{ if (NR == '$i') print $1 }' $prefix-all.timese)
l=0
echo $l

k=$(($i-1))
a=$(awk '{ if (NR == '$i') print $2 }' $prefix-all.timese)
echo $a
b=$(awk '{ if (NR == '$k') print $2 }' $prefix-all.timese)
echo $b

file=$prefix-$a
file2=$prefix-$b

dir_coord=/Volumes/Data3/Citcoms-runs/Global/Results/Lith_Def/${prefcoord}/Topo/check/DataSurf

proc=0
while (( $proc <= 95 ))
do

	if [ ${i} -eq 1 ];
		then
		cp ${dir_coord}/${prefcoord}ct.coord.${proc} DataSurf/${file}.coord.${proc}
	else
		b=$(awk '{ if (NR == '$k') print $2 }' $prefix-all.timese)
		mv DataSurf/${file2}.coord.${proc} DataSurf/${file}.coord.${proc}
	fi

proc=$(($proc + 1 ))
done

rm $file.topo.$l
rm $file.heat.$l

touch $file.topo.$l
touch $file.heat.$l

echo $file

batchsurf.py localhost DataSurf/ $file $l 129 129 65 12 2 2 2

        if [ ${i} -eq ${n} ];
                then
		rm DataSurf/${file}.coord.*
	fi


j=0
while (( $j <= 11 ))
do

#Surface dynamic topography
awk '{ if ( NR > 1 ) print $2*57.2957795, 90-($1*57.2957795), $3*1e15/(3340*9.81*(6.371e6)^2) }' $file.surf$j.$l > $file.topo$j.$l

cat $file.topo$j.$l >> $file.topo.$l
rm $file.topo$j.$l

#Surface heat_flux
#scaling: k * Delta T/R; where k = rho *  kappa * Cp; *1000 to be in mW m-2; *2 if T = 0.5
awk '{ if ( NR > 1 ) print $2*57.2957795, 90-($1*57.2957795), $4*1000*4000*1e-6*1200*1400/6371e3}' $file.surf$j.$l > $file.heat$j.$l

cat $file.heat$j.$l >> $file.heat.$l
rm $file.heat$j.$l

j=$(($j + 1 ))
done

i=$(($i + 1 ))
done

