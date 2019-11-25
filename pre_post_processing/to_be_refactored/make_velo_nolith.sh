prefix=gld20

n=$(wc -l $prefix.timese | awk '{ print $1 }')
echo $n

mkdir coor_no-header restart-nolith

i=1
#i=9
#i=$n
while (($i <= $n))
#while (($i <= 3))
do


step=$(awk '{ if (NR == '$i') print $1 }' $prefix.timese)

echo $step

proc=0
while (($proc <= 95))
do

if [ "$i" == "1" ];
        then
	awk '{ if (NR >1) print $0 }' Data/$proc/${prefix}.coord.$proc > ./coor_no-header/coord.$proc.tmp
fi

awk '{ if (NR >2) print $0 }' Data/$proc/${prefix}.velo.$proc.$step > velo.$proc.$step.tmp
awk '{ if (NR <3) print $0 }' Data/$proc/${prefix}.velo.$proc.$step > head.$proc.$step.tmp

paste ./coor_no-header/coord.$proc.tmp velo.$proc.$step.tmp > velo_coord.$proc.$step.tmp

# Hard-coded depth 0.945063 is 350km

awk '{ 
if ($3>=0.945063) 
	printf "%12.6e %12.6e %12.6e %12.6e\n", 0, 0, 0, 0.5;
else if ($3<0.945063)
	print $4,$5,$6,$7;
}' velo_coord.$proc.$step.tmp > velo_nolith.$proc.$step.tmp

cat head.$proc.$step.tmp velo_nolith.$proc.$step.tmp > ${prefix}NL.velo.$proc.$step

rm velo.$proc.$step.tmp head.$proc.$step.tmp velo_coord.$proc.$step.tmp velo_nolith.$proc.$step.tmp

mv ${prefix}NL.velo.$proc.$step restart-nolith/

proc=$(( $proc + 1 ))
done

i=$(($i + 1 ))
done
