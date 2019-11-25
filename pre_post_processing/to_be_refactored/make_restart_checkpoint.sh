prefix=gld20
prefixct=${prefix}ct


n=$(wc -l $prefix.timese | awk '{ print $1 }')
echo $n

i=1
#i=$n
#i=10
#while (($i <= 1))
while (($i <= $n))
do

j=$(awk '{ if (NR == '$i') print $1 }' $prefix.timese)
age=$(awk '{ if (NR == '$i') print $2 }' $prefix.timese)

steps_i=$(($j))
nsteps=$(($steps_i + 2 ))

echo $prefix	$steps_i	$nsteps		$age
sed "s/nsteps/$nsteps/g
s/steps_i/$steps_i/g
s/prefix/$prefix/g" checkpoint-template-$prefix.cfg > ${prefixct}-$age.cfg

jobname=${prefixct}${age}
filename=${prefixct}-$age.cfg

echo $filename $jobname

sed "s/jobname/$jobname/g
s/filename/$filename/g" AAcheckpoint-template-$prefix.pbs > AA${prefixct}-$age.pbs

qsub AA${prefixct}-$age.pbs

i=$(($i + 1 ))
done
