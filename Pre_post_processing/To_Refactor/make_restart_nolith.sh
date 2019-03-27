prefix1=gld20
prefix=gld20
prefixNLt=${prefix}NLt

wc -l $prefix.timese > aaa
n=$(awk '{ print $1 }' aaa)
echo $n
rm aaa

i=1
#while (($i <= 5))
while (($i <= $n))
do

steps_i=$(awk '{ if (NR == '$i') print $1 }' $prefix.timese)
age=$(awk '{ if (NR == '$i') print $2 }' $prefix.timese)
sage=$(awk '{ if (NR == '$i') print $2 }' $prefix.times)

filename=${prefixNLt}-$age.cfg

echo $prefix	$steps_i	$age		$sage
sed "s/sage/$sage/g
s/steps_i/$steps_i/g
s/mage/$age/g
s/prefix1/$prefix1/g
s/prefix/$prefix/g
s/prefixNLt/$prefixNLt/g" nolith-template-$prefix.cfg > $filename

jobname=${prefixNLt}${age}

echo $filename $jobname

sed "s/jobname/$jobname/g
s/filename/$filename/g" AAnolith-template-$prefix.pbs > AA${prefixNLt}-$age.pbs

qsub AA${prefixNLt}-$age.pbs

i=$(($i + 1 ))
done
