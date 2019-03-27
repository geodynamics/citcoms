folder=/Volumes/Data1/CitcomS-runs/Global/Input/Continental_Types/20141022/NoAssimilationStencil

mkdir ${folder}/Deforming_networks_all

age=0
while (( $age <= 150 ))
do

dynamic=$folder/Deforming_networks_dynamic/topology_network_polygons_${age}.00Ma.xy
static=$folder/Deforming_networks_static/topology_network_polygons_${age}.00Ma.xy

if [ -f ${dynamic} -a -f ${static} ];
then
        cp ${dynamic} tmp
        echo "dynamic and static both exist"
        echo -e -n "\n" >> tmp
        cat tmp ${static} > $folder/Deforming_networks_all/topology_network_polygons_${age}.00Ma.xy
        rm tmp

elif [ -f ${static} ];
then
	echo "only static exists"
        cp ${static} $folder/Deforming_networks_all/topology_network_polygons_${age}.00Ma.xy

elif [ -f ${dynamic} ];
then
	echo "only dynamic exists"
        cp ${dynamic} $folder/Deforming_networks_all/topology_network_polygons_${age}.00Ma.xy
else
	echo "something has gone wrong mate, I can't find either file"
fi

age=$(($age +1 ))
done
