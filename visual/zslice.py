#!/usr/bin/env python
"""
Slice horizontal layer(s) from CitcomS output (from batchcombine.py).
Also convert the (x, y) coordinate to (lon, lat).

usage:
  zslice(prefix, layer1 [,layer2 [,...]] )
    -- slice the specified layer(s)
       layer# : layer(s) to plot (count from 0)

  zslice(prefix)
    -- slice all layers

  It can be used in the command line as well.

input:
  prefix
outout:
  prefix.z###
"""

all = ('zslice')

def zslice(prefix, *ilayers):

    from math import pi
    r2d = 180.0 / pi

    # open input file and read header
    input = open(prefix)
    nodex, nodey, nodez = input.readline().split('x')
    nodez = int(nodez)
    #print nodez, ilayers

    layers = check_layers(ilayers, nodez)
    nlayer = len(layers)

    # allocate arrays
    output = range(nlayer)
    lines = range(nodez)

    # open output files
    for i in range(nlayer):
        zfile = '%s.z%03d' % (prefix, layers[i])
        output[i] = open(zfile, 'w')

    while 1:
        for j in range(nodez):
            lines[j] = input.readline()

        # file is empty or EOF?
        if not lines[nodez-1]:
            break

        for i in range(nlayer):
            layer = layers[i]

            # spilt the first 3 columns only, the rest will be
            # output as-is
            data = lines[layer].split(' ', 3)
            lat = 90 - float(data[0])*r2d
            lon = float(data[1])*r2d
            output[i].write( '%f %f %s' % (lon, lat, data[3]) )

    input.close()

    for i in range(nlayer):
        output[i].close()

    return



def check_layers(layers, nodez):
    if layers == ():
        # if empty, we will slice every layer
        layers = range(nodez)
    else:
        # otherwise, check bounds of layers
        for layer in layers:
            if not (0<= layer < nodez):
                raise ValueError, 'layer out of range (0-%d)' % (nodez - 1)

    return layers



# if run as a script
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(1)

    prefix = sys.argv[1]
    layers = [ int(x) for x in sys.argv[2:] ]

    zslice(prefix, *layers)
