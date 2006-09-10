#!/usr/bin/env python

"""
This script estimates the size of the output from CitcomS,
assuming binary floating point data.

Options are:

    estimate.py [ --help | -h ]
                [ --steps <number> | -t <number> ]
                [ --nodex <number> | -x <number> ]
                [ --nodey <number> | -y <number> ]
                [ --nodez <number> | -z <number> ]
                [ --caps  <number> | -c <number> | --full | --regional ]
                [ --all | -a ]
                [ --connectivity ]
                [ --stress ]
                [ --pressure ]
                [ --surf ]
                [ --botm ]
                [ --average ]

Examples:

    python estimate.py
    python estimate.py --all
    python estimate.py --full --all --steps=100 --nodex=33 --nodey=33 -z65
    python estimate.py --regional --surf --botm --pressure
    python estimate.py --regional --pressure --stress -x55 -y55 -z43
    python estimate.py --caps=12 --steps 100 -x 33 -y 33 -z 33 --stress
    python estimate.py -c12 -t100 -x33 -y33 -z33 -a
    python estimate.py --help
"""

def TiB(x):
    """Convert from bytes to terabytes"""
    return x / 1024.0 ** 4

def GiB(x):
    """Convert from bytes to gigabytes"""
    return x / 1024.0 ** 3

def MiB(x):
    """Convert from bytes to megabytes"""
    return x / 1024.0 ** 2

def KiB(x):
    """Convert from bytes to kilobytes"""
    return x / 1024.0

def ps(x):
    """print size"""
    if x < 1024:
        return '%g bytes' % x
    elif x < 1024**2:
        return '%g KiB' % KiB(x)
    elif x < 1024**3:
        return '%g MiB' % MiB(x)
    elif x < 1024**4:
        return '%g GiB' % GiB(x)
    else:
        return '%g TiB' % TiB(x)

def pc(x,total):
    """print percentage"""
    return '%2.3f%%' % ((100*x)/float(total))

def main():

    import sys
    import getopt

    out = {
        'connectivity': False,
        'stress': False,
        'pressure': False,
        'surf': False,
        'botm': False,
        'average': False,
    }
    caps = None
    steps = None
    nodex = None
    nodey = None
    nodez = None

    opts, args = getopt.getopt(sys.argv[1:], "hac:t:x:y:z:",
        ['help','full','regional','caps=','steps=','nodex=','nodey=','nodez=',
         'all','connectivity','stress','pressure','surf','botm','average'])
    
    for opt,arg in opts:
        
        if opt in ('-h','--help'):
            print __doc__
            sys.exit(1)

        if opt == '--regional':
            caps = 1
        if opt == '--full':
            caps = 12
        if opt in ('-c','--caps'):
            caps = int(arg)

        if opt in ('-t','--steps'):
            steps = int(arg)
        if opt in ('-x','--nodex'):
            nodex = int(arg)
        if opt in ('-y','--nodey'):
            nodey = int(arg)
        if opt in ('-z','--nodez'):
            nodez = int(arg)
        
        if opt in ('-a','--all'):
            for k in out:
                out[k] = True
        if opt == '--connectivity':
            out['connectivity'] = True
        if opt == '--stress':
            out['stress'] = True
        if opt == '--pressure':
            out['pressure'] = True
        if opt == '--surf':
            out['surf'] = True
        if opt == '--botm':
            out['botm'] = True
        if opt == '--average':
            out['average'] = True
    
    if not caps or not steps or not nodex or not nodey or not nodez:
        print "Enter the following quantities:\n"

    if not caps:
        caps  = int(raw_input('\tcaps  = '))

    if not steps:
        steps = int(raw_input('\tsteps = '))
    
    if not nodex:
        nodex = int(raw_input('\tnodex = '))
    
    if not nodey:
        nodey = int(raw_input('\tnodey = '))
    
    if not nodez:
        nodez = int(raw_input('\tnodez = '))


    nno = nodex * nodey * nodez
    nsf = nodex * nodey

    elx = nodex - 1
    ely = nodey - 1
    elz = nodez - 1
    nel = elx * ely * elz

    # conversion factor (double = 8 bytes, float = 4 bytes, int = 4 bytes)
    f = 4


    # fields
    tensor3d = f * nno * 6
    vector3d = f * nno * 3
    scalar3d = f * nno * 1
    vector2d = f * nsf * 2
    scalar2d = f * nsf * 1
    scalar1d = f * nodez
    buffer_total = tensor3d + vector3d + scalar3d + \
                   vector2d + scalar2d + \
                   scalar1d

    connectivity = f * nel * 8
    coord        = caps * scalar3d
    velocity     = steps * caps * vector3d
    temperature  = steps * caps * scalar3d
    viscosity    = steps * caps * scalar3d
    pressure     = steps * caps * scalar3d
    stress       = steps * caps * tensor3d

    surf_coord       = caps * vector2d
    surf_velocity    = steps * caps * vector2d
    surf_temperature = steps * caps * scalar2d
    surf_heatflux    = steps * caps * scalar2d
    surf_topography  = steps * caps * scalar2d
    surf_total       = surf_coord + surf_velocity + surf_temperature + \
                       surf_heatflux + surf_topography

    have_coord = caps * scalar1d
    have_temp  = steps * caps * scalar1d
    have_vxy   = steps * caps * scalar1d
    have_vz    = steps * caps * scalar1d
    have_total = have_coord + have_temp + have_vxy + have_vz

    total  = coord + velocity + temperature + viscosity

    if out['connectivity']:
        total += connectivity
    if out['stress']:
        total += stress
    if out['pressure']:
        total += pressure
    if out['surf']:
        total += surf_total
    if out['botm']:
        total += surf_total
    if out['average']:
        total += have_total

    hr = 33

    print "\n"
    print "By Type (per cap per timestep):"
    print "=" * hr
    print "3D Tensor Field: %s" % ps(tensor3d)
    print "3D Vector Field: %s" % ps(vector3d)
    print "3D Scalar Field: %s" % ps(scalar3d)
    print "2D Vector Field: %s" % ps(vector2d)
    print "2D Scalar Field: %s" % ps(scalar2d)
    print "1D Scalar Field: %s" % ps(scalar1d)
    print "-" * hr
    print "total            %s" % ps(buffer_total)

    print "\n"
    print "By Dataset:"
    print "=" * hr
    if out['connectivity']:
        print "connectivity         %s" % ps(connectivity)
    if True:
        print "coord                %s" % ps(coord)
        print "velocity             %s" % ps(velocity)
        print "temperature          %s" % ps(temperature)
        print "viscosity            %s" % ps(viscosity)
    if out['pressure']:
        print "pressure             %s" % ps(pressure)
    if out['stress']:
        print "stress               %s" % ps(stress)
    if out['surf']:
        print "surf/coord           %s" % ps(surf_coord)
        print "surf/velocity        %s" % ps(surf_velocity)
        print "surf/heatflux        %s" % ps(surf_heatflux)
        print "surf/topography      %s" % ps(surf_topography)
    if out['botm']:
        print "botm/coord           %s" % ps(surf_coord)
        print "botm/velocity        %s" % ps(surf_velocity)
        print "botm/heatflux        %s" % ps(surf_heatflux)
        print "botm/topography      %s" % ps(surf_topography)
    if out['average']:
        print "average/coord        %s" % ps(have_coord)
        print "average/temperature  %s" % ps(have_temp)
        print "average/horiz_velo   %s" % ps(have_vxy)
        print "average/vert_velo    %s" % ps(have_vz)
    if True:
        print "-" * hr
        print "total                %s" % ps(total)

    print "\n"
    print "By Percentage:"
    print "=" * hr
    if out['connectivity']:
        print "connectivity  %s" % pc(connectivity,total)
    if True:
        print "coord         %s" % pc(coord,total)
        print "velocity      %s" % pc(velocity,total)
        print "temperature   %s" % pc(temperature,total)
        print "viscosity     %s" % pc(viscosity,total)
    if out['pressure']:
        print "pressure      %s" % pc(pressure,total)
    if out['stress']:
        print "stress        %s" % pc(stress,total)
    if out['surf']:
        print "surf          %s" % pc(surf_total,total)
    if out['botm']:
        print "botm          %s" % pc(surf_total,total)
    if out['average']:
        print "average       %s" % pc(have_total,total)


if __name__ == '__main__':
    main()
