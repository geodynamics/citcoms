#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

'''
Combine Citcom Surface Data

usage: batchsurf.py machinefile modeldir modelname timestep nodex nodey nodez n_surf_proc nprocx nprocy nprocz
'''

if __name__ == '__main__':

    import sys, os

    if not len(sys.argv) == 12:
        print __doc__
        sys.exit(1)

    machinefile = sys.argv[1]
    modeldir = sys.argv[2]
    modelname = sys.argv[3]
    timestep = int(sys.argv[4])
    nodex = int(sys.argv[5])
    nodey = int(sys.argv[6])
    nodez = int(sys.argv[7])
    ncap = int(sys.argv[8])
    nprocx = int(sys.argv[9])
    nprocy = int(sys.argv[10])
    nprocz = int(sys.argv[11])

    try:
        nodelist = file(machinefile).readlines()
    except IOError:
        nodelist = machinefile.split()

    # check the length of nodelist
    totalnodes = nprocx * nprocy * nprocz * ncap
    n = len(nodelist)
    if not n == totalnodes:
        print 'WARNING: length of machinefile does not match number of processors'
        if (totalnodes > n) and ((totalnodes % n) == 0):
            # try to match number of processors by duplicating nodelist
            nodelist *= (totalnodes / n)
        else:
            print 'ERROR: incorrect machinefile size'
            sys.exit(1)

    # generate a string of machine names
    nodes = ''
    for n in range(0, totalnodes/nprocz, nprocz):
        nodes += '%s ' % nodelist[n].strip()

    # get coordinate, if necessary
    coord_exist = True
    for n in range(totalnodes):
        coord_file = '%(modelname)s.coord.%(n)d' % vars()
        coord_exist &= os.path.exists(coord_file)

    if not coord_exist:
        cmd = 'getcoord.sh %(modeldir)s %(modelname)s %(nodes)s' % vars()
        print cmd
        os.system(cmd)

    # get surf data
    cmd = 'getsurf.sh %(modeldir)s %(modelname)s %(timestep)d %(nprocz)d %(nodes)s' % vars()
    print cmd
    os.system(cmd)

    # combine
    cmd = 'combinesurf.py %(modelname)s %(timestep)d %(nodex)d %(nodey)d %(nodez)d %(ncap)d %(nprocx)d %(nprocy)d %(nprocz)d' % vars()
    print cmd
    os.system(cmd)

    # delete
    cmd = 'rm %(modelname)s.surf.[0-9]*.%(timestep)d' % vars()
    print cmd
    os.system(cmd)

    # create .general file
    cmd = 'dxgeneralsurf.sh %(modelname)s.surf[0-9]*.%(timestep)d' % vars()
    print cmd
    os.system(cmd)


# version
# $Id: batchsurf.py,v 1.2 2004/08/06 23:25:20 tan2 Exp $

# End of file
