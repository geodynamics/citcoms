#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

'''
Combine the pasted Citcom Data

usage: batchcombine.py machinefile modeldir modelname timestep nodex nodey nodez n_surf_proc nprocx nprocy nprocz
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

    # generate a list of machines
    nodelist = ''
    for node in file(machinefile).readlines():
        nodelist += '%s ' % node.strip()

    # paste
    cmd = 'batchpaste.sh %(modeldir)s %(modelname)s %(timestep)d %(nodelist)s' \
          % vars()
    print cmd
    os.system(cmd)

    # combine
    cmd = 'combine.py %(modelname)s %(timestep)d %(nodex)d %(nodey)d %(nodez)d %(ncap)d %(nprocx)d %(nprocy)d %(nprocz)d' % vars()
    print cmd
    os.system(cmd)

    # delete
    cmd = 'rm %(modelname)s.[0-9]*.%(timestep)d' % vars()
    print cmd
    os.system(cmd)

    # create .general file
    cmd = 'dxgeneral.sh %(modelname)s.cap*.%(timestep)d' % vars()
    print cmd
    os.system(cmd)


# version
# $Id: batchcombine.py,v 1.1 2004/01/15 00:48:48 tan2 Exp $

# End of file
