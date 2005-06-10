#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

'''
Paste and combine Citcom data

Usage: batchcombine.py <machinefile | node-list> model-dir model-name timestep nodex nodey nodez ncap nprocx nprocy nprocz
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
    for node in nodelist:
        nodes += '%s ' % node.strip()

    # paste
    cmd = 'batchpaste.sh %(modeldir)s %(modelname)s %(timestep)d %(nodes)s' \
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
# $Id: batchcombine.py,v 1.6 2005/06/10 02:23:25 leif Exp $

# End of file
