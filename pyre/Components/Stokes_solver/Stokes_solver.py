#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2003  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application

class Stokes_solver(Application):


    def main(self):
	# Solver doesn't run stand along
	return

    def run(self, *args, **kwds):
	Application.run(*args, **kwds)
	return


    def output(self, *args, **kwds):
	return


    def __init__(self, name):
	Application.__init__(self, name)
	return


# version
__id__ = "$Id: Stokes_solver.py,v 1.1.1.1 2003/05/15 00:07:54 tan2 Exp $"

# End of file 
