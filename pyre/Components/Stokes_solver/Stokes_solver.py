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
	raise NotImplementedError, \
	      "Solver doesn't run stand-along. Call init(), run(), fini() sequentially in stead."
	return

    def run(self, *args, **kwds):
	Application.run(self, *args, **kwds)
	
	self.form_RHS()
	self.form_LHS()
	self.solve()

	return


    def form_RHS(self):
	return


    def form_LHS(self):
	return


    def solve(self):
	return


    def output(self, *args, **kwds):
	return



# version
__id__ = "$Id: Stokes_solver.py,v 1.3 2003/05/20 18:56:58 tan2 Exp $"

# End of file 
