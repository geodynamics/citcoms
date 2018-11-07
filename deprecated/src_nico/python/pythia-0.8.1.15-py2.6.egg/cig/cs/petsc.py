#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               cig.cs
#
# Copyright (c) 2006, California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#
#    * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
#    * Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.components import Component
from mpi.Application import Application as MPIApplication
import pyre.parsing.locators
import pyre.util.bool
import sys


class Petsc(Component):


    def setDefaults(self, dct):
        locator = pyre.parsing.locators.default()
        for key, value in dct.iteritems():
            self.options.setProperty(key, value, locator)
        return


    def updateConfiguration(self, registry):
        self.options.update(registry)
        return []


    def getArgs(self):
        options = [
            (name, descriptor.value)
            for name, descriptor in self.options.properties.iteritems()
            ]
        args = []
        for iname, value in options:
            try:
                if pyre.util.bool.bool(value):
                    args.append('-' + iname)
                else:
                    # The only way to turn off a PETSc option is to omit it.
                    pass
            except KeyError:
                # non-boolean option
                args.append('-' + iname)
                args.append(value)
        return args


    def __init__(self, name):
        Component.__init__(self, name, name)
        self.options = self.createRegistry()
        return



class PetscApplication(MPIApplication):


    class Inventory(MPIApplication.Inventory):
        
        import pyre.inventory as pyre

        # a dummy facility for passing arbitrary options to PETSc
        petsc = pyre.facility("petsc", factory=Petsc, args=["petsc"])


    def setPetscDefaults(self, dct):
        """Set the default options passed to PetscInitialize().
        
        This method should be called from _defaults()."""
        
        self.inventory.petsc.setDefaults(dct)
        
        return


    def _configure(self):

        super(PetscApplication, self)._configure()
        
        self.petscArgs = [sys.executable]
        self.petscArgs.extend(self.inventory.petsc.getArgs())
        self._debug.log("PetscInitialize args: %r" % self.petscArgs)

        return


    def _onComputeNodes(self, *args, **kwds):
        petsc = self.petsc
        petsc.PetscInitialize(self.petscArgs)
        super(PetscApplication, self)._onComputeNodes(*args, **kwds)
        petsc.PetscFinalize()



# end of file 
