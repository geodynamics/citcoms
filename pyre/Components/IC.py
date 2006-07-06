#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class IC(CitcomComponent):


    def __init__(self, name="ic", facility="ic"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):

        from CitcomS.CitcomS import IC_set_properties
        
        inv = self.inventory
        inv.perturbmag = map(float, inv.perturbmag)
        inv.perturbl = map(float, inv.perturbl)
        inv.perturbm = map(float, inv.perturbm)
        inv.blob_center = map(float, inv.blob_center)

        IC_set_properties(self.all_variables, inv)
        
        return



    def launch(self):
        self.initTemperature()
        self.initPressure()
        self.initVelocity()
        self.initViscosity()
        return



    def initTemperature(self):
        from CitcomS.CitcomS import constructTemperature, restartTemperature
        if self.inventory.restart:
            restartTemperature(self.all_variables)
        else:
            constructTemperature(self.all_variables)
        return



    def initPressure(self):
        from CitcomS.CitcomS import initPressure
        initPressure(self.all_variables)
        return



    def initVelocity(self):
        from CitcomS.CitcomS import initVelocity
        initVelocity(self.all_variables)
        return



    def initViscosity(self):
        from CitcomS.CitcomS import initViscosity
        initViscosity(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory



        restart = pyre.inventory.bool("restart", default=False)
        post_p = pyre.inventory.bool("post_p", default=False)
        solution_cycles_init = pyre.inventory.int("solution_cycles_init", default=0)
        zero_elapsed_time = pyre.inventory.bool("zero_elapsed_time", default=True)

        tic_method = pyre.inventory.int("tic_method", default=0,
                            validator=pyre.inventory.choice([0, 1, 2]))

        half_space_age = pyre.inventory.float("half_space_age", default=40,
                              validator=pyre.inventory.greater(1e-3))

        num_perturbations = pyre.inventory.int("num_perturbations", default=1,
                            validator=pyre.inventory.less(255))
        perturbmag = pyre.inventory.list("perturbmag", default=[0.05])
        perturbl = pyre.inventory.list("perturbl", default=[1])
        perturbm = pyre.inventory.list("perturbm", default=[1])
        perturblayer = pyre.inventory.slice("perturblayer", default=[5])


        blob_center = pyre.inventory.list("blob_center", default=[-999., -999., -999.])
        blob_radius = pyre.inventory.float("blob_radius", default=0.063)
        blob_dT = pyre.inventory.float("blob_dT", default=0.18)


# version
__id__ = "$Id$"

# End of file
