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



    def setProperties(self, stream):

        from CitcomSLib import IC_set_properties

        inv = self.inventory
        inv.perturbmag = map(float, inv.perturbmag)
        inv.perturbl = map(int, inv.perturbl)
        inv.perturbm = map(int, inv.perturbm)
        inv.blob_center = map(float, inv.blob_center)
        inv.silo_base_center = map(float, inv.silo_base_center)

        IC_set_properties(self.all_variables, self.inventory, stream)

        return



    def launch(self):
        self.initMaterial()
        self.initTracer()
        self.initTemperature()
        self.initPressure()
        self.initVelocity()
        self.initViscosity()
        return


    def initMaterial(self):
        from CitcomSLib import initialize_material
        initialize_material(self.all_variables)
        return


    def initTracer(self):
        from CitcomSLib import init_tracer_composition
        init_tracer_composition(self.all_variables)
        return


    def initTemperature(self):
        from CitcomSLib import constructTemperature
        constructTemperature(self.all_variables)
        return



    def initPressure(self):
        from CitcomSLib import initPressure
        initPressure(self.all_variables)
        return



    def initVelocity(self):
        from CitcomSLib import initVelocity
        initVelocity(self.all_variables)
        return



    def initViscosity(self):
        from CitcomSLib import initViscosity
        initViscosity(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory



        restart = pyre.inventory.bool("restart", default=False)
        post_p = pyre.inventory.bool("post_p", default=False)
        solution_cycles_init = pyre.inventory.int("solution_cycles_init", default=0)
        zero_elapsed_time = pyre.inventory.bool("zero_elapsed_time", default=True)

        tic_method = pyre.inventory.int("tic_method", default=0)

        # for tic_method=0 or 3
        num_perturbations = pyre.inventory.int("num_perturbations", default=1,
                            validator=pyre.inventory.less(255))
        perturbl = pyre.inventory.list("perturbl", default=[1])
        perturbm = pyre.inventory.list("perturbm", default=[1])
        perturblayer = pyre.inventory.slice("perturblayer", default=[5])
        perturbmag = pyre.inventory.list("perturbmag", default=[0.05])

        # for tic_method=1 or 2
        half_space_age = pyre.inventory.float("half_space_age", default=40,
                              validator=pyre.inventory.greater(1e-3))
        mantle_temp = pyre.inventory.float("mantle_temp", default=1.0)

        # for tic_method=2
        blob_center = pyre.inventory.list("blob_center", default=[-999., -999., -999.])
        blob_radius = pyre.inventory.float("blob_radius", default=0.063)
        blob_dT = pyre.inventory.float("blob_dT", default=0.18)
        blob_profile = pyre.inventory.str("blob_profile", default="default")
        
        # for tic_method=400
        silo_base_center = pyre.inventory.list("silo_base_center", default=[-999., -999., -999.])
        silo_radius = pyre.inventory.float("silo_radius", default=0.063)
        silo_cylinder_height = pyre.inventory.float("silo_cylinder_height", default=0.0784)
        silo_dT = pyre.inventory.float("silo_dT", default=0.18)
        silo_profile = pyre.inventory.str("silo_profile", default="default")
        
        # for tic_method=401
        mantle_temp_adiabatic_increase = pyre.inventory.float("mantle_temp_adiabatic_increase", default=0.3)

# version
__id__ = "$Id: IC.py 14735 2009-04-16 18:24:28Z tan2 $"

# End of file
