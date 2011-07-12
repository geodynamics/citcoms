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

class Output(CitcomComponent):


    def __init__(self, name="output", facility="output"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self, stream):
        from CitcomSLib import Output_set_properties
        Output_set_properties(self.all_variables, self.inventory, stream)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.inventory as inv

        output_format = inv.str("output_format", default="ascii",
                                validator=inv.choice(["ascii",
                                                      "ascii-gz",
						      "vtk",
                                                      "hdf5"]))
        output_optional = inv.str("output_optional",
                                  default="surf,botm,tracer")

        vtk_format = inv.str("vtk_format", default="ascii",
                             validator=inv.choice(["binary", "ascii"]))

        # experimental vtk output
        gzdir_vtkio = inv.int("gzdir_vtkio", default=0)
        # remove net rotation
        gzdir_rnr = inv.bool("gzdir_rnr", default=False)

        # write additional heat flux files? if yes, how frequent?
        write_q_files = inv.int("write_q_files", default=0)

        # max. degree for spherical harmonics output
        output_ll_max = inv.int("output_ll_max", default=20)

        # self-gravitation, for geoid only
        self_gravitation = inv.bool("self_gravitation", default=False)

        # compute stress/topography by consistent-boundary-flux method
        use_cbf_topo = inv.bool("use_cbf_topo", default=False)

        mega1 = 1024*1024
        #megaq = 256*1024

        # size of collective buffer used by MPI-IO
        cb_block_size = inv.int("cb_block_size", default=mega1)
        cb_buffer_size = inv.int("cb_buffer_size", default=mega1*4)

        # size of data sieve buffer used by HDF5
        sieve_buf_size = inv.int("sieve_buf_size", default=mega1)

        # memory alignment used by HDF5
        output_alignment = inv.int("output_alignment", default=mega1/4)
        output_alignment_threshold = inv.int("output_alignment_threshold",
                                        default=mega1/2)

        # cache for chunked dataset used by HDF5
        cache_mdc_nelmts = inv.int("cache_mdc_nelmts", default=10330)
        cache_rdcc_nelmts = inv.int("cache_rdcc_nelmts", default=521)
        cache_rdcc_nbytes = inv.int("cache_rdcc_nbytes", default=mega1)


# version
__id__ = "$Id$"

# End of file
