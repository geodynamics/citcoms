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



class Inlet(object):

    def __init__(self):
        self._handle = None
        return


    def impose(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.Inlet_impose(self._handle)
        return


    def recv(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.Inlet_recv(self._handle)
        return


    def storeTimestep(self, fge_t, cge_t):
        import CitcomS.Exchanger as Exchanger
        Exchanger.Inlet_storeTimestep(self._handle, fge_t, cge_t)
        return




class SVTInlet(Inlet):

    def __init__(self, mesh, sink, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.SVTInlet_create(mesh,
                                                 sink,
                                                 all_variables)
        return




class TInlet(Inlet):

	def __init__(self, mesh, sink, all_variables):
		import CitcomS.Exchanger as Exchanger
		self._handle = Exchanger.TInlet_create(mesh,
											   sink,
											   all_variables)
		return


class SInlet(Inlet):

	def __init__(self, mesh, sink, all_variables):
		import CitcomS.Exchanger as Exchanger
		self._handle = Exchanger.SInlet_create(mesh,
											   sink,
											   all_variables)
		return


class VTInlet(Inlet):

	def __init__(self, mesh, sink, all_variables):
		import CitcomS.Exchanger as Exchanger
		self._handle = Exchanger.VTInlet_create(mesh,
												sink,
												all_variables)
		return



"""
class BoundaryVTInlet(Inlet):
    '''Available modes -- see above
    '''


    def __init__(self, communicator, boundary, sink, all_variables, mode="VT"):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.BoundaryVTInlet_create(communicator.handle(),
                                                        boundary,
                                                        sink,
                                                        all_variables,
                                                        mode)
        return




class TractionInlet(Inlet):
    '''Inlet that impose velocity and/or traction on the boundary
    Available modes --
    "F": traction only
    "V": velocity only
    "FV": normal velocity and tangent traction
    '''

    def __init__(self, boundary, sink, all_variables, mode='F'):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.TractionInlet_create(boundary,
                                                      sink,
                                                      all_variables,
                                                      mode)
        return

"""

# version
__id__ = "$Id: Inlet.py,v 1.8 2005/06/10 02:23:22 leif Exp $"

# End of file
