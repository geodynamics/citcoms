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



def coupledFullSolver(name, facility):
    from CoupledFullSolver import CoupledFullSolver
    return CoupledFullSolver(name, facility)



def coupledRegionalSolver(name, facility):
    from CoupledRegionalSolver import CoupledRegionalSolver
    return CoupledRegionalSolver(name, facility)



def multicoupledFullSolver(name, facility):
    from MultiC_FullSolver import MultiC_FullSolver
    return MultiC_FullSolver(name, facility)



def multicoupledRegionalSolver(name, facility):
    from MultiC_RegionalSolver import MultiC_RegionalSolver
    return MultiC_RegionalSolver(name, facility)



def fullSolver(name='full', facility='solver'):
    from FullSolver import FullSolver
    return FullSolver(name, facility)



def regionalSolver(name='regional', facility='solver'):
    from RegionalSolver import RegionalSolver
    return RegionalSolver(name, facility)



# version
__id__ = "$Id: __init__.py 7683 2007-07-17 22:48:26Z tan2 $"

# End of file
