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
"""Classes supporting conversion of depth and radius on the fly.
<usage example:>
import RadiusDepth

earth_radius = 6371
phase660 = RadiusDepth.Depth(660, earth_radius)

# this will report depth of phase660
phase660.inDepth

# this will report radius of phase660
phase660.inRadius

cmb = RadiusDepth.Radius(3480, earth_radius)
cmb.inDepth
cmb.inRadius
"""


class RadiusDepthBase(object):

    def __init__(self, outer_radius):
	self.R0 = outer_radius



class Radius(RadiusDepthBase):

    def __init__(self, radius, outer_radius):
	if not 0 <= radius <= outer_radius:
	    raise ValueError, "radius out of range (0 - outer_radius)"
	RadiusDepthBase.__init__(self, outer_radius)
	self.inRadius = radius

    def get_inDepth(self):
	return self.R0 - self.inRadius

    inDepth = property(get_inDepth, None, None,
		       "convert radius to depth, read-only")

    __slots__ = ("R0", "inRadius", "inDepth")



class Depth(RadiusDepthBase):

    def __init__(self, depth, outer_radius):
	if not 0 <= depth <= outer_radius:
	    raise ValueError, "depth out of range (0 - outer_radius)"
	RadiusDepthBase.__init__(self, outer_radius)
	self.inDepth = depth

    def get_inRadius(self):
	return self.R0 - self.inDepth

    inRadius = property(get_inRadius, None, None,
			"convert depth to radius, read-only")

    __slots__ = ("R0", "inRadius", "inDepth")



# version
__id__ = "$Id: RadiusDepth.py,v 1.1 2003/03/24 19:48:11 tan2 Exp $"

# End of file 
