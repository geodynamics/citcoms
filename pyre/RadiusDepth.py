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

class:
    Radius(radius, outer_radius=1)
    Depth(depth, outer_radius=1)

    both classes have 3 read-only attributes:
      outer_radius, inRadius, inDepth


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

__all__ = ['Radius', 'Depth']


class RadiusDepthBase(object):

    def __init__(self, outer_radius):
	self.Ro = outer_radius

    def get_outer_radius(self):
	return self.Ro

    outer_radius = property(get_outer_radius)



class Radius(RadiusDepthBase):

    def __init__(self, radius, outer_radius=1):
	if not 0 <= radius <= outer_radius:
	    raise ValueError, "radius out of range (0 - outer_radius)"
	RadiusDepthBase.__init__(self, outer_radius)
	self.radius = radius    

    def get_inRadius(self):
	return self.radius

    def get_inDepth(self):
	return self.outer_radius - self.radius

    inRadius = property(get_inRadius)

    inDepth = property(get_inDepth)

    __slots__ = ('inRadius', 'inDepth')


class Depth(RadiusDepthBase):

    def __init__(self, depth, outer_radius=1):
	if not 0 <= depth <= outer_radius:
	    raise ValueError, "depth out of range (0 - outer_radius)"
	RadiusDepthBase.__init__(self, outer_radius)
	self.depth = depth

    def get_inRadius(self):
	return self.outer_radius - self.depth

    def get_inDepth(self):
	return self.depth

    inRadius = property(get_inRadius)

    inDepth = property(get_inDepth)


####################################################

if __name__ == "__main__":

    print dir()
    
    earth_radius = 6371
    phase660 = Depth(660, earth_radius)
    
    # this will report depth of phase660
    print "in depth:", phase660.inDepth
    
    # this will report radius of phase660
    print "in radius:",  phase660.inRadius

    try:
	# this will raise an exception
	phase660.inRadius = 100
    except:
	print "exception catched"

    print dir(phase660)

    
    cmb = Radius(3480, earth_radius)
    print cmb.inDepth
    print cmb.inRadius
    print dir(cmb)


# version
__id__ = "$Id: RadiusDepth.py,v 1.3 2003/04/03 19:40:25 tan2 Exp $"

# End of file 
