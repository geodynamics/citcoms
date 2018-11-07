#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Renderer import Renderer


class ColorRenderer(Renderer):


    def subst(self, dct, device):
        
        TermColors = device.TermColors
        colorScheme = self.colorScheme
        
        colorized = {}
        for k, v in dct.iteritems():
            colorKey = k
            if colorKey == 'severity':
                colorKey = colorKey + '-' + v
            try:
                colorized[k] = (
                    getattr(TermColors, colorScheme[colorKey]) +
                    str(v) +
                    getattr(TermColors, colorScheme['normal']))
            except AttributeError:
                colorized[k] = v
        
        return colorized


    def __init__(self, header=None, format=None, footer=None, colorScheme=None):
        Renderer.__init__(self, header, format, footer)
        self.colorScheme = colorScheme


# end of file 
