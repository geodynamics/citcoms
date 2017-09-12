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


from Console import Console


class ColorConsole(Console):


    class Inventory(Console.Inventory):

        from RendererFacility import RendererFacility
        from ColorRenderer import ColorRenderer

        renderer = RendererFacility(factory=ColorRenderer)
        renderer.meta['tip'] = 'the facility that controls how the messages are formatted'


    def createDevice(self):
        from journal.devices.ANSIColorConsole import ANSIColorConsole
        return ANSIColorConsole()


# end of file 
