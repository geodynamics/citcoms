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


    class Inventory(Renderer.Inventory):

        import journal.colors

        colorScheme = journal.colors.colorScheme("color-scheme", default="dark-bg")


    def _init(self):
        renderer = Renderer._init(self)
        renderer.colorScheme = self.inventory.colorScheme
        return renderer


    def createRenderer(self):
        from journal.devices.ColorRenderer import ColorRenderer
        return ColorRenderer()


# end of file 
