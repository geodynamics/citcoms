#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.components.Component import Component


class Greeter(Component):

    class Inventory(Component.Inventory):

        import pyre.inventory

        greeting = pyre.inventory.str("greeting", default="Hello")


    def __init__(self):
        Component.__init__(self, name="greeter", facility="greeter")
        self.greeting = ''
        return


    def _configure(self):
        Component._configure(self)
        self.greeting = self.inventory.greeting
        return


    def _init(self):
        Component._init(self)
        return


    def _fini(self):
        Component._fini(self)
        return


from pyre.applications.Script import Script


class HelloApp(Script):


    class Inventory(Script.Inventory):

        import pyre.inventory

        name = pyre.inventory.str("name", default="Michael Aivazis")
        name.meta['tip'] = "the name of my friend"
        
        address = pyre.inventory.str("address")
        address.meta['tip'] = "the address of my friend"

        greeter = pyre.inventory.facility("greeter", default="morning")
        greeter.meta['tip'] = "the facility that manages the generated greeting"


    def main(self, *args, **kwds):
        curator = self.getCurator()

        self._debug.log("greeter: %s" % self.greeter)
        print '%s %s!' % (self.greeter.greeting, self.friend)

        return


    def __init__(self):
        Script.__init__(self, 'hello')
        self.friend = ''
        self.greeter = ''
        return


    def _configure(self):
        Script._configure(self)
        
        self.friend = self.inventory.name
        self.greeter = self.inventory.greeter
        return


# main
if __name__ == '__main__':
    app = HelloApp()
    app.run()


# version
__id__ = "$Id: hello.py,v 1.4 2005/03/10 21:35:37 aivazis Exp $"

# End of file 
