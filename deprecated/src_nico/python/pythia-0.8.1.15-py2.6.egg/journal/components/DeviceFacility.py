#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.inventory.Facility import Facility


class DeviceFacility(Facility):


    def __init__(self, factory=None, args=[]):
        Facility.__init__(self, name="device", factory=factory, args=args,
                          vault=['devices'])


    def _getBuiltInDefaultValue(self, instance):
        
        import pyre.parsing.locators
        locator = pyre.parsing.locators.default()

        import sys
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            from os import environ
            term = environ.get('TERM', 'console')
            component = instance.retrieveComponent(term, factory=self.family, vault=self.vault)
            if component is None:
                from Console import Console
                component = Console()
            else:
                component.aliases.append(self.name)
                locator = pyre.parsing.locators.chain(component.getLocator(), locator)
        else:
            from Stream import Stream
            component = Stream(sys.stdout, "stdout")

        return component, locator


# version
__id__ = "$Id: DeviceFacility.py,v 1.1.1.1 2005/03/08 16:13:53 aivazis Exp $"

# End of file 
