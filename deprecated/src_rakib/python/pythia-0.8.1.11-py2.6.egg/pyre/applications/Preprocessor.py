#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2007  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.components import Component
from pyre.util import expandMacros
from pyre.inventory.odb.Registry import Registry
import os, getpass, socket


class Preprocessor(Component):


    def updateConfiguration(self, registry):
        self.macros.update(registry)
        return []


    def expandMacros(self, value):
        return expandMacros(value, self.rp)


    def defineMacro(self, name, value):
        self.macros[name] = value
        return


    def updateMacros(self, dct):
        self.macros.update(dct)
        return


    def _init(self):
        
        class RecursionProtector(object):
            def __init__(self, macros):
                self.macros = macros
                self.recur = {}
            def __getitem__(self, key):
                if self.recur.has_key(key):
                    return ""
                self.recur[key] = True
                try:
                    value = expandMacros(self.macros[key], self)
                finally:
                    del self.recur[key]
                return value

        macros = {}
        
        # add environment variables
        macros.update(os.environ)

        # add useful macros
        macros['home'] = os.path.expanduser("~")
        if False:
            macros['user'] = os.getlogin() # OSError: [Errno 25] Inappropriate ioctl for device
        else:
            macros['user'] = getpass.getuser()
        macros['hostname'] = socket.gethostname()
        macros['wd'] = os.getcwd()
        macros['pid'] = str(os.getpid())

        # flatten my macro registry
        for path, value, locator in self.macros.allProperties():
            key = '.'.join(path[1:])
            macros[key] = value

        self.macros = macros
        self.rp = RecursionProtector(macros)

        return


    def __init__(self, name):
        Component.__init__(self, "macros", name)
        self.macros = Registry(self.name)
        self.rp = None
        return


# end of file
