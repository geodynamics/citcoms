#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.applications.Script import Script


class Component(Script):


    class Inventory(Script.Inventory):

        import pyre.inventory

        name = pyre.inventory.str("name", default="component")
        name.meta['tip'] = "the name of the component"
        
        facility = pyre.inventory.str("facility", default="facility")
        facility.meta['tip'] = "the facility this component implements"
        
        base = pyre.inventory.str("base", default="Component")
        facility.meta["tip"] = "the name of the base class for this component"


    def main(self, *args, **kwds):

        self.weaver.begin()
        self.weaver.contents(self._template())
        self.weaver.end()

        name = self.inventory.name
        filename = name + '.py'
        print "creating component '%s' in '%s'" % (name, filename)

        stream = file(filename, "w")
        for line in self.weaver.document():
            print >> stream, line
        stream.close()
        
        return


    def __init__(self):
        Script.__init__(self, "app")
        return


    def _init(self):
        Script._init(self)
        self.weaver.language = 'python'
        return


    def _template(self):
        base = self.inventory.base
        name = self.inventory.name
        facility = self.inventory.facility

        if self.inventory.base == "Component":
            importStmt = "from pyre.components.Component import Component"
        else:
            importStmt = "from %s import %s" % (self.inventory.base, self.inventory.base)
        
        text = [
            "",
            "",
            importStmt,
            "",
            "",
            "class %s(%s):" % (name, base),
            "",
            "",
            "    class Inventory(%s.Inventory):" % base,
            "",
            "        import pyre.inventory",
            "",
            "",
            "    def __init__(self, name):",
            "        if name is None:",
            "            name = '%s'" % facility,
            "",
            "        %s.__init__(self, name, facility=%r)" % (base, facility),
            "",
            "        return",
            "",
            "",
            "    def _defaults(self):",
            "        %s._defaults(self)" % base,
            "        return",
            "",
            "",
            "    def _configure(self):",
            "        %s._configure(self)" % base,
            "        return",
            "",
            "",
            "    def _init(self):",
            "        %s._init(self)" % base,
            "        return",
            "",
            ]

        return text


# main

if __name__ == "__main__":
    app = Component()
    app.run()


# version
__id__ = "$Id: component.py,v 1.2 2005/03/18 21:07:59 aivazis Exp $"

# End of file
