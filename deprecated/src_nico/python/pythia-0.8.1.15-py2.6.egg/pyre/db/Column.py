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


class Column(object):


    def type(self):
        raise NotImplementedError("class %r must override 'type'" % self.__class__.__name__)


    def declaration(self):
        text = [ self.type() ]
        if self.default:
            text.append("DEFAULT %s" % self.default)
        if self.constraints:
            text.append(self.constraints)

        return " ".join(text)


    def __init__(self, name, default=None, auto=False, constraints=None, meta=None):
        self.name = name
        self.default = default
        self.auto = auto
        self.constraints = constraints

        if meta is None:
            meta = {}
        self.meta = meta

        return


    def __get__(self, instance, cls=None):

        # attempt to get hold of the instance's attribute record
        try:
            return instance._getColumnValue(self.name)

        # instance is None when accessed as a class variable
        except AttributeError:
            # catch bad descriptors or changes in the python conventions
            if instance is not None:
                import journal
                firewall = journal.firewall("pyre.inventory")
                firewall.log("AttributeError on non-None instance. Bad descriptor?")

            # interpret this usage as a request for the trait object itself
            return self

        except KeyError:
            # column value is uinitialized
            return None

        # not reachable
        import journal
        journal.firewall('pyre.db').log("UNREACHABLE")
        return None


    def __set__(self, instance, value):
        return instance._setColumnValue(self.name, value)


# version
__id__ = "$Id: Column.py,v 1.5 2005/04/08 18:11:23 aivazis Exp $"

# End of file 
