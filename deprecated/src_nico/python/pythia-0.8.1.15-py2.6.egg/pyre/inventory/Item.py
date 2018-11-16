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


class Item(object):
    """A Trait paired with its value and locator."""
    
    def __init__(self, trait, descriptor):
        self._trait = trait
        self._descriptor = descriptor

    attr     = property(lambda self: self._trait.attr)
    name     = property(lambda self: self._trait.name)
    default  = property(lambda self: self._trait.default)
    type     = property(lambda self: self._trait.type)
    meta     = property(lambda self: self._trait.meta)
    value    = property(lambda self: self._descriptor.value)
    locator  = property(lambda self: self._descriptor.locator)


# end of file
