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


from pyre.inventory.odb.Curator import Curator as InventoryCurator


class Curator(InventoryCurator):


    # indices
    def compilers(self, language=None):
        candidates = self.vaults(address=['compilers'])
        if not language:
            return candidates

        return candidates

    
    def languages(self):
        languages = self.getShelves(address=['languages'], extension='odb')
        return languages


    def __init__(self, name=None):
        if name is None:
            name = 'merlin'
            
        InventoryCurator.__init__(self, name)

        return
            

    def _registerCodecs(self):
        InventoryCurator._registerCodecs(self)
        
        import pyre.odb
        codec = pyre.odb.odb(name="merlin")
        self.codecs[codec.encoding] = codec
        
        return


# version
__id__ = "$Id: Curator.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
