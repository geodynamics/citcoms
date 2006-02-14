#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


from pyre.weaver.components.BlockMill import BlockMill

class CMill(BlockMill):


    names = ["c"]


    def __init__(self):
        BlockMill.__init__(self, "/*", " *", " */", "/*  -*- C -*-  */")
        return


# version
__id__ = "$Id: CMill.py,v 1.1.1.1 2005/03/08 16:13:47 aivazis Exp $"

#  End of file 
