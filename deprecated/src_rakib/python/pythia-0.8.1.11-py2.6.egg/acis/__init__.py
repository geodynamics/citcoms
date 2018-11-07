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


def facet(body, options=None):
    from Faceter import Faceter
    faceter = Faceter(options)

    return faceter.facet(body)

    
def mesh(body):
    rep = pickle(body)

    from Faceter import Faceter
    faceter = Faceter()

    return faceter.mesh(rep)


def save(file, bodies, format=1):
    import acis

    entities = [ pickle(body).handle() for body in  bodies ]
    acis.save(file, entities, format)
    return

    
def pickle(body):

    from Pickler import Pickler
    pickler = Pickler()
    return pickler.pickle(body)


def surfaceMesher():
    from Faceter import Faceter
    return Faceter()


def copyright():
    return "acis: Copyright (c) 1998-2005 Michael A.G. Aivazis"


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $"

#
# End of file
