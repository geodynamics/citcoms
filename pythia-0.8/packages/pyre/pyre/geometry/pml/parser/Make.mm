# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = pyre
PACKAGE = geometry/pml/parser


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    AbstractNode.py \
    Angle.py \
    Binary.py \
    Block.py \
    Composition.py \
    Cone.py \
    Cylinder.py \
    Difference.py \
    Dilation.py \
    Document.py \
    GeneralizedCone.py \
    Geometry.py \
    Intersection.py \
    Prism.py \
    Pyramid.py \
    Reflection.py \
    Reversal.py \
    Rotation.py \
    Scale.py \
    Sphere.py \
    Torus.py \
    Transformation.py \
    Translation.py \
    Union.py \
    Vector.py \
    __init__.py \


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:45 aivazis Exp $

# End of file
