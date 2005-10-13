// -*- C++ -*-
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>

#include "bindings.h"
#include "misc.h"

// bindings
#include "attributes.h"
#include "debug.h"
#include "entities.h"
#include "faceting.h"
#include "intersections.h"
#include "meshing.h"
#include "solids.h"
#include "operators.h"
#include "transformations.h"
#include "util.h"

// method table
PyMethodDef pyacis_methods[] = {
    // debug
    {pyacis_printBRep__name__, pyacis_printBRep, 
     METH_VARARGS, pyacis_printBRep__doc__},
    {pyacis_printFaces__name__, pyacis_printFaces, 
     METH_VARARGS, pyacis_printFaces__doc__},

    // util
    {pyacis_save__name__, pyacis_save, METH_VARARGS, pyacis_save__doc__},
    {pyacis_setFileinfo__name__, pyacis_setFileinfo, METH_VARARGS, pyacis_setFileinfo__doc__},
    {pyacis_setSaveFileVersion__name__, pyacis_setSaveFileVersion, 
     METH_VARARGS, pyacis_setSaveFileVersion__doc__},

    // solids
    {pyacis_block__name__, pyacis_block, METH_VARARGS, pyacis_block__doc__},
    {pyacis_cone__name__, pyacis_cone, METH_VARARGS, pyacis_cone__doc__},
    {pyacis_cylinder__name__, pyacis_cylinder, METH_VARARGS, pyacis_cylinder__doc__},
    {pyacis_prism__name__, pyacis_prism, METH_VARARGS, pyacis_prism__doc__},
    {pyacis_pyramid__name__, pyacis_pyramid, METH_VARARGS, pyacis_pyramid__doc__},
    {pyacis_sphere__name__, pyacis_sphere, METH_VARARGS, pyacis_sphere__doc__},
    {pyacis_torus__name__, pyacis_torus, METH_VARARGS, pyacis_torus__doc__},
    {pyacis_generalizedCone__name__, pyacis_generalizedCone, METH_VARARGS,
     pyacis_generalizedCone__doc__},

    // boolean operators
    {pyacis_union__name__, pyacis_union, METH_VARARGS, pyacis_union__doc__},
    {pyacis_difference__name__, pyacis_difference, METH_VARARGS, pyacis_difference__doc__},
    {pyacis_intersection__name__, pyacis_intersection, METH_VARARGS, pyacis_intersection__doc__},

    // transformations
    {pyacis_dilation__name__, pyacis_dilation, METH_VARARGS, pyacis_dilation__doc__},
    {pyacis_reflection__name__, pyacis_reflection, METH_VARARGS, pyacis_reflection__doc__},
    {pyacis_reversal__name__, pyacis_reversal, METH_VARARGS, pyacis_reversal__doc__},
    {pyacis_rotation__name__, pyacis_rotation, METH_VARARGS, pyacis_rotation__doc__},
    {pyacis_translation__name__, pyacis_translation, METH_VARARGS, pyacis_translation__doc__},

    // intersections
    {pyacis_facesIntersectQ__name__, pyacis_facesIntersectQ, 
     METH_VARARGS, pyacis_facesIntersectQ__doc__},
    {pyacis_bodiesIntersectQ__name__, pyacis_bodiesIntersectQ, 
     METH_VARARGS, pyacis_bodiesIntersectQ__doc__},

    // attributes
    {pyacis_setAttributeInt__name__, pyacis_setAttributeInt,
     METH_VARARGS, pyacis_setAttributeInt__doc__},
    {pyacis_setAttributeDouble__name__, pyacis_setAttributeDouble,
     METH_VARARGS, pyacis_setAttributeDouble__doc__},
    {pyacis_setAttributeString__name__, pyacis_setAttributeString,
     METH_VARARGS, pyacis_setAttributeString__doc__},

    // entities
    {pyacis_box__name__, pyacis_box, METH_VARARGS, pyacis_box__doc__},
    {pyacis_faces__name__, pyacis_faces, METH_VARARGS, pyacis_faces__doc__},
    {pyacis_distance__name__, pyacis_distance, METH_VARARGS, pyacis_distance__doc__},
    {pyacis_touch__name__, pyacis_touch, METH_VARARGS, pyacis_touch__doc__},

#ifdef ACIS_HAS_MESHER
    // meshing
    {pyacis_mesh__name__, pyacis_mesh, METH_VARARGS, pyacis_mesh__doc__},
#endif

    // faceting
    {pyacis_facet__name__, pyacis_facet, METH_VARARGS, pyacis_facet__doc__},

    // misc
    {pyacis_copyright__name__, pyacis_copyright, METH_VARARGS, pyacis_copyright__doc__},

    // sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
