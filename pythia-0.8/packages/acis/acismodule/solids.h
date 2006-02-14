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

#if !defined(pyacis_solids_h)
#define pyacis_solids_h

extern char pyacis_block__doc__[];
extern char pyacis_block__name__[];
PyObject * pyacis_block(PyObject *, PyObject *);

extern char pyacis_cone__doc__[];
extern char pyacis_cone__name__[];
PyObject * pyacis_cone(PyObject *, PyObject *);

extern char pyacis_cylinder__doc__[];
extern char pyacis_cylinder__name__[];
PyObject * pyacis_cylinder(PyObject *, PyObject *);

extern char pyacis_prism__doc__[];
extern char pyacis_prism__name__[];
PyObject * pyacis_prism(PyObject *, PyObject *);

extern char pyacis_pyramid__doc__[];
extern char pyacis_pyramid__name__[];
PyObject * pyacis_pyramid(PyObject *, PyObject *);

extern char pyacis_sphere__doc__[];
extern char pyacis_sphere__name__[];
PyObject * pyacis_sphere(PyObject *, PyObject *);

extern char pyacis_torus__doc__[];
extern char pyacis_torus__name__[];
PyObject * pyacis_torus(PyObject *, PyObject *);

extern char pyacis_generalizedCone__doc__[];
extern char pyacis_generalizedCone__name__[];
PyObject * pyacis_generalizedCone(PyObject *, PyObject *);

#endif

// End of file
