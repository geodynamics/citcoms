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

#if !defined(pyacis_operators_h)
#define pyacis_operators_h

// boolean operators

extern char pyacis_union__doc__[];
extern char pyacis_union__name__[];
PyObject * pyacis_union(PyObject *, PyObject*);


extern char pyacis_difference__doc__[];
extern char pyacis_difference__name__[];
PyObject * pyacis_difference(PyObject *, PyObject*);

extern char pyacis_intersection__doc__[];
extern char pyacis_intersection__name__[];
PyObject * pyacis_intersection(PyObject *, PyObject*);

#endif

// End of file

