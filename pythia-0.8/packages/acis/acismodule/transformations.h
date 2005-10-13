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

#if !defined(pyacis_transformations_h)
#define pyacis_transformations_h

// transformations

extern char pyacis_dilation__doc__[];
extern char pyacis_dilation__name__[];
PyObject * pyacis_dilation(PyObject *, PyObject*);

extern char pyacis_reflection__doc__[];
extern char pyacis_reflection__name__[];
PyObject * pyacis_reflection(PyObject *, PyObject*);

extern char pyacis_rotation__doc__[];
extern char pyacis_rotation__name__[];
PyObject * pyacis_rotation(PyObject *, PyObject*);

extern char pyacis_translation__doc__[];
extern char pyacis_translation__name__[];
PyObject * pyacis_translation(PyObject *, PyObject*);

extern char pyacis_reversal__doc__[];
extern char pyacis_reversal__name__[];
PyObject * pyacis_reversal(PyObject *, PyObject*);

#endif

// version
// $Id: transformations.h,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
