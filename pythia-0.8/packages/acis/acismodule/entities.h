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

#if !defined(pyacis__entities_h)
#define pyacis__entities_h

extern char pyacis_box__doc__[];
extern char pyacis_box__name__[];
PyObject * pyacis_box(PyObject *, PyObject *);

extern char pyacis_faces__doc__[];
extern char pyacis_faces__name__[];
PyObject * pyacis_faces(PyObject *, PyObject *);

extern char pyacis_distance__doc__[];
extern char pyacis_distance__name__[];
PyObject * pyacis_distance(PyObject *, PyObject *);

extern char pyacis_touch__doc__[];
extern char pyacis_touch__name__[];
PyObject * pyacis_touch(PyObject *, PyObject *);

#endif

// version
// $Id: entities.h,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
