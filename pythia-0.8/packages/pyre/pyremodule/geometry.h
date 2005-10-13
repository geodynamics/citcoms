// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#if !defined(pyremodule_geometry_h)
#define pyremodule_geometry_h

// createMesh
extern char pyremodule_createMesh__name__[];
extern char pyremodule_createMesh__doc__[];
extern "C"
PyObject * pyremodule_createMesh(PyObject *, PyObject *);

// statistics
extern char pyremodule_statistics__name__[];
extern char pyremodule_statistics__doc__[];
extern "C"
PyObject * pyremodule_statistics(PyObject *, PyObject *);

// vertex
extern char pyremodule_vertex__name__[];
extern char pyremodule_vertex__doc__[];
extern "C"
PyObject * pyremodule_vertex(PyObject *, PyObject *);

// simplex
extern char pyremodule_simplex__name__[];
extern char pyremodule_simplex__doc__[];
extern "C"
PyObject * pyremodule_simplex(PyObject *, PyObject *);

#endif

// version
// $Id: geometry.h,v 1.1.1.1 2005/03/08 16:13:52 aivazis Exp $

// End of file
