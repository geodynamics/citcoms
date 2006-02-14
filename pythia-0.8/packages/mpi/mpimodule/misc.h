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

#if !defined(pympi_misc_h)
#define pympi_misc_h

// wtime
extern char pympi_wtime__name__[];
extern char pympi_wtime__doc__[];
extern "C"
PyObject * pympi_wtime(PyObject *, PyObject *);

// copyright
extern char pympi_copyright__name__[];
extern char pympi_copyright__doc__[];
extern "C"
PyObject * pympi_copyright(PyObject *, PyObject *);

#endif

// version
// $Id: misc.h,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
