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

#if !defined(pyjournal_facility_h)
#define pyjournal_facility_h

// firewall
extern char pyjournal_firewall__name__[];
extern char pyjournal_firewall__doc__[];
extern "C"
PyObject * pyjournal_firewall(PyObject *, PyObject *);

// debug
extern char pyjournal_debug__name__[];
extern char pyjournal_debug__doc__[];
extern "C"
PyObject * pyjournal_debug(PyObject *, PyObject *);

// info
extern char pyjournal_info__name__[];
extern char pyjournal_info__doc__[];
extern "C"
PyObject * pyjournal_info(PyObject *, PyObject *);

// warning
extern char pyjournal_warning__name__[];
extern char pyjournal_warning__doc__[];
extern "C"
PyObject * pyjournal_warning(PyObject *, PyObject *);

// error
extern char pyjournal_error__name__[];
extern char pyjournal_error__doc__[];
extern "C"
PyObject * pyjournal_error(PyObject *, PyObject *);

#endif

// version
// $Id: facility.h,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file
