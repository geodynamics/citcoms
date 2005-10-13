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

#if !defined(pyjournal_state_h)
#define pyjournal_state_h

// state
extern char pyjournal_getState__name__[];
extern char pyjournal_getState__doc__[];
extern "C" PyObject * pyjournal_getState(PyObject *, PyObject *);

extern char pyjournal_setState__name__[];
extern char pyjournal_setState__doc__[];
extern "C" PyObject * pyjournal_setState(PyObject *, PyObject *);

extern char pyjournal_activate__name__[];
extern char pyjournal_activate__doc__[];
extern "C" PyObject * pyjournal_activate(PyObject *, PyObject *);

extern char pyjournal_deactivate__name__[];
extern char pyjournal_deactivate__doc__[];
extern "C" PyObject * pyjournal_deactivate(PyObject *, PyObject *);

extern char pyjournal_flip__name__[];
extern char pyjournal_flip__doc__[];
extern "C" PyObject * pyjournal_flip(PyObject *, PyObject *);

#endif

// version
// $Id: state.h,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file
