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

#if !defined(pyjtest_misc_h)
#define pyjtest_misc_h

// copyright
extern char pyjtest_copyright__name__[];
extern char pyjtest_copyright__doc__[];
extern "C"
PyObject * pyjtest_copyright(PyObject *, PyObject *);

// info
extern char pyjtest_info__name__[];
extern char pyjtest_info__doc__[];
extern "C"
PyObject * pyjtest_info(PyObject *, PyObject *);

// error
extern char pyjtest_error__name__[];
extern char pyjtest_error__doc__[];
extern "C"
PyObject * pyjtest_error(PyObject *, PyObject *);

// info
extern char pyjtest_warning__name__[];
extern char pyjtest_warning__doc__[];
extern "C"
PyObject * pyjtest_warning(PyObject *, PyObject *);

#endif


// $Id: misc.h,v 1.1.1.1 2005/03/18 17:01:42 aivazis Exp $

// End of file
