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

#if !defined(pytabulator_tabulator_h)
#define pytabulator_tabulator_h

// tabulate
extern char pytabulator_tabulate__name__[];
extern char pytabulator_tabulate__doc__[];
extern "C"
PyObject * pytabulator_tabulate(PyObject *, PyObject *);

// simpletab
extern char pytabulator_simpletab__name__[];
extern char pytabulator_simpletab__doc__[];
extern "C"
PyObject * pytabulator_simpletab(PyObject *, PyObject *);

// exponential
extern char pytabulator_exponential__name__[];
extern char pytabulator_exponential__doc__[];
extern "C"
PyObject * pytabulator_exponential(PyObject *, PyObject *);

// exponentialSet
extern char pytabulator_exponentialSet__name__[];
extern char pytabulator_exponentialSet__doc__[];
extern "C"
PyObject * pytabulator_exponentialSet(PyObject *, PyObject *);

// exponential
extern char pytabulator_quadratic__name__[];
extern char pytabulator_quadratic__doc__[];
extern "C"
PyObject * pytabulator_quadratic(PyObject *, PyObject *);

// quadraticSet
extern char pytabulator_quadraticSet__name__[];
extern char pytabulator_quadraticSet__doc__[];
extern "C"
PyObject * pytabulator_quadraticSet(PyObject *, PyObject *);

#endif

// version
// $Id: tabulator.h,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

// End of file
