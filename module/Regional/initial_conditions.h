// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_initial_conditions_h)
#define pyCitcom_initial_conditions_h


extern char pyCitcom_ic_initPressure__name__[];
extern char pyCitcom_ic_initPressure__doc__[];
extern "C"
PyObject * pyCitcom_ic_initPressure(PyObject *, PyObject *);


extern char pyCitcom_ic_initTemperature__name__[];
extern char pyCitcom_ic_initTemperature__doc__[];
extern "C"
PyObject * pyCitcom_ic_initTemperature(PyObject *, PyObject *);


extern char pyCitcom_ic_initVelocity__name__[];
extern char pyCitcom_ic_initVelocity__doc__[];
extern "C"
PyObject * pyCitcom_ic_initVelocity(PyObject *, PyObject *);


extern char pyCitcom_ic_initViscosity__name__[];
extern char pyCitcom_ic_initViscosity__doc__[];
extern "C"
PyObject * pyCitcom_ic_initViscosity(PyObject *, PyObject *);

#endif

// version
// $Id: initial_conditions.h,v 1.1 2003/10/29 18:40:00 tan2 Exp $

// End of file
