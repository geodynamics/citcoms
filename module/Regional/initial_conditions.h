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


extern char pyCitcom_ic_constructTemperature__name__[];
extern char pyCitcom_ic_constructTemperature__doc__[];
extern "C"
PyObject * pyCitcom_ic_constructTemperature(PyObject *, PyObject *);


extern char pyCitcom_ic_restartTemperature__name__[];
extern char pyCitcom_ic_restartTemperature__doc__[];
extern "C"
PyObject * pyCitcom_ic_restartTemperature(PyObject *, PyObject *);


extern char pyCitcom_ic_initPressure__name__[];
extern char pyCitcom_ic_initPressure__doc__[];
extern "C"
PyObject * pyCitcom_ic_initPressure(PyObject *, PyObject *);


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
// $Id: initial_conditions.h,v 1.2 2003/11/28 22:20:23 tan2 Exp $

// End of file
