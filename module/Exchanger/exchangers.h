// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyExchanger_exchangers_h)
#define pyExchanger_exchangers_h


extern char pyExchanger_createBoundary__name__[];
extern char pyExchanger_createBoundary__doc__[];
extern "C"
PyObject * pyExchanger_createBoundary(PyObject *, PyObject *);


extern char pyExchanger_createEmptyBoundary__name__[];
extern char pyExchanger_createEmptyBoundary__doc__[];
extern "C"
PyObject * pyExchanger_createEmptyBoundary(PyObject *, PyObject *);


extern char pyExchanger_createEmptyInterior__name__[];
extern char pyExchanger_createEmptyInterior__doc__[];
extern "C"
PyObject * pyExchanger_createEmptyInterior(PyObject *, PyObject *);


extern char pyExchanger_createGlobalBoundedBox__name__[];
extern char pyExchanger_createGlobalBoundedBox__doc__[];
extern "C"
PyObject * pyExchanger_createGlobalBoundedBox(PyObject *, PyObject *);


extern char pyExchanger_createInterior__name__[];
extern char pyExchanger_createInterior__doc__[];
extern "C"
PyObject * pyExchanger_createInterior(PyObject *, PyObject *);


extern char pyExchanger_createSink__name__[];
extern char pyExchanger_createSink__doc__[];
extern "C"
PyObject * pyExchanger_createSink(PyObject *, PyObject *);


extern char pyExchanger_createTractionSource__name__[];
extern char pyExchanger_createTractionSource__doc__[];
extern "C"
PyObject * pyExchanger_createTractionSource(PyObject *, PyObject *);


extern char pyExchanger_VTSource_create__name__[];
extern char pyExchanger_VTSource_create__doc__[];
extern "C"
PyObject * pyExchanger_VTSource_create(PyObject *, PyObject *);


extern char pyExchanger_initConvertor__name__[];
extern char pyExchanger_initConvertor__doc__[];
extern "C"
PyObject * pyExchanger_initConvertor(PyObject *, PyObject *);


extern char pyExchanger_initTemperatureSink__name__[];
extern char pyExchanger_initTemperatureSink__doc__[];
extern "C"
PyObject * pyExchanger_initTemperatureSink(PyObject *, PyObject *);


extern char pyExchanger_initTemperatureSource__name__[];
extern char pyExchanger_initTemperatureSource__doc__[];
extern "C"
PyObject * pyExchanger_initTemperatureSource(PyObject *, PyObject *);


extern char pyExchanger_modifyT__name__[];
extern char pyExchanger_modifyT__doc__[];
extern "C"
PyObject * pyExchanger_modifyT(PyObject *, PyObject *);


extern char pyExchanger_exchangeBoundedBox__name__[];
extern char pyExchanger_exchangeBoundedBox__doc__[];
extern "C"
PyObject * pyExchanger_exchangeBoundedBox(PyObject *, PyObject *);


extern char pyExchanger_exchangeSignal__name__[];
extern char pyExchanger_exchangeSignal__doc__[];
extern "C"
PyObject * pyExchanger_exchangeSignal(PyObject *, PyObject *);


extern char pyExchanger_exchangeTimestep__name__[];
extern char pyExchanger_exchangeTimestep__doc__[];
extern "C"
PyObject * pyExchanger_exchangeTimestep(PyObject *, PyObject *);


#endif

// version
// $Id: exchangers.h,v 1.33 2004/03/11 22:46:25 tan2 Exp $

// End of file
