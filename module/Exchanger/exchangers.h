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


extern char pyExchanger_createBCSink__name__[];
extern char pyExchanger_createBCSink__doc__[];
extern "C"
PyObject * pyExchanger_createBCSink(PyObject *, PyObject *);


extern char pyExchanger_createBCSource__name__[];
extern char pyExchanger_createBCSource__doc__[];
extern "C"
PyObject * pyExchanger_createBCSource(PyObject *, PyObject *);


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


extern char pyExchanger_createSource__name__[];
extern char pyExchanger_createSource__doc__[];
extern "C"
PyObject * pyExchanger_createSource(PyObject *, PyObject *);


extern char pyExchanger_initTemperature__name__[];
extern char pyExchanger_initTemperature__doc__[];
extern "C"
PyObject * pyExchanger_initTemperature(PyObject *, PyObject *);


extern char pyExchanger_recvTandV__name__[];
extern char pyExchanger_recvTandV__doc__[];
extern "C"
PyObject * pyExchanger_recvTandV(PyObject *, PyObject *);


extern char pyExchanger_sendTandV__name__[];
extern char pyExchanger_sendTandV__doc__[];
extern "C"
PyObject * pyExchanger_sendTandV(PyObject *, PyObject *);


extern char pyExchanger_imposeBC__name__[];
extern char pyExchanger_imposeBC__doc__[];
extern "C"
PyObject * pyExchanger_imposeBC(PyObject *, PyObject *);


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


extern char pyExchanger_storeTimestep__name__[];
extern char pyExchanger_storeTimestep__doc__[];
extern "C"
PyObject * pyExchanger_storeTimestep(PyObject *, PyObject *);



#endif

// version
// $Id: exchangers.h,v 1.19 2003/11/07 01:08:01 tan2 Exp $

// End of file
