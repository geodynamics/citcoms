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


extern char pyExchanger_createIISink__name__[];
extern char pyExchanger_createIISink__doc__[];
extern "C"
PyObject * pyExchanger_createIISink(PyObject *, PyObject *);


extern char pyExchanger_createIISource__name__[];
extern char pyExchanger_createIISource__doc__[];
extern "C"
PyObject * pyExchanger_createIISource(PyObject *, PyObject *);


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


extern char pyExchanger_recvTandV__name__[];
extern char pyExchanger_recvTandV__doc__[];
extern "C"
PyObject * pyExchanger_recvTandV(PyObject *, PyObject *);


extern char pyExchanger_sendTandV__name__[];
extern char pyExchanger_sendTandV__doc__[];
extern "C"
PyObject * pyExchanger_sendTandV(PyObject *, PyObject *);


extern char pyExchanger_sendTraction__name__[];
extern char pyExchanger_sendTraction__doc__[];
extern "C"
PyObject * pyExchanger_sendTraction(PyObject *, PyObject *);


extern char pyExchanger_domain_cutout__name__[];
extern char pyExchanger_domain_cutout__doc__[];
extern "C"
PyObject * pyExchanger_domain_cutout(PyObject *, PyObject *);


extern char pyExchanger_recvT__name__[];
extern char pyExchanger_recvT__doc__[];
extern "C"
PyObject * pyExchanger_recvT(PyObject *, PyObject *);

extern char pyExchanger_recvV__name__[];
extern char pyExchanger_recvV__doc__[];
extern "C"
PyObject * pyExchanger_recvV(PyObject *, PyObject *);


extern char pyExchanger_sendT__name__[];
extern char pyExchanger_sendT__doc__[];
extern "C"
PyObject * pyExchanger_sendT(PyObject *, PyObject *);


extern char pyExchanger_imposeBC__name__[];
extern char pyExchanger_imposeBC__doc__[];
extern "C"
PyObject * pyExchanger_imposeBC(PyObject *, PyObject *);


extern char pyExchanger_imposeIC__name__[];
extern char pyExchanger_imposeIC__doc__[];
extern "C"
PyObject * pyExchanger_imposeIC(PyObject *, PyObject *);


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
// $Id: exchangers.h,v 1.27 2004/01/07 21:54:00 tan2 Exp $

// End of file
