// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyExchanger_exchangers_h)
#define pyExchanger_exchangers_h


extern char pyExchanger_createCoarseGridExchanger__name__[];
extern char pyExchanger_createCoarseGridExchanger__doc__[];
extern "C"
PyObject * pyExchanger_createCoarseGridExchanger(PyObject *, PyObject *);


extern char pyExchanger_createFineGridExchanger__name__[];
extern char pyExchanger_createFineGridExchanger__doc__[];
extern "C"
PyObject * pyExchanger_createFineGridExchanger(PyObject *, PyObject *);


extern char pyExchanger_createBoundary__name__[];
extern char pyExchanger_createBoundary__doc__[];
extern "C"
PyObject * pyExchanger_createBoundary(PyObject *, PyObject *);


extern char pyExchanger_mapBoundary__name__[];
extern char pyExchanger_mapBoundary__doc__[];
extern "C"
PyObject * pyExchanger_mapBoundary(PyObject *, PyObject *);


extern char pyExchanger_receiveBoundary__name__[];
extern char pyExchanger_receiveBoundary__doc__[];
extern "C"
PyObject * pyExchanger_receiveBoundary(PyObject *, PyObject *);


extern char pyExchanger_sendBoundary__name__[];
extern char pyExchanger_sendBoundary__doc__[];
extern "C"
PyObject * pyExchanger_sendBoundary(PyObject *, PyObject *);


extern char pyExchanger_receiveTemperature__name__[];
extern char pyExchanger_receiveTemperature__doc__[];
extern "C"
PyObject * pyExchanger_receiveTemperature(PyObject *, PyObject *);


extern char pyExchanger_sendTemperature__name__[];
extern char pyExchanger_sendTemperature__doc__[];
extern "C"
PyObject * pyExchanger_sendTemperature(PyObject *, PyObject *);


extern char pyExchanger_distribute__name__[];
extern char pyExchanger_distribute__doc__[];
extern "C"
PyObject * pyExchanger_distribute(PyObject *, PyObject *);


extern char pyExchanger_gather__name__[];
extern char pyExchanger_gather__doc__[];
extern "C"
PyObject * pyExchanger_gather(PyObject *, PyObject *);


extern char pyExchanger_receive__name__[];
extern char pyExchanger_receive__doc__[];
extern "C"
PyObject * pyExchanger_receive(PyObject *, PyObject *);


extern char pyExchanger_send__name__[];
extern char pyExchanger_send__doc__[];
extern "C"
PyObject * pyExchanger_send(PyObject *, PyObject *);


extern char pyExchanger_exchangeTimestep__name__[];
extern char pyExchanger_exchangeTimestep__doc__[];
extern "C"
PyObject * pyExchanger_exchangeTimestep(PyObject *, PyObject *);


extern char pyExchanger_wait__name__[];
extern char pyExchanger_wait__doc__[];
extern "C"
PyObject * pyExchanger_wait(PyObject *, PyObject *);


extern char pyExchanger_nowait__name__[];
extern char pyExchanger_nowait__doc__[];
extern "C"
PyObject * pyExchanger_nowait(PyObject *, PyObject *);



#endif

// version
// $Id: exchangers.h,v 1.5 2003/09/10 04:03:54 tan2 Exp $

// End of file
