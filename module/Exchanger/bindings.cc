// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>

#include "exchangers.h"
#include "misc.h"          // miscellaneous methods

#include "bindings.h"

// the method table

struct PyMethodDef pyExchanger_methods[] = {

    // from misc.h

    {pyExchanger_hello__name__,
     pyExchanger_hello,
     METH_VARARGS,
     pyExchanger_hello__doc__},

    {pyExchanger_copyright__name__,
     pyExchanger_copyright,
     METH_VARARGS,
     pyExchanger_copyright__doc__},

    {pyExchanger_FinereturnE__name__,
     pyExchanger_FinereturnE,
     METH_VARARGS,
     pyExchanger_FinereturnE__doc__},

    {pyExchanger_CoarsereturnE__name__,
     pyExchanger_CoarsereturnE,
     METH_VARARGS,
     pyExchanger_CoarsereturnE__doc__},

    {pyExchanger_initTemperatureTest__name__,
     pyExchanger_initTemperatureTest,
     METH_VARARGS,
     pyExchanger_initTemperatureTest__doc__},

    // from exchangers.h

    {pyExchanger_createBCSink__name__,
     pyExchanger_createBCSink,
     METH_VARARGS,
     pyExchanger_createBCSink__doc__},

    {pyExchanger_createBCSource__name__,
     pyExchanger_createBCSource,
     METH_VARARGS,
     pyExchanger_createBCSource__doc__},

    {pyExchanger_createBoundary__name__,
     pyExchanger_createBoundary,
     METH_VARARGS,
     pyExchanger_createBoundary__doc__},

    {pyExchanger_createEmptyBoundary__name__,
     pyExchanger_createEmptyBoundary,
     METH_VARARGS,
     pyExchanger_createEmptyBoundary__doc__},

    {pyExchanger_createEmptyInterior__name__,
     pyExchanger_createEmptyInterior,
     METH_VARARGS,
     pyExchanger_createEmptyInterior__doc__},

    {pyExchanger_createGlobalBoundedBox__name__,
     pyExchanger_createGlobalBoundedBox,
     METH_VARARGS,
     pyExchanger_createGlobalBoundedBox__doc__},

    {pyExchanger_createInterior__name__,
     pyExchanger_createInterior,
     METH_VARARGS,
     pyExchanger_createInterior__doc__},

    {pyExchanger_initTemperature__name__,
     pyExchanger_initTemperature,
     METH_VARARGS,
     pyExchanger_initTemperature__doc__},

    {pyExchanger_createSink__name__,
     pyExchanger_createSink,
     METH_VARARGS,
     pyExchanger_createSink__doc__},

    {pyExchanger_createSource__name__,
     pyExchanger_createSource,
     METH_VARARGS,
     pyExchanger_createSource__doc__},

    {pyExchanger_recvTandV__name__,
     pyExchanger_recvTandV,
     METH_VARARGS,
     pyExchanger_recvTandV__doc__},

    {pyExchanger_sendTandV__name__,
     pyExchanger_sendTandV,
     METH_VARARGS,
     pyExchanger_sendTandV__doc__},

    {pyExchanger_recvT__name__,
     pyExchanger_recvT,
     METH_VARARGS,
     pyExchanger_recvT__doc__},

    {pyExchanger_sendT__name__,
     pyExchanger_sendT,
     METH_VARARGS,
     pyExchanger_sendTandV__doc__},

    {pyExchanger_imposeBC__name__,
     pyExchanger_imposeBC,
     METH_VARARGS,
     pyExchanger_imposeBC__doc__},

    {pyExchanger_imposeIC__name__,
     pyExchanger_imposeIC,
     METH_VARARGS,
     pyExchanger_imposeIC__doc__},

    {pyExchanger_exchangeBoundedBox__name__,
     pyExchanger_exchangeBoundedBox,
     METH_VARARGS,
     pyExchanger_exchangeBoundedBox__doc__},

    {pyExchanger_exchangeSignal__name__,
     pyExchanger_exchangeSignal,
     METH_VARARGS,
     pyExchanger_exchangeSignal__doc__},

    {pyExchanger_exchangeTimestep__name__,
     pyExchanger_exchangeTimestep,
     METH_VARARGS,
     pyExchanger_exchangeTimestep__doc__},

    {pyExchanger_storeTimestep__name__,
     pyExchanger_storeTimestep,
     METH_VARARGS,
     pyExchanger_storeTimestep__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.22 2003/11/07 21:43:47 puru Exp $

// End of file
