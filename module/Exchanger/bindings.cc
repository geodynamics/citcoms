// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>

#include "bindings.h"

#include "exchangers.h"
#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pyExchanger_methods[] = {

    // dummy entry for testing
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

    // from exchangers.h

    {pyExchanger_createCoarseGridExchanger__name__,
     pyExchanger_createCoarseGridExchanger,
     METH_VARARGS,
     pyExchanger_createCoarseGridExchanger__doc__},

    {pyExchanger_createFineGridExchanger__name__,
     pyExchanger_createFineGridExchanger,
     METH_VARARGS,
     pyExchanger_createFineGridExchanger__doc__},

    {pyExchanger_createBoundary__name__,
     pyExchanger_createBoundary,
     METH_VARARGS,
     pyExchanger_createBoundary__doc__},

    {pyExchanger_mapBoundary__name__,
     pyExchanger_mapBoundary,
     METH_VARARGS,
     pyExchanger_mapBoundary__doc__},

    {pyExchanger_receiveBoundary__name__,
     pyExchanger_receiveBoundary,
     METH_VARARGS,
     pyExchanger_receiveBoundary__doc__},

    {pyExchanger_sendBoundary__name__,
     pyExchanger_sendBoundary,
     METH_VARARGS,
     pyExchanger_sendBoundary__doc__},

    {pyExchanger_createDataArrays__name__,
     pyExchanger_createDataArrays,
     METH_VARARGS,
     pyExchanger_createDataArrays__doc__},

    {pyExchanger_deleteDataArrays__name__,
     pyExchanger_deleteDataArrays,
     METH_VARARGS,
     pyExchanger_deleteDataArrays__doc__},

    {pyExchanger_initTemperature__name__,
     pyExchanger_initTemperature,
     METH_VARARGS,
     pyExchanger_initTemperature__doc__},

    {pyExchanger_receiveTemperature__name__,
     pyExchanger_receiveTemperature,
     METH_VARARGS,
     pyExchanger_receiveTemperature__doc__},

    {pyExchanger_sendTemperature__name__,
     pyExchanger_sendTemperature,
     METH_VARARGS,
     pyExchanger_sendTemperature__doc__},

    {pyExchanger_receiveVelocities__name__,
     pyExchanger_receiveVelocities,
     METH_VARARGS,
     pyExchanger_receiveVelocities__doc__},

    {pyExchanger_sendVelocities__name__,
     pyExchanger_sendVelocities,
     METH_VARARGS,
     pyExchanger_sendVelocities__doc__},

    {pyExchanger_distribute__name__,
     pyExchanger_distribute,
     METH_VARARGS,
     pyExchanger_distribute__doc__},

    {pyExchanger_gather__name__,
     pyExchanger_gather,
     METH_VARARGS,
     pyExchanger_gather__doc__},

    {pyExchanger_imposeBC__name__,
     pyExchanger_imposeBC,
     METH_VARARGS,
     pyExchanger_imposeBC__doc__},

    {pyExchanger_storeTimestep__name__,
     pyExchanger_storeTimestep,
     METH_VARARGS,
     pyExchanger_storeTimestep__doc__},

    {pyExchanger_exchangeTimestep__name__,
     pyExchanger_exchangeTimestep,
     METH_VARARGS,
     pyExchanger_exchangeTimestep__doc__},

    {pyExchanger_exchangeSignal__name__,
     pyExchanger_exchangeSignal,
     METH_VARARGS,
     pyExchanger_exchangeSignal__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.17 2003/10/01 22:21:14 tan2 Exp $

// End of file
