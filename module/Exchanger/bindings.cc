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

    {pyExchanger_receiveTemperature__name__,
     pyExchanger_receiveTemperature,
     METH_VARARGS,
     pyExchanger_receiveTemperature__doc__},

    {pyExchanger_sendTemperature__name__,
     pyExchanger_sendTemperature,
     METH_VARARGS,
     pyExchanger_sendTemperature__doc__},

    {pyExchanger_distribute__name__,
     pyExchanger_distribute,
     METH_VARARGS,
     pyExchanger_distribute__doc__},

    {pyExchanger_gather__name__,
     pyExchanger_gather,
     METH_VARARGS,
     pyExchanger_gather__doc__},

    {pyExchanger_receive__name__,
     pyExchanger_receive,
     METH_VARARGS,
     pyExchanger_receive__doc__},

    {pyExchanger_send__name__,
     pyExchanger_send,
     METH_VARARGS,
     pyExchanger_send__doc__},

    {pyExchanger_exchangeTimestep__name__,
     pyExchanger_exchangeTimestep,
     METH_VARARGS,
     pyExchanger_exchangeTimestep__doc__},

    {pyExchanger_wait__name__,
     pyExchanger_wait,
     METH_VARARGS,
     pyExchanger_wait__doc__},

    {pyExchanger_nowait__name__,
     pyExchanger_nowait,
     METH_VARARGS,
     pyExchanger_nowait__doc__},




// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.7 2003/09/18 22:03:48 ces74 Exp $

// End of file
