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

#include "inlets_outlets.h"
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

    // from inlets_outlets.h

    {pyExchanger_Inlet_storeTimestep__name__,
     pyExchanger_Inlet_storeTimestep,
     METH_VARARGS,
     pyExchanger_Inlet_storeTimestep__doc__},

    {pyExchanger_BoundaryVTInlet_create__name__,
     pyExchanger_BoundaryVTInlet_create,
     METH_VARARGS,
     pyExchanger_BoundaryVTInlet_create__doc__},

    {pyExchanger_BoundaryVTInlet_impose__name__,
     pyExchanger_BoundaryVTInlet_impose,
     METH_VARARGS,
     pyExchanger_BoundaryVTInlet_impose__doc__},

    {pyExchanger_BoundaryVTInlet_recv__name__,
     pyExchanger_BoundaryVTInlet_recv,
     METH_VARARGS,
     pyExchanger_BoundaryVTInlet_recv__doc__},

    {pyExchanger_VTInlet_create__name__,
     pyExchanger_VTInlet_create,
     METH_VARARGS,
     pyExchanger_VTInlet_create__doc__},

    {pyExchanger_VTInlet_impose__name__,
     pyExchanger_VTInlet_impose,
     METH_VARARGS,
     pyExchanger_VTInlet_impose__doc__},

    {pyExchanger_VTInlet_recv__name__,
     pyExchanger_VTInlet_recv,
     METH_VARARGS,
     pyExchanger_VTInlet_recv__doc__},

    {pyExchanger_VTOutlet_create__name__,
     pyExchanger_VTOutlet_create,
     METH_VARARGS,
     pyExchanger_VTOutlet_create__doc__},

    {pyExchanger_VTOutlet_send__name__,
     pyExchanger_VTOutlet_send,
     METH_VARARGS,
     pyExchanger_VTOutlet_send__doc__},

    // from exchangers.h

    {pyExchanger_createTractionSource__name__,
     pyExchanger_createTractionSource,
     METH_VARARGS,
     pyExchanger_createTractionSource__doc__},

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

    {pyExchanger_createSink__name__,
     pyExchanger_createSink,
     METH_VARARGS,
     pyExchanger_createSink__doc__},

    {pyExchanger_VTSource_create__name__,
     pyExchanger_VTSource_create,
     METH_VARARGS,
     pyExchanger_VTSource_create__doc__},

    {pyExchanger_initConvertor__name__,
     pyExchanger_initConvertor,
     METH_VARARGS,
     pyExchanger_initConvertor__doc__},

    {pyExchanger_modifyT__name__,
     pyExchanger_modifyT,
     METH_VARARGS,
     pyExchanger_modifyT__doc__},

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

// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.37 2004/03/11 22:46:25 tan2 Exp $

// End of file
