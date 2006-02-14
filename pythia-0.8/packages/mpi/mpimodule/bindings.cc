// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "bindings.h"

#include "misc.h"          // miscellaneous methods
#include "groups.h"        // communicator group methods
#include "communicators.h" // communicator methods
#include "ports.h"         // sends and receives

// The method table

struct PyMethodDef pympi_methods[] = {

// communicators
    {pympi_communicatorCreate__name__, pympi_communicatorCreate, 
         METH_VARARGS, pympi_communicatorCreate__doc__},
    {pympi_communicatorSize__name__, pympi_communicatorSize, 
         METH_VARARGS, pympi_communicatorSize__doc__},
    {pympi_communicatorRank__name__, pympi_communicatorRank, 
         METH_VARARGS, pympi_communicatorRank__doc__},
    {pympi_communicatorBarrier__name__, pympi_communicatorBarrier, 
         METH_VARARGS, pympi_communicatorBarrier__doc__},

    {pympi_communicatorCreateCartesian__name__, pympi_communicatorCreateCartesian, 
         METH_VARARGS, pympi_communicatorCreateCartesian__doc__},
    {pympi_communicatorCartesianCoordinates__name__, pympi_communicatorCartesianCoordinates, 
         METH_VARARGS, pympi_communicatorCartesianCoordinates__doc__},

// groups
    {pympi_groupCreate__name__, pympi_groupCreate, METH_VARARGS, pympi_groupCreate__doc__},
    {pympi_groupSize__name__, pympi_groupSize, METH_VARARGS, pympi_groupSize__doc__},
    {pympi_groupRank__name__, pympi_groupRank, METH_VARARGS, pympi_groupRank__doc__},
    {pympi_groupInclude__name__, pympi_groupInclude, METH_VARARGS, pympi_groupInclude__doc__},
    {pympi_groupExclude__name__, pympi_groupExclude, METH_VARARGS, pympi_groupExclude__doc__},

// misc
    {pympi_wtime__name__, pympi_wtime, METH_VARARGS, pympi_wtime__doc__},
    {pympi_copyright__name__, pympi_copyright, METH_VARARGS, pympi_copyright__doc__},

// ports
    {pympi_sendString__name__, pympi_sendString, METH_VARARGS, pympi_sendString__doc__},
    {pympi_receiveString__name__, pympi_receiveString, METH_VARARGS, pympi_receiveString__doc__},

// sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
