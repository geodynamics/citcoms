// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "journal/debug.h"

#include "memory.h"

// helpers
extern "C" void _deleteField(void *);

// allocateField

char pyelc_allocateField__doc__[] = "";
char pyelc_allocateField__name__[] = "allocateField";

PyObject * pyelc_allocateField(PyObject *, PyObject *args)
{
    int rank, length;

    int ok = PyArg_ParseTuple(args, "ii:allocateField", &rank, &length); 

    if (!ok) {
        return 0;
    }

    // allocate the memory
    double * field = new double[rank*length];

    journal::debug_t debug("elc.memory");
    debug
        << journal::at(__HERE__)
        << "allocated field@" << field
        << ", rank=" << rank << ", length=" << length << " (" << rank*length << " doubles)"
        << journal::endl;

    // return
    return PyCObject_FromVoidPtr(field, _deleteField);
}
    
// helpers
void _deleteField(void * object)
{
    double * field = (double *) object;

    journal::debug_t debug("elc.memory");
    debug
        << journal::at(__HERE__)
        << "deleting field@" << field
        << journal::endl;

    delete [] field;

    return;
}


// version
// $Id: memory.cc,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $

// End of file
