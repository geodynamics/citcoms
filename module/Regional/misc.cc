// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2003 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "misc.h"


extern "C" {
#include "global_defs.h"
#include "citcom_init.h"

double return1_test();
void read_instructions(char*);

struct All_variables *E;
}


// copyright

char pyRegional_copyright__doc__[] = "";
char pyRegional_copyright__name__[] = "copyright";

static char pyRegional_copyright_note[] = 
    "CitcomSRegional python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyRegional_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyRegional_copyright_note);
}



char pyRegional_return1_test__doc__[] = "";
char pyRegional_return1_test__name__[] = "return1_test";

PyObject * pyRegional_return1_test(PyObject *, PyObject *)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}


char pyRegional_Citcom_Init__doc__[] = "";
char pyRegional_Citcom_Init__name__[] = "Citcom_Init";

PyObject * pyRegional_Citcom_Init(PyObject *self, PyObject *args)
{
    PyObject *Obj;
    MPI_Comm *world;
    
    if (!PyArg_ParseTuple(args, "O", &Obj))
        return NULL;

    world = static_cast <MPI_Comm*> (PyCObject_AsVoidPtr(Obj));

    // Allocate global pointer E
    Citcom_Init(world);

    // if E is NULL, raise an exception here.
    if (E == NULL)
      return PyErr_NoMemory();
      

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_read_instructions__doc__[] = "";
char pyRegional_read_instructions__name__[] = "read_instructions";

PyObject * pyRegional_read_instructions(PyObject *self, PyObject *args)
{
    char *filename;

    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    read_instructions(filename);

    // test
    // fprintf(stderr,"output file prefix: %s\n", E->control.data_file);

    Py_INCREF(Py_None);
    return Py_None;
}




// version
// $Id: misc.cc,v 1.8 2003/05/13 19:51:11 tan2 Exp $

// End of file
