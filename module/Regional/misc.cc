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
#include "global_defs.h"
//#include "citcom_init.h"

extern "C" double return1_test();
extern "C" struct All_variables* Citcom_Init(int, int);
extern "C" void read_instructions(char*);


struct All_variables *E;


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
    int nproc, rank;
    
    if (!PyArg_ParseTuple(args, "ii", &nproc, &rank))
        return NULL;

    Citcom_Init(nproc, rank);

    // if E is NULL, raise an exception here... to be done.

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
    fprintf(stderr,"output file prefix: %s\n", E->control.data_file);

    Py_INCREF(Py_None);
    return Py_None;
}




// version
// $Id: misc.cc,v 1.6 2003/04/10 23:18:24 tan2 Exp $

// End of file
