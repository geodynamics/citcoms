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
extern "C" struct All_variables* Citcom_Init();
extern "C" void read_instructions(char*);


struct All_variables *E;


// copyright

char pyCitcomSRegional_copyright__doc__[] = "";
char pyCitcomSRegional_copyright__name__[] = "copyright";

static char pyCitcomSRegional_copyright_note[] = 
    "CitcomSRegional python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyCitcomSRegional_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyCitcomSRegional_copyright_note);
}



char pyCitcomSRegional_return1_test__doc__[] = "";
char pyCitcomSRegional_return1_test__name__[] = "return1_test";

PyObject * pyCitcomSRegional_return1_test(PyObject *, PyObject *)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}


char pyCitcomSRegional_Citcom_Init__doc__[] = "";
char pyCitcomSRegional_Citcom_Init__name__[] = "Citcom_Init";

PyObject * pyCitcomSRegional_Citcom_Init(PyObject *, PyObject *)
{
    Citcom_Init();

    // if E is NULL, raise an exception here... to be done.

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcomSRegional_read_instructions__doc__[] = "";
char pyCitcomSRegional_read_instructions__name__[] = "read_instructions";

PyObject * pyCitcomSRegional_read_instructions(PyObject *self, PyObject *args)
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
// $Id: misc.cc,v 1.4 2003/04/05 23:51:35 tan2 Exp $

// End of file
