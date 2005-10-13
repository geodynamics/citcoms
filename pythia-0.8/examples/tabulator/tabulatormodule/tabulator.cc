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

#include "tabulator.h"
#include "tabulator_externs.h"


// tabulate
char pytabulator_tabulate__doc__[] = "";
char pytabulator_tabulate__name__[] = "tabulate";

PyObject * pytabulator_tabulate(PyObject *, PyObject * args)
{
    PyObject * py_func;
    double low, high, step;

    int ok = PyArg_ParseTuple(args, "dddO:tabulate", &low, &high, &step, &py_func);
    if (!ok) {
        return 0;
    }

    model_t model = (model_t) PyCObject_AsVoidPtr(py_func);

    tabulator_f(&low, &high, &step, model);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
    

// simpletab
char pytabulator_simpletab__doc__[] = "";
char pytabulator_simpletab__name__[] = "simpletab";

PyObject * pytabulator_simpletab(PyObject *, PyObject * args)
{
    double a, low, high, step;

    int ok = PyArg_ParseTuple(args, "dddd:simpletab", &a, &low, &high, &step);
    if (!ok) {
        return 0;
    }

    simpletab_f(&a, &low, &high, &step);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
    

// quadratic

char pytabulator_quadratic__doc__[] = "";
char pytabulator_quadratic__name__[] = "quadratic";

PyObject * pytabulator_quadratic(PyObject *, PyObject * args)
{
    int ok = PyArg_ParseTuple(args, ":quadratic");
    if (!ok) {
        return 0;
    }

    return PyCObject_FromVoidPtr((void *)(quadratic_f), 0);
}
    
// quadraticSet

char pytabulator_quadraticSet__doc__[] = "";
char pytabulator_quadraticSet__name__[] = "quadraticSet";

PyObject * pytabulator_quadraticSet(PyObject *, PyObject * args)
{
    double a, b, c;

    int ok = PyArg_ParseTuple(args, "ddd:quadraticSet", &a, &b, &c);
    if (!ok) {
        return 0;
    }

    quadratic_set_f(&a, &b, &c);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
    
// exponential

char pytabulator_exponential__doc__[] = "";
char pytabulator_exponential__name__[] = "exponential";

PyObject * pytabulator_exponential(PyObject *, PyObject * args)
{
    int ok = PyArg_ParseTuple(args, ":exponential");
    if (!ok) {
        return 0;
    }

    return PyCObject_FromVoidPtr((void *)(exponential_f), 0);
}
    
// exponentialSet

char pytabulator_exponentialSet__doc__[] = "";
char pytabulator_exponentialSet__name__[] = "exponentialSet";

PyObject * pytabulator_exponentialSet(PyObject *, PyObject * args)
{
    double a;

    int ok = PyArg_ParseTuple(args, "d:exponentialSet", &a);
    if (!ok) {
        return 0;
    }

    exponential_set_f(&a);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
    
// version
// $Id: tabulator.cc,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

// End of file
