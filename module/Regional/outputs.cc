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
#include <cstdio>
#include <iostream>

#include "outputs.h"

extern "C" {
#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"
#include "output.h"

}


char pyRegional_output__doc__[] = "";
char pyRegional_output__name__[] = "output";

PyObject * pyRegional_output(PyObject *self, PyObject *args)
{
    int cycles;

    if (!PyArg_ParseTuple(args, "i:output_coord", &cycles))
        return NULL;

    std::cerr << "cycles = " << cycles << std::endl;
    output(E, cycles);

    Py_INCREF(Py_None);
    return Py_None;
}


#if 0
char pyRegional_output_velo_header__doc__[] = "";
char pyRegional_output_velo_header__name__[] = "output_velo_header";

PyObject * pyRegional_output_velo_header(PyObject *self, PyObject *args)
{
    PyObject *obj;
    FILE *fp;
    int step;

    if (!PyArg_ParseTuple(args, "Oi:output_velo_header", &obj, &step))
        return NULL;

    fp = static_cast<FILE*> (PyCObject_AsVoidPtr(obj));
    output_velo_header(E, fp, step);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_output_velo__doc__[] = "";
char pyRegional_output_velo__name__[] = "output_velo";

PyObject * pyRegional_output_velo(PyObject *self, PyObject *args)
{
    PyObject *obj;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "O:output_velo", &obj))
        return NULL;

    fp = static_cast<FILE*> (PyCObject_AsVoidPtr(obj));
    output_velo(E, fp);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_output_visc_prepare__doc__[] = "";
char pyRegional_output_visc_prepare__name__[] = "output_visc_prepare";

PyObject * pyRegional_output_visc_prepare(PyObject *self, PyObject *args)
{
    float **pt;
    PyObject *obj;
    float **visc;

    if (!PyArg_ParseTuple(args, "O:output_visc_prepare", &obj))
        return NULL;

    visc = static_cast<float**> (PyCObject_AsVoidPtr(obj));

    output_visc_prepare(E,visc);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_output_visc__doc__[] = "";
char pyRegional_output_visc__name__[] = "output_visc";

PyObject * pyRegional_output_visc(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2;
    FILE *fp;
    float **visc;

    if (!PyArg_ParseTuple(args, "OO:output_visc", &obj1, &obj2))
        return NULL;

    fp = static_cast<FILE*> (PyCObject_AsVoidPtr(obj1));
    visc = static_cast<float**> (PyCObject_AsVoidPtr(obj2));
    output_visc(E, fp, visc);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_output_surface_prepare__doc__[] = "";
char pyRegional_output_surface_prepare__name__[] = "output_surface_prepare";

PyObject * pyRegional_output_surface_prepare(PyObject *self, PyObject *args)
{
    float **pt;
    PyObject *obj;

    pt = output_surface_prepare(E);
    obj = PyCObject_FromVoidPtr((void *)pt, NULL);

    return Py_BuildValue("O", obj);
}


char pyRegional_output_surface__doc__[] = "";
char pyRegional_output_surface__name__[] = "output_surface";

PyObject * pyRegional_output_surface(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2;
    FILE *fp;
    float **surface;

    if (!PyArg_ParseTuple(args, "OO:output_surface", &obj1, &obj2))
        return NULL;

    fp = static_cast<FILE*> (PyCObject_AsVoidPtr(obj1));
    surface = static_cast<float**> (PyCObject_AsVoidPtr(obj2));
    output_surface(E, fp, surface);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_output_bottom_prepare__doc__[] = "";
char pyRegional_output_bottom_prepare__name__[] = "output_bottom_prepare";

PyObject * pyRegional_output_bottom_prepare(PyObject *self, PyObject *args)
{
    float **pt;
    PyObject *obj;

    pt = output_bottom_prepare(E);
    obj = PyCObject_FromVoidPtr((void *)pt, NULL);

    return Py_BuildValue("O", obj);
}


char pyRegional_output_bottom__doc__[] = "";
char pyRegional_output_bottom__name__[] = "output_bottom";

PyObject * pyRegional_output_bottom(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2;
    FILE *fp;
    float **bottom;

    if (!PyArg_ParseTuple(args, "OO:output_bottom", &obj1, &obj2))
        return NULL;

    fp = static_cast<FILE*> (PyCObject_AsVoidPtr(obj1));
    bottom = static_cast<float**> (PyCObject_AsVoidPtr(obj2));
    output_bottom(E, fp, bottom);

    Py_INCREF(Py_None);
    return Py_None;
}

#endif


// version
// $Id: outputs.cc,v 1.7 2003/07/26 21:47:51 tan2 Exp $

// End of file
