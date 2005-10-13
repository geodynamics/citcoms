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

#include "imports"

// local
#include "util.h"
#include "exceptions.h"
#include "support.h"

#include <kernel/kerndata/savres/fileinfo.hxx> 


char pyacis_setSaveFileVersion__name__[] = "setSaveFileVersion";
char pyacis_setSaveFileVersion__doc__[] = "set the required fileinfo fields";
PyObject * pyacis_setSaveFileVersion(PyObject *, PyObject *args)
{
    int major = 0;
    int minor = -1;

    int ok = PyArg_ParseTuple(args, "|ii:setSaveFileVersion", &major, &minor);
    if (!ok) {
        return 0;
    }

    set_save_file_version(major, minor);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


char pyacis_setFileinfo__name__[] = "fileinfo";
char pyacis_setFileinfo__doc__[] = "set the required fileinfo fields";
PyObject * pyacis_setFileinfo(PyObject *, PyObject *args)
{
    double units;
    char * product;
    int ok = PyArg_ParseTuple(args, "sd:fileinfo", &product, &units);
    if (!ok) {
        return 0;
    }

    FileInfo info;

    info.set_product_id(product);
    info.set_units(units);

    api_set_file_info(FileId | FileUnits, info);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


char pyacis_save__name__[] = "save";
char pyacis_save__doc__[] = "save a body in a file in SAT format";
PyObject * pyacis_save(PyObject *, PyObject *args)
{
    int flag;
    PyObject * py_bodies;
    PyObject * py_file;

    int ok = PyArg_ParseTuple(
        args, 
        "O!Oi:save", &PyFile_Type, &py_file, &py_bodies, &flag);

    if (!ok) {
        return 0;
    }


    if (!PySequence_Check(py_bodies)) {
        PyErr_SetString(PyExc_TypeError, "save() argument 3 must be a sequence");
        return 0;
    }

    FILE * file = PyFile_AsFile(py_file);

    ENTITY_LIST entities;

    for (int i = 0; i < PySequence_Length(py_bodies); ++i) {
        ENTITY * body = (ENTITY *) PyCObject_AsVoidPtr(PySequence_GetItem(py_bodies, i));
        entities.add(body);
    }

    outcome check = api_save_entity_list(file, flag, entities);
    if (!check.ok()) {
        throwACISError(check, "saving", pyacis_runtimeError);
        return 0;
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// version
// $Id: util.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
