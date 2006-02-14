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
#include "debug.h"

// ACIS includes
#include <kernel/kerndata/top/lump.hxx>
#include <kernel/kerndata/top/shell.hxx>
#include <kernel/kerndata/top/subshell.hxx>
#include <kernel/kerndata/top/face.hxx>
#include <kernel/kerndata/top/query.hxx>


char pyacis_printBRep__name__[] = "printBRep";
char pyacis_printBRep__doc__[] = "print the BRep of the given body";
PyObject * pyacis_printBRep(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.debug");

    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "O:printBRep", &py_body);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);

    info
        << journal::at(__HERE__)
        << "printing brep of body@" << body << ":";

    for (LUMP * lump = body->lump(); lump; lump = lump->next()) {
        info 
            << journal::newline
            << "    lump@" << lump;

        for (SHELL * shell = lump->shell(); shell; shell = shell->next()) {
            info 
                << journal::newline
                << "        shell@" << shell;

            for (FACE * face = shell->face_list(); face; face = face->next()) {
                info 
                    << journal::newline
                    << "            face@" << face;
            }

        }
    }

    info << journal::endl;

#if 0
    //info.out(__HERE__, "lump@0x%08lx", body->lump());

    ENTITY_LIST face_list;
    ENTITY_LIST edge_list;
    ENTITY_LIST vert_list;

    body_face_edge_vert(body, face_list, edge_list, vert_list);


    ENTITY * face;
    face_list.init();
    while ((face = face_list.next())) {
        //info.out(__HERE__, "face@0x%08lx", face);
    }
#endif

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


char pyacis_printFaces__name__[] = "printFaces";
char pyacis_printFaces__doc__[] = "print the faces of the given body";
PyObject * pyacis_printFaces(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.debug");

    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "O:printFaces", &py_body);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);
    info
        << journal::at(__HERE__)
        << "printing faces of body@" << body << ":";


    for (FACE * face = body->lump()->shell()->face_list(); face; face = face->next_in_list()) {
        info 
            << journal::newline
            << "            face@" << face;
    }

    info << journal::endl;

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// version
// $Id: debug.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
