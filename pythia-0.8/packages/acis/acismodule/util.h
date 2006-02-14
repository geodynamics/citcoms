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

#if !defined(pyacis_util_h)
#define pyacis_util_h

// miscellaneous utilities

extern char pyacis_setSaveFileVersion__doc__[];
extern char pyacis_setSaveFileVersion__name__[];
PyObject * pyacis_setSaveFileVersion(PyObject *, PyObject *);

extern char pyacis_setFileinfo__doc__[];
extern char pyacis_setFileinfo__name__[];
PyObject * pyacis_setFileinfo(PyObject *, PyObject *);

extern char pyacis_save__doc__[];
extern char pyacis_save__name__[];
PyObject * pyacis_save(PyObject *, PyObject*);

#endif

// version
// $Id: util.h,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
