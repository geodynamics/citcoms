// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyRegional_outputs_h)
#define pyRegional_outputs_h

extern char pyRegional_output_init__name__[];
extern char pyRegional_output_init__doc__[];
extern "C"
PyObject * pyRegional_output_init(PyObject *, PyObject *);


extern char pyRegional_output_close__name__[];
extern char pyRegional_output_close__doc__[];
extern "C"
PyObject * pyRegional_output_close(PyObject *, PyObject *);


extern char pyRegional_output_coord__name__[];
extern char pyRegional_output_coord__doc__[];
extern "C"
PyObject * pyRegional_output_coord(PyObject *, PyObject *);


extern char pyRegional_output_velo_header__name__[];
extern char pyRegional_output_velo_header__doc__[];
extern "C"
PyObject * pyRegional_output_velo_header(PyObject *, PyObject *);


extern char pyRegional_output_velo__name__[];
extern char pyRegional_output_velo__doc__[];
extern "C"
PyObject * pyRegional_output_velo(PyObject *, PyObject *);


extern char pyRegional_output_visc_prepare__name__[];
extern char pyRegional_output_visc_prepare__doc__[];
extern "C"
PyObject * pyRegional_output_visc_prepare(PyObject *, PyObject *);


extern char pyRegional_output_visc__name__[];
extern char pyRegional_output_visc__doc__[];
extern "C"
PyObject * pyRegional_output_visc(PyObject *, PyObject *);






#endif

// version
// $Id: outputs.h,v 1.2 2003/05/22 23:08:59 tan2 Exp $

// End of file
