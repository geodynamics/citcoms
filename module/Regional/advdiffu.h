// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_advection_diffusion_h)
#define pyCitcom_advection_diffusion_h

// advection_diffustion routines
extern char pyCitcom_PG_timestep_init__doc__[];
extern char pyCitcom_PG_timestep_init__name__[];
extern "C"
PyObject * pyCitcom_PG_timestep_init(PyObject *, PyObject *);

extern char pyCitcom_PG_timestep_solve__doc__[];
extern char pyCitcom_PG_timestep_solve__name__[];
extern "C"
PyObject * pyCitcom_PG_timestep_solve(PyObject *, PyObject *);

extern char pyCitcom_set_convection_defaults__doc__[];
extern char pyCitcom_set_convection_defaults__name__[];
extern "C"
PyObject * pyCitcom_set_convection_defaults(PyObject *, PyObject *);

#endif

// version
// $Id: advdiffu.h,v 1.4 2003/08/01 22:53:50 tan2 Exp $

// End of file
