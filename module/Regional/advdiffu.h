// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyRegional_advection_diffusion_h)
#define pyRegional_advection_diffusion_h

// advection_diffustion routines
extern char pyRegional_PG_timestep_init__doc__[];
extern char pyRegional_PG_timestep_init__name__[];
extern "C"
PyObject * pyRegional_PG_timestep_init(PyObject *, PyObject *);

extern char pyRegional_PG_timestep_solve__doc__[];
extern char pyRegional_PG_timestep_solve__name__[];
extern "C"
PyObject * pyRegional_PG_timestep_solve(PyObject *, PyObject *);

extern char pyRegional_set_convection_defaults__doc__[];
extern char pyRegional_set_convection_defaults__name__[];
extern "C"
PyObject * pyRegional_set_convection_defaults(PyObject *, PyObject *);

#endif

// version
// $Id: advdiffu.h,v 1.3 2003/07/24 00:04:04 tan2 Exp $

// End of file
