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

#endif

// version
// $Id: advdiffu.h,v 1.2 2003/05/23 05:12:29 ces74 Exp $

// End of file
