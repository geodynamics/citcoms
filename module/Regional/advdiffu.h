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

extern char pyRegional_PG_timemarching_control__doc__[];
extern char pyRegional_PG_timemarching_control__name__[];
extern "C"
PyObject * pyRegional_PG_timemarching_control(PyObject *, PyObject *);

extern char pyRegional_PG_timestep_fini__doc__[];
extern char pyRegional_PG_timestep_fini__name__[];
extern "C"
PyObject * pyRegional_PG_timestep_fini(PyObject *, PyObject *);


#endif

// version
// $Id: advdiffu.h,v 1.1 2003/05/22 18:32:14 ces74 Exp $

// End of file
