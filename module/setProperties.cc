// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "global_defs.h"

#include "setProperties.h"


void getStringProperty(PyObject* properties, char* attribute,
		       char* value, int mute);

template <class T>
void getScalarProperty(PyObject* properties, char* attribute,
		       T& value, int mute);

template <class T>
void getVectorProperty(PyObject* properties, char* attribute,
		       T* vector, int len, int mute);

//
//

char pyCitcom_Advection_diffusion_set_properties__doc__[] = "";
char pyCitcom_Advection_diffusion_set_properties__name__[] = "Advection_diffusion_set_properties";

PyObject * pyCitcom_Advection_diffusion_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Advection_diffusion_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Advection_diffusion.inventory:" << std::endl;

    getScalarProperty(properties, "inputdiffusivity", E->control.inputdiff, m);

    getScalarProperty(properties, "ADV", E->advection.ADVECTION, m);
    getScalarProperty(properties, "fixed_timestep", E->advection.fixed_timestep, m);
    getScalarProperty(properties, "finetunedt", E->advection.fine_tune_dt, m);

    getScalarProperty(properties, "adv_sub_iterations", E->advection.temp_iterations, m);
    getScalarProperty(properties, "maxadvtime", E->advection.max_dimensionless_time, m);

    getScalarProperty(properties, "aug_lagr", E->control.augmented_Lagr, m);
    getScalarProperty(properties, "aug_number", E->control.augmented, m);

    getScalarProperty(properties, "filter_temp", E->control.filter_temperature, m);

    E->advection.total_timesteps = 1;
    E->advection.sub_iterations = 1;
    E->advection.last_sub_iterations = 1;
    E->advection.gamma = 0.5;
    E->advection.dt_reduced = 1.0;

    E->monitor.T_maxvaried = 1.05;

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_BC_set_properties__doc__[] = "";
char pyCitcom_BC_set_properties__name__[] = "BC_set_properties";

PyObject * pyCitcom_BC_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:BC_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#BC.inventory:" << std::endl;

    getScalarProperty(properties, "side_sbcs", E->control.side_sbcs, m);

    getScalarProperty(properties, "topvbc", E->mesh.topvbc, m);
    getScalarProperty(properties, "topvbxval", E->control.VBXtopval, m);
    getScalarProperty(properties, "topvbyval", E->control.VBYtopval, m);

    getScalarProperty(properties, "pseudo_free_surf", E->control.pseudo_free_surf, m);

    getScalarProperty(properties, "botvbc", E->mesh.botvbc, m);
    getScalarProperty(properties, "botvbxval", E->control.VBXbotval, m);
    getScalarProperty(properties, "botvbyval", E->control.VBYbotval, m);

    getScalarProperty(properties, "toptbc", E->mesh.toptbc, m);
    getScalarProperty(properties, "toptbcval", E->control.TBCtopval, m);

    getScalarProperty(properties, "bottbc", E->mesh.bottbc, m);
    getScalarProperty(properties, "bottbcval", E->control.TBCbotval, m);

    getScalarProperty(properties, "temperature_bound_adj", E->control.temperature_bound_adj, m);
    getScalarProperty(properties, "depth_bound_adj", E->control.depth_bound_adj, m);
    getScalarProperty(properties, "width_bound_adj", E->control.width_bound_adj, m);


    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Const_set_properties__doc__[] = "";
char pyCitcom_Const_set_properties__name__[] = "Const_set_properties";

PyObject * pyCitcom_Const_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Const_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Const.inventory:" << std::endl;

    float radius;
    getScalarProperty(properties, "layerd", radius, m);
    getScalarProperty(properties, "density", E->data.density, m);
    getScalarProperty(properties, "thermdiff", E->data.therm_diff, m);
    getScalarProperty(properties, "gravacc", E->data.grav_acc, m);
    getScalarProperty(properties, "thermexp", E->data.therm_exp, m);
    getScalarProperty(properties, "refvisc", E->data.ref_viscosity, m);
    getScalarProperty(properties, "cp", E->data.Cp, m);
    getScalarProperty(properties, "wdensity", E->data.density_above, m);
    getScalarProperty(properties, "surftemp", E->data.surf_temp, m);

    E->data.therm_cond = E->data.therm_diff * E->data.density * E->data.Cp;
    E->data.ref_temperature = E->control.Atemp * E->data.therm_diff
	* E->data.ref_viscosity / (radius * radius * radius)
	/ (E->data.density * E->data.grav_acc * E->data.therm_exp);

    getScalarProperty(properties, "z_lith", E->viscosity.zlith, m);
    getScalarProperty(properties, "z_410", E->viscosity.z410, m);
    getScalarProperty(properties, "z_lmantle", E->viscosity.zlm, m);
    getScalarProperty(properties, "z_cmb", E->viscosity.zcmb, m); //this is used as the D" phase change depth

    // convert meter to kilometer
    E->data.layer_km = radius / 1e3;
    E->data.radius_km = E->data.layer_km;

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_IC_set_properties__doc__[] = "";
char pyCitcom_IC_set_properties__name__[] = "IC_set_properties";

PyObject * pyCitcom_IC_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:IC_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#IC.inventory:" << std::endl;

    getScalarProperty(properties, "restart", E->control.restart, m);
    getScalarProperty(properties, "post_p", E->control.post_p, m);
    getScalarProperty(properties, "solution_cycles_init", E->monitor.solution_cycles_init, m);
    getScalarProperty(properties, "zero_elapsed_time", E->control.zero_elapsed_time, m);

    getScalarProperty(properties, "tic_method", E->convection.tic_method, m);

    if (E->convection.tic_method == 0) {
	int num_perturb;

	getScalarProperty(properties, "num_perturbations", num_perturb, m);
	if(num_perturb > PERTURB_MAX_LAYERS) {
	    std::cerr << "'num_perturb' greater than allowed value, set to "
		      << PERTURB_MAX_LAYERS << std::endl;
	    num_perturb = PERTURB_MAX_LAYERS;
	}
	E->convection.number_of_perturbations = num_perturb;

	getVectorProperty(properties, "perturbl", E->convection.perturb_ll,
			  num_perturb, m);
	getVectorProperty(properties, "perturbm", E->convection.perturb_mm,
			  num_perturb, m);
	getVectorProperty(properties, "perturblayer", E->convection.load_depth,
			  num_perturb, m);
	getVectorProperty(properties, "perturbmag", E->convection.perturb_mag,
			  num_perturb, m);
    }
    else if (E->convection.tic_method == 1) {
	getScalarProperty(properties, "half_space_age", E->convection.half_space_age, m);
    }
    else if (E->convection.tic_method == 2) {
        getScalarProperty(properties, "half_space_age", E->convection.half_space_age, m);
        getVectorProperty(properties, "blob_center", E->convection.blob_center, 3, m);
        if( E->convection.blob_center[0] == -999.0 && E->convection.blob_center[1] == -999.0 && E->convection.blob_center[2] == -999.0 ) {
            E->convection.blob_center[0] = 0.5*(E->control.theta_min+E->control.theta_max);
            E->convection.blob_center[1] = 0.5*(E->control.fi_min+E->control.fi_max);
            E->convection.blob_center[2] = 0.5*(E->sphere.ri+E->sphere.ro);
        }
        getScalarProperty(properties, "blob_radius", E->convection.blob_radius, m);
        getScalarProperty(properties, "blob_dT", E->convection.blob_dT, m);
    }
    if (PyErr_Occurred())
      return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Param_set_properties__doc__[] = "";
char pyCitcom_Param_set_properties__name__[] = "Param_set_properties";

PyObject * pyCitcom_Param_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Param_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Param.inventory:" << std::endl;

    getScalarProperty(properties, "file_vbcs", E->control.vbcs_file, m);
    getStringProperty(properties, "vel_bound_file", E->control.velocity_boundary_file, m);

    getScalarProperty(properties, "mat_control", E->control.mat_control, m);
    getStringProperty(properties, "mat_file", E->control.mat_file, m);

    getScalarProperty(properties, "lith_age", E->control.lith_age, m);
    getStringProperty(properties, "lith_age_file", E->control.lith_age_file, m);
    getScalarProperty(properties, "lith_age_time", E->control.lith_age_time, m);
    getScalarProperty(properties, "lith_age_depth", E->control.lith_age_depth, m);
    getScalarProperty(properties, "mantle_temp", E->control.lith_age_mantle_temp, m);

    getScalarProperty(properties, "start_age", E->control.start_age, m);
    getScalarProperty(properties, "reset_startage", E->control.reset_startage, m);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_Phase_set_properties__doc__[] = "";
char pyCitcom_Phase_set_properties__name__[] = "Phase_set_properties";

PyObject * pyCitcom_Phase_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Phase_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Phase.inventory:" << std::endl;

    getScalarProperty(properties, "Ra_410", E->control.Ra_410, m);
    getScalarProperty(properties, "clapeyron410", E->control.clapeyron410, m);
    getScalarProperty(properties, "transT410", E->control.transT410, m);
    getScalarProperty(properties, "width410", E->control.width410, m);

    if (E->control.width410!=0.0)
	E->control.width410 = 1.0/E->control.width410;

    getScalarProperty(properties, "Ra_670", E->control.Ra_670 , m);
    getScalarProperty(properties, "clapeyron670", E->control.clapeyron670, m);
    getScalarProperty(properties, "transT670", E->control.transT670, m);
    getScalarProperty(properties, "width670", E->control.width670, m);

    if (E->control.width670!=0.0)
	E->control.width670 = 1.0/E->control.width670;

    getScalarProperty(properties, "Ra_cmb", E->control.Ra_cmb, m);
    getScalarProperty(properties, "clapeyroncmb", E->control.clapeyroncmb, m);
    getScalarProperty(properties, "transTcmb", E->control.transTcmb, m);
    getScalarProperty(properties, "widthcmb", E->control.widthcmb, m);

    if (E->control.widthcmb!=0.0)
	E->control.widthcmb = 1.0/E->control.widthcmb;

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_Solver_set_properties__doc__[] = "";
char pyCitcom_Solver_set_properties__name__[] = "Solver_set_properties";

PyObject * pyCitcom_Solver_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Solver_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Solver.inventory:" << std::endl;

    getStringProperty(properties, "datafile", E->control.data_file, m);
    getStringProperty(properties, "datafile_old", E->control.old_P_file, m);

    getScalarProperty(properties, "rayleigh", E->control.Atemp, m);
    getScalarProperty(properties, "Q0", E->control.Q0, m);

    getScalarProperty(properties, "stokes_flow_only", E->control.stokes, m);

    getStringProperty(properties, "output_format", E->output.format, m);
    getStringProperty(properties, "output_optional", E->output.optional, m);

    getScalarProperty(properties, "verbose", E->control.verbose, m);
    getScalarProperty(properties, "see_convergence", E->control.print_convergence, m);

    // parameters not used in pyre version,
    // assigned value here to prevent uninitialized access
    E->advection.min_timesteps = 1;
    E->advection.max_timesteps = 1;
    E->advection.max_total_timesteps = 1;
    E->control.record_every = 1;
    E->control.record_all_until = 1;

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Sphere_set_properties__doc__[] = "";
char pyCitcom_Sphere_set_properties__name__[] = "Sphere_set_properties";

PyObject * pyCitcom_Sphere_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Sphere_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Sphere.inventory:" << std::endl;

    getScalarProperty(properties, "nproc_surf", E->parallel.nprocxy, m);
    getScalarProperty(properties, "nprocx", E->parallel.nprocx, m);
    getScalarProperty(properties, "nprocy", E->parallel.nprocy, m);
    getScalarProperty(properties, "nprocz", E->parallel.nprocz, m);

    if (E->parallel.nprocxy == 12)
	if (E->parallel.nprocx != E->parallel.nprocy) {
	    char errmsg[] = "!!!! nprocx must equal to nprocy";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
    }

    getScalarProperty(properties, "coor", E->control.coor, m);
    getStringProperty(properties, "coor_file", E->control.coor_file, m);

    getScalarProperty(properties, "nodex", E->mesh.nox, m);
    getScalarProperty(properties, "nodey", E->mesh.noy, m);
    getScalarProperty(properties, "nodez", E->mesh.noz, m);
    getScalarProperty(properties, "levels", E->mesh.levels, m);

    E->mesh.mgunitx = (E->mesh.nox - 1) / E->parallel.nprocx /
	(int) std::pow(2.0, E->mesh.levels - 1);
    E->mesh.mgunity = (E->mesh.noy - 1) / E->parallel.nprocy /
	(int) std::pow(2.0, E->mesh.levels - 1);
    E->mesh.mgunitz = (E->mesh.noz - 1) / E->parallel.nprocz /
	(int) std::pow(2.0, E->mesh.levels - 1);

    if (E->parallel.nprocxy == 12) {
	if (E->mesh.nox != E->mesh.noy) {
	    char errmsg[] = "!!!! nodex must equal to nodey";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
	}
    }

    getScalarProperty(properties, "radius_outer", E->sphere.ro, m);
    getScalarProperty(properties, "radius_inner", E->sphere.ri, m);

    E->mesh.nsd = 3;
    E->mesh.dof = 3;
    E->sphere.max_connections = 6;

    if (E->parallel.nprocxy == 12) {

	E->sphere.caps = 12;

	int i, j;
	double offset = 10.0/180.0*M_PI;
	for (i=1;i<=4;i++)  {
	    E->sphere.cap[(i-1)*3+1].theta[1] = 0.0;
	    E->sphere.cap[(i-1)*3+1].theta[2] = M_PI/4.0+offset;
	    E->sphere.cap[(i-1)*3+1].theta[3] = M_PI/2.0;
	    E->sphere.cap[(i-1)*3+1].theta[4] = M_PI/4.0+offset;
	    E->sphere.cap[(i-1)*3+1].fi[1] = 0.0;
	    E->sphere.cap[(i-1)*3+1].fi[2] = (i-1)*M_PI/2.0;
	    E->sphere.cap[(i-1)*3+1].fi[3] = (i-1)*M_PI/2.0 + M_PI/4.0;
	    E->sphere.cap[(i-1)*3+1].fi[4] = i*M_PI/2.0;

	    E->sphere.cap[(i-1)*3+2].theta[1] = M_PI/4.0+offset;
	    E->sphere.cap[(i-1)*3+2].theta[2] = M_PI/2.0;
	    E->sphere.cap[(i-1)*3+2].theta[3] = 3*M_PI/4.0-offset;
	    E->sphere.cap[(i-1)*3+2].theta[4] = M_PI/2.0;
	    E->sphere.cap[(i-1)*3+2].fi[1] = i*M_PI/2.0;
	    E->sphere.cap[(i-1)*3+2].fi[2] = i*M_PI/2.0 - M_PI/4.0;
	    E->sphere.cap[(i-1)*3+2].fi[3] = i*M_PI/2.0;
	    E->sphere.cap[(i-1)*3+2].fi[4] = i*M_PI/2.0 + M_PI/4.0;
	}

	for (i=1;i<=4;i++)  {
	    j = (i-1)*3;
	    if (i==1) j=12;
	    E->sphere.cap[j].theta[1] = M_PI/2.0;
	    E->sphere.cap[j].theta[2] = 3*M_PI/4.0-offset;
	    E->sphere.cap[j].theta[3] = M_PI;
	    E->sphere.cap[j].theta[4] = 3*M_PI/4.0-offset;
	    E->sphere.cap[j].fi[1] = (i-1)*M_PI/2.0 + M_PI/4.0;
	    E->sphere.cap[j].fi[2] = (i-1)*M_PI/2.0;
	    E->sphere.cap[j].fi[3] = 0.0;
	    E->sphere.cap[j].fi[4] = i*M_PI/2.0;
	}

    } else {

	E->sphere.caps = 1;

	getScalarProperty(properties, "theta_min", E->control.theta_min, m);
	getScalarProperty(properties, "theta_max", E->control.theta_max, m);
	getScalarProperty(properties, "fi_min", E->control.fi_min, m);
	getScalarProperty(properties, "fi_max", E->control.fi_max, m);

	E->sphere.cap[1].theta[1] = E->control.theta_min;
	E->sphere.cap[1].theta[2] = E->control.theta_max;
	E->sphere.cap[1].theta[3] = E->control.theta_max;
	E->sphere.cap[1].theta[4] = E->control.theta_min;
	E->sphere.cap[1].fi[1] = E->control.fi_min;
	E->sphere.cap[1].fi[2] = E->control.fi_min;
	E->sphere.cap[1].fi[3] = E->control.fi_max;
	E->sphere.cap[1].fi[4] = E->control.fi_max;
    }

    getScalarProperty(properties, "ll_max", E->sphere.llmax, m);
    getScalarProperty(properties, "nlong", E->sphere.noy, m);
    getScalarProperty(properties, "nlati", E->sphere.nox, m);
    getScalarProperty(properties, "output_ll_max", E->sphere.output_llmax, m);

    E->mesh.layer[1] = 1;
    E->mesh.layer[2] = 1;
    E->mesh.layer[3] = 1;

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Tracer_set_properties__doc__[] = "";
char pyCitcom_Tracer_set_properties__name__[] = "Tracer_set_properties";

PyObject * pyCitcom_Tracer_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Tracer_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Tracer.inventory:" << std::endl;

    getScalarProperty(properties, "tracer", E->control.tracer, m);
    getStringProperty(properties, "tracer_file", E->control.tracer_file, m);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Visc_set_properties__doc__[] = "";
char pyCitcom_Visc_set_properties__name__[] = "Visc_set_properties";

PyObject * pyCitcom_Visc_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Visc_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Visc.inventory:" << std::endl;

    getStringProperty(properties, "Viscosity", E->viscosity.STRUCTURE, m);
    if ( strcmp(E->viscosity.STRUCTURE,"system") == 0)
	E->viscosity.FROM_SYSTEM = 1;
    else
	E->viscosity.FROM_SYSTEM = 0;

    getScalarProperty(properties, "visc_smooth_method", E->viscosity.smooth_cycles, m);
    getScalarProperty(properties, "VISC_UPDATE", E->viscosity.update_allowed, m);

    int num_mat;
    const int max_mat = 40;

    getScalarProperty(properties, "num_mat", num_mat, m);
    if(num_mat > max_mat) {
	// max. allowed material types = 40
	std::cerr << "'num_mat' greater than allowed value, set to "
		  << max_mat << std::endl;
	num_mat = max_mat;
    }
    E->viscosity.num_mat = num_mat;

    getVectorProperty(properties, "visc0",
			E->viscosity.N0, num_mat, m);

    getScalarProperty(properties, "TDEPV", E->viscosity.TDEPV, m);
    getScalarProperty(properties, "rheol", E->viscosity.RHEOL, m);
    getVectorProperty(properties, "viscE",
			E->viscosity.E, num_mat, m);
    getVectorProperty(properties, "viscT",
			E->viscosity.T, num_mat, m);
    getVectorProperty(properties, "viscZ",
			E->viscosity.Z, num_mat, m);

    getScalarProperty(properties, "SDEPV", E->viscosity.SDEPV, m);
    getScalarProperty(properties, "sdepv_misfit", E->viscosity.sdepv_misfit, m);
    getVectorProperty(properties, "sdepv_expt",
			E->viscosity.sdepv_expt, num_mat, m);

    getScalarProperty(properties, "VMIN", E->viscosity.MIN, m);
    getScalarProperty(properties, "visc_min", E->viscosity.min_value, m);

    getScalarProperty(properties, "VMAX", E->viscosity.MAX, m);
    getScalarProperty(properties, "visc_max", E->viscosity.max_value, m);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_Incompressible_set_properties__doc__[] = "";
char pyCitcom_Incompressible_set_properties__name__[] = "Incompressible_set_properties";

PyObject * pyCitcom_Incompressible_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties;

    if (!PyArg_ParseTuple(args, "OO:Incompressible_set_properties",
			  &obj, &properties))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    int m = E->parallel.me;
    if (not m)
	std::cout << "#Incompressible.inventory:" << std::endl;

    getScalarProperty(properties, "node_assemble", E->control.NASSEMBLE, m);
    getScalarProperty(properties, "precond", E->control.precondition, m);

    getScalarProperty(properties, "accuracy", E->control.accuracy, m);
    getScalarProperty(properties, "tole_compressibility", E->control.tole_comp, m);

    getScalarProperty(properties, "mg_cycle", E->control.mg_cycle, m);
    getScalarProperty(properties, "down_heavy", E->control.down_heavy, m);
    getScalarProperty(properties, "up_heavy", E->control.up_heavy, m);

    getScalarProperty(properties, "vlowstep", E->control.v_steps_low, m);
    getScalarProperty(properties, "vhighstep", E->control.v_steps_high, m);
    getScalarProperty(properties, "piterations", E->control.p_iterations, m);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}




//==========================================================
// helper functions


void getStringProperty(PyObject* properties, char* attribute, char* value, int mute)
{
    std::ofstream outf("/dev/null");
    std::ostream *out;

    if (mute)
	out = &outf;
    else
	out = &std::cout;

    *out << '\t' << attribute << "=";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	return;
    }

    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PyString_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a string", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return;
    }

    strcpy(value, PyString_AsString(prop));
    *out << '"' << value << '"' << std::endl;

    return;
}



template <class T>
void getScalarProperty(PyObject* properties, char* attribute, T& value, int mute)
{
    std::ofstream outf("/dev/null");
    std::ostream *out;

    if (mute)
	out = &outf;
    else
	out = &std::cout;

    *out << '\t' << attribute << "=";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	return;
    }

    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PyNumber_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a number", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return;
    }

    value = static_cast<T>(PyFloat_AsDouble(prop));
    *out << value << std::endl;

    return;
}



template <class T>
void getVectorProperty(PyObject* properties, char* attribute,
		       T* vector, const int len, int mute)
{
    std::ofstream outf("/dev/null");
    std::ostream *out;

    if (mute)
	out = &outf;
    else
	out = &std::cout;

    *out << '\t' << attribute << "=";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	return;
    }

    // is it a sequence?
    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PySequence_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a sequence", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return;
    }

    // is it of length len?
    int n = PySequence_Size(prop);
    if(n < len) {
	char errmsg[255];
	sprintf(errmsg, "length of '%s' < %d", attribute, len);
	PyErr_SetString(PyExc_IndexError, errmsg);
	return;
    } else if (n > len) {
	char warnmsg[255];
	sprintf(warnmsg, "WARNING: length of '%s' > %d", attribute, len);
	*out << warnmsg << std::endl;
    }

    for (int i=0; i<len; i++) {
	PyObject* item = PySequence_GetItem(prop, i);
	if(!item) {
	    char errmsg[255];
	    sprintf(errmsg, "can't get %s[%d]", attribute, i);
	    PyErr_SetString(PyExc_IndexError, errmsg);
	    return;
	}

	if(PyNumber_Check(item)) {
	    vector[i] = static_cast<T>(PyFloat_AsDouble(item));
	} else {
	    char errmsg[255];
	    sprintf(errmsg, "'%s[%d]' is not a number ", attribute, i);
	    PyErr_SetString(PyExc_TypeError, errmsg);
	    return;
	}
	*out << vector[i] << ",";
    }
    *out << std::endl;

    return;
}


// version
// $Id$

// End of file
