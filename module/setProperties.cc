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


std::ofstream* get_output_stream(int mute, int pid);

void getStringProperty(PyObject* properties, char* attribute,
		       char* value, std::ostream* out);

template <class T>
void getScalarProperty(PyObject* properties, char* attribute,
		       T& value, std::ostream* out);

template <class T>
void getVectorProperty(PyObject* properties, char* attribute,
		       T* vector, int len, std::ostream* out);

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.tsolver]" << std::endl;

    getScalarProperty(properties, "ADV", E->advection.ADVECTION, out);
    getScalarProperty(properties, "filter_temp", E->control.filter_temperature, out);

    getScalarProperty(properties, "finetunedt", E->advection.fine_tune_dt, out);
    getScalarProperty(properties, "fixed_timestep", E->advection.fixed_timestep, out);
    getScalarProperty(properties, "inputdiffusivity", E->control.inputdiff, out);

    getScalarProperty(properties, "adv_sub_iterations", E->advection.temp_iterations, out);
    getScalarProperty(properties, "maxadvtime", E->advection.max_dimensionless_time, out);

    getScalarProperty(properties, "aug_lagr", E->control.augmented_Lagr, out);
    getScalarProperty(properties, "aug_number", E->control.augmented, out);


    E->advection.total_timesteps = 1;
    E->advection.sub_iterations = 1;
    E->advection.last_sub_iterations = 1;
    E->advection.gamma = 0.5;
    E->advection.dt_reduced = 1.0;

    E->monitor.T_maxvaried = 1.05;

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.bc]" << std::endl;

    getScalarProperty(properties, "side_sbcs", E->control.side_sbcs, out);
    getScalarProperty(properties, "pseudo_free_surf", E->control.pseudo_free_surf, out);

    getScalarProperty(properties, "topvbc", E->mesh.topvbc, out);
    getScalarProperty(properties, "topvbxval", E->control.VBXtopval, out);
    getScalarProperty(properties, "topvbyval", E->control.VBYtopval, out);

    getScalarProperty(properties, "botvbc", E->mesh.botvbc, out);
    getScalarProperty(properties, "botvbxval", E->control.VBXbotval, out);
    getScalarProperty(properties, "botvbyval", E->control.VBYbotval, out);

    getScalarProperty(properties, "toptbc", E->mesh.toptbc, out);
    getScalarProperty(properties, "toptbcval", E->control.TBCtopval, out);

    getScalarProperty(properties, "bottbc", E->mesh.bottbc, out);
    getScalarProperty(properties, "bottbcval", E->control.TBCbotval, out);

    getScalarProperty(properties, "temperature_bound_adj", E->control.temperature_bound_adj, out);
    getScalarProperty(properties, "depth_bound_adj", E->control.depth_bound_adj, out);
    getScalarProperty(properties, "width_bound_adj", E->control.width_bound_adj, out);


    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.const]" << std::endl;

    float radius;
    getScalarProperty(properties, "layerd", radius, out);
    getScalarProperty(properties, "density", E->data.density, out);
    getScalarProperty(properties, "thermdiff", E->data.therm_diff, out);
    getScalarProperty(properties, "gravacc", E->data.grav_acc, out);
    getScalarProperty(properties, "thermexp", E->data.therm_exp, out);
    getScalarProperty(properties, "refvisc", E->data.ref_viscosity, out);
    getScalarProperty(properties, "cp", E->data.Cp, out);
    getScalarProperty(properties, "wdensity", E->data.density_above, out);
    getScalarProperty(properties, "surftemp", E->data.surf_temp, out);

    E->data.therm_cond = E->data.therm_diff * E->data.density * E->data.Cp;
    E->data.ref_temperature = E->control.Atemp * E->data.therm_diff
	* E->data.ref_viscosity / (radius * radius * radius)
	/ (E->data.density * E->data.grav_acc * E->data.therm_exp);

    getScalarProperty(properties, "z_lith", E->viscosity.zlith, out);
    getScalarProperty(properties, "z_410", E->viscosity.z410, out);
    getScalarProperty(properties, "z_lmantle", E->viscosity.zlm, out);
    getScalarProperty(properties, "z_cmb", E->viscosity.zcmb, out); //this is used as the D" phase change depth

    // convert meter to kilometer
    E->data.layer_km = radius / 1e3;
    E->data.radius_km = E->data.layer_km;

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.ic]" << std::endl;

    getScalarProperty(properties, "restart", E->control.restart, out);
    getScalarProperty(properties, "post_p", E->control.post_p, out);
    getScalarProperty(properties, "solution_cycles_init", E->monitor.solution_cycles_init, out);
    getScalarProperty(properties, "zero_elapsed_time", E->control.zero_elapsed_time, out);

    getScalarProperty(properties, "tic_method", E->convection.tic_method, out);

    if (E->convection.tic_method == 0) {
	int num_perturb;

	getScalarProperty(properties, "num_perturbations", num_perturb, out);
	if(num_perturb > PERTURB_MAX_LAYERS) {
	    std::cerr << "'num_perturb' greater than allowed value, set to "
		      << PERTURB_MAX_LAYERS << std::endl;
	    num_perturb = PERTURB_MAX_LAYERS;
	}
	E->convection.number_of_perturbations = num_perturb;

	getVectorProperty(properties, "perturbl", E->convection.perturb_ll,
			  num_perturb, out);
	getVectorProperty(properties, "perturbm", E->convection.perturb_mm,
			  num_perturb, out);
	getVectorProperty(properties, "perturblayer", E->convection.load_depth,
			  num_perturb, out);
	getVectorProperty(properties, "perturbmag", E->convection.perturb_mag,
			  num_perturb, out);
    }
    else if (E->convection.tic_method == 1) {
	getScalarProperty(properties, "half_space_age", E->convection.half_space_age, out);
    }
    else if (E->convection.tic_method == 2) {
        getScalarProperty(properties, "half_space_age", E->convection.half_space_age, out);
        getVectorProperty(properties, "blob_center", E->convection.blob_center, 3, out);
        if( E->convection.blob_center[0] == -999.0 && E->convection.blob_center[1] == -999.0 && E->convection.blob_center[2] == -999.0 ) {
            E->convection.blob_center[0] = 0.5*(E->control.theta_min+E->control.theta_max);
            E->convection.blob_center[1] = 0.5*(E->control.fi_min+E->control.fi_max);
            E->convection.blob_center[2] = 0.5*(E->sphere.ri+E->sphere.ro);
        }
        getScalarProperty(properties, "blob_radius", E->convection.blob_radius, out);
        getScalarProperty(properties, "blob_dT", E->convection.blob_dT, out);
    }

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.param]" << std::endl;

    getScalarProperty(properties, "file_vbcs", E->control.vbcs_file, out);
    getStringProperty(properties, "vel_bound_file", E->control.velocity_boundary_file, out);

    getScalarProperty(properties, "mat_control", E->control.mat_control, out);
    getStringProperty(properties, "mat_file", E->control.mat_file, out);

    getScalarProperty(properties, "lith_age", E->control.lith_age, out);
    getStringProperty(properties, "lith_age_file", E->control.lith_age_file, out);
    getScalarProperty(properties, "lith_age_time", E->control.lith_age_time, out);
    getScalarProperty(properties, "lith_age_depth", E->control.lith_age_depth, out);
    getScalarProperty(properties, "mantle_temp", E->control.lith_age_mantle_temp, out);

    getScalarProperty(properties, "start_age", E->control.start_age, out);
    getScalarProperty(properties, "reset_startage", E->control.reset_startage, out);

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.phase]" << std::endl;

    getScalarProperty(properties, "Ra_410", E->control.Ra_410, out);
    getScalarProperty(properties, "clapeyron410", E->control.clapeyron410, out);
    getScalarProperty(properties, "transT410", E->control.transT410, out);
    getScalarProperty(properties, "width410", E->control.width410, out);

    if (E->control.width410!=0.0)
	E->control.width410 = 1.0/E->control.width410;

    getScalarProperty(properties, "Ra_670", E->control.Ra_670 , out);
    getScalarProperty(properties, "clapeyron670", E->control.clapeyron670, out);
    getScalarProperty(properties, "transT670", E->control.transT670, out);
    getScalarProperty(properties, "width670", E->control.width670, out);

    if (E->control.width670!=0.0)
	E->control.width670 = 1.0/E->control.width670;

    getScalarProperty(properties, "Ra_cmb", E->control.Ra_cmb, out);
    getScalarProperty(properties, "clapeyroncmb", E->control.clapeyroncmb, out);
    getScalarProperty(properties, "transTcmb", E->control.transTcmb, out);
    getScalarProperty(properties, "widthcmb", E->control.widthcmb, out);

    if (E->control.widthcmb!=0.0)
	E->control.widthcmb = 1.0/E->control.widthcmb;

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver]" << std::endl;

    getStringProperty(properties, "datadir", E->control.data_dir, out);
    getStringProperty(properties, "datafile", E->control.data_file, out);
    getStringProperty(properties, "datafile_old", E->control.old_P_file, out);

    getScalarProperty(properties, "rayleigh", E->control.Atemp, out);
    getScalarProperty(properties, "Q0", E->control.Q0, out);

    getScalarProperty(properties, "stokes_flow_only", E->control.stokes, out);

    getStringProperty(properties, "output_format", E->output.format, out);
    getStringProperty(properties, "output_optional", E->output.optional, out);

    getScalarProperty(properties, "verbose", E->control.verbose, out);
    getScalarProperty(properties, "see_convergence", E->control.print_convergence, out);

    // parameters not used in pyre version,
    // assigned value here to prevent uninitialized access
    E->advection.min_timesteps = 1;
    E->advection.max_timesteps = 1;
    E->advection.max_total_timesteps = 1;
    E->control.record_every = 1;
    E->control.record_all_until = 1;

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.mesher]" << std::endl;

    getScalarProperty(properties, "nproc_surf", E->parallel.nprocxy, out);

    getScalarProperty(properties, "nprocx", E->parallel.nprocx, out);
    getScalarProperty(properties, "nprocy", E->parallel.nprocy, out);
    getScalarProperty(properties, "nprocz", E->parallel.nprocz, out);

    if (E->parallel.nprocxy == 12)
	if (E->parallel.nprocx != E->parallel.nprocy) {
	    char errmsg[] = "!!!! nprocx must equal to nprocy";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
    }

    getScalarProperty(properties, "coor", E->control.coor, out);
    getStringProperty(properties, "coor_file", E->control.coor_file, out);

    getScalarProperty(properties, "nodex", E->mesh.nox, out);
    getScalarProperty(properties, "nodey", E->mesh.noy, out);
    getScalarProperty(properties, "nodez", E->mesh.noz, out);
    getScalarProperty(properties, "levels", E->mesh.levels, out);

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

    getScalarProperty(properties, "radius_outer", E->sphere.ro, out);
    getScalarProperty(properties, "radius_inner", E->sphere.ri, out);

    E->mesh.nsd = 3;
    E->mesh.dof = 3;
    E->sphere.max_connections = 6;

    if (E->parallel.nprocxy == 12) {
	// full spherical version
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
	// regional version
	E->sphere.caps = 1;

	getScalarProperty(properties, "theta_min", E->control.theta_min, out);
	getScalarProperty(properties, "theta_max", E->control.theta_max, out);
	getScalarProperty(properties, "fi_min", E->control.fi_min, out);
	getScalarProperty(properties, "fi_max", E->control.fi_max, out);

	E->sphere.cap[1].theta[1] = E->control.theta_min;
	E->sphere.cap[1].theta[2] = E->control.theta_max;
	E->sphere.cap[1].theta[3] = E->control.theta_max;
	E->sphere.cap[1].theta[4] = E->control.theta_min;
	E->sphere.cap[1].fi[1] = E->control.fi_min;
	E->sphere.cap[1].fi[2] = E->control.fi_min;
	E->sphere.cap[1].fi[3] = E->control.fi_max;
	E->sphere.cap[1].fi[4] = E->control.fi_max;
    }

    getScalarProperty(properties, "ll_max", E->sphere.llmax, out);
    getScalarProperty(properties, "nlong", E->sphere.noy, out);
    getScalarProperty(properties, "nlati", E->sphere.nox, out);
    getScalarProperty(properties, "output_ll_max", E->sphere.output_llmax, out);

    E->mesh.layer[1] = 1;
    E->mesh.layer[2] = 1;
    E->mesh.layer[3] = 1;

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.tracer]" << std::endl;

    getScalarProperty(properties, "tracer", E->control.tracer, out);
    getStringProperty(properties, "tracer_file", E->control.tracer_file, out);

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.visc]" << std::endl;

    getStringProperty(properties, "Viscosity", E->viscosity.STRUCTURE, out);
    if (strcmp(E->viscosity.STRUCTURE,"system") == 0)
	E->viscosity.FROM_SYSTEM = 1;
    else
	E->viscosity.FROM_SYSTEM = 0;

    getScalarProperty(properties, "visc_smooth_method", E->viscosity.smooth_cycles, out);
    getScalarProperty(properties, "VISC_UPDATE", E->viscosity.update_allowed, out);

    int num_mat;
    const int max_mat = 40;

    getScalarProperty(properties, "num_mat", num_mat, out);
    if(num_mat > max_mat) {
	// max. allowed material types = 40
	std::cerr << "'num_mat' greater than allowed value, set to "
		  << max_mat << std::endl;
	num_mat = max_mat;
    }
    E->viscosity.num_mat = num_mat;

    getVectorProperty(properties, "visc0",
			E->viscosity.N0, num_mat, out);

    getScalarProperty(properties, "TDEPV", E->viscosity.TDEPV, out);
    getScalarProperty(properties, "rheol", E->viscosity.RHEOL, out);
    getVectorProperty(properties, "viscE",
			E->viscosity.E, num_mat, out);
    getVectorProperty(properties, "viscT",
			E->viscosity.T, num_mat, out);
    getVectorProperty(properties, "viscZ",
			E->viscosity.Z, num_mat, out);

    getScalarProperty(properties, "SDEPV", E->viscosity.SDEPV, out);
    getScalarProperty(properties, "sdepv_misfit", E->viscosity.sdepv_misfit, out);
    getVectorProperty(properties, "sdepv_expt",
			E->viscosity.sdepv_expt, num_mat, out);

    getScalarProperty(properties, "VMIN", E->viscosity.MIN, out);
    getScalarProperty(properties, "visc_min", E->viscosity.min_value, out);

    getScalarProperty(properties, "VMAX", E->viscosity.MAX, out);
    getScalarProperty(properties, "visc_max", E->viscosity.max_value, out);

    *out << std::endl;
    out->close();
    delete out;

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
    std::ofstream *out = get_output_stream(E->parallel.me, E->control.PID);
    *out << "[CitcomS.solver.vsolver]" << std::endl;

    getStringProperty(properties, "Solver", E->control.SOLVER_TYPE, out);
    getScalarProperty(properties, "node_assemble", E->control.NASSEMBLE, out);
    getScalarProperty(properties, "precond", E->control.precondition, out);

    getScalarProperty(properties, "accuracy", E->control.accuracy, out);
    getScalarProperty(properties, "tole_compressibility", E->control.tole_comp, out);

    getScalarProperty(properties, "mg_cycle", E->control.mg_cycle, out);
    getScalarProperty(properties, "down_heavy", E->control.down_heavy, out);
    getScalarProperty(properties, "up_heavy", E->control.up_heavy, out);

    getScalarProperty(properties, "vlowstep", E->control.v_steps_low, out);
    getScalarProperty(properties, "vhighstep", E->control.v_steps_high, out);
    getScalarProperty(properties, "piterations", E->control.p_iterations, out);

    *out << std::endl;
    out->close();
    delete out;

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}




//==========================================================
// helper functions

std::ofstream* get_output_stream(int mute, int pid)
{
    std::ofstream *out = new std::ofstream;

    if (mute)
	out->open("/dev/null");
    else {
	char filename[255];
	sprintf(filename, "pid%d.cfg", pid);
	out->open(filename, std::ios::app);
    }

    return out;
}


void getStringProperty(PyObject* properties, char* attribute,
		       char* value, std::ostream* out)
{
    *out << attribute << "=";

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
void getScalarProperty(PyObject* properties, char* attribute,
		       T& value, std::ostream* out)
{
    *out << attribute << "=";

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
		       T* vector, const int len, std::ostream* out)
{
    *out << attribute << "=";

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
