// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>
#include <cstdio>
#include <iostream>
#include <string>

#include "setProperties.h"

extern "C" {
#include "global_defs.h"
#include "citcom_init.h"

}


void getStringProperty(PyObject* properties, char* attribute, char* value);

template <class T>
void getScalarProperty(PyObject* properties, char* attribute, T& value);

template <class T>
void getVectorProperty(PyObject* properties, char* attribute,
		       T* vector, int len);

//
//

char pyCitcom_Advection_diffusion_set_properties__doc__[] = "";
char pyCitcom_Advection_diffusion_set_properties__name__[] = "Advection_diffusion_set_properties";

PyObject * pyCitcom_Advection_diffusion_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Advection_diffusion_set_properties", &properties))
        return NULL;

    std::cerr << "Advection_diffusion.inventories:" << std::endl;

    getScalarProperty(properties, "ADV", E->advection.ADVECTION);
    getScalarProperty(properties, "fixed_timestep", E->advection.fixed_timestep);
    getScalarProperty(properties, "finetunedt", E->advection.fine_tune_dt);

    getScalarProperty(properties, "adv_sub_iterations", E->advection.temp_iterations);
    getScalarProperty(properties, "maxadvtime", E->advection.max_dimensionless_time);

    getScalarProperty(properties, "aug_lagr", E->control.augmented_Lagr);
    getScalarProperty(properties, "aug_number", E->control.augmented);

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
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:BC_set_properties", &properties))
        return NULL;

    std::cerr << "BC.inventories:" << std::endl;

    getScalarProperty(properties, "topvbc", E->mesh.topvbc);
    getScalarProperty(properties, "topvbxval", E->control.VBXtopval);
    getScalarProperty(properties, "topvbyval", E->control.VBYtopval);

    getScalarProperty(properties, "botvbc", E->mesh.botvbc);
    getScalarProperty(properties, "botvbxval", E->control.VBXbotval);
    getScalarProperty(properties, "botvbyval", E->control.VBYbotval);

    getScalarProperty(properties, "toptbc", E->mesh.toptbc);
    getScalarProperty(properties, "toptbcval", E->control.TBCtopval);

    getScalarProperty(properties, "bottbc", E->mesh.bottbc);
    getScalarProperty(properties, "bottbcval", E->control.TBCbotval);

    getScalarProperty(properties, "temperature_bound_adj", E->control.temperature_bound_adj);
    getScalarProperty(properties, "depth_bound_adj", E->control.depth_bound_adj);
    getScalarProperty(properties, "width_bound_adj", E->control.width_bound_adj);


    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Const_set_properties__doc__[] = "";
char pyCitcom_Const_set_properties__name__[] = "Const_set_properties";

PyObject * pyCitcom_Const_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;
    float zlith, z410, zlm, zcmb;

    if (!PyArg_ParseTuple(args, "O:Const_set_properties", &properties))
        return NULL;

    std::cerr << "Const.inventories:" << std::endl;

    getScalarProperty(properties, "layerd", E->data.layer_km);
    getScalarProperty(properties, "density", E->data.density);
    getScalarProperty(properties, "thermdiff", E->data.therm_diff);
    getScalarProperty(properties, "gravacc", E->data.grav_acc);
    getScalarProperty(properties, "thermexp", E->data.therm_exp);
    getScalarProperty(properties, "refvisc", E->data.ref_viscosity);
    getScalarProperty(properties, "cp", E->data.Cp);
    getScalarProperty(properties, "wdensity", E->data.density_above);

    E->data.therm_cond = E->data.therm_diff * E->data.density * E->data.Cp;

    getScalarProperty(properties, "depth_lith", zlith);
    getScalarProperty(properties, "depth_410", z410);
    getScalarProperty(properties, "depth_660", zlm);
    getScalarProperty(properties, "depth_cmb", zcmb); //this is used as the D" phase change depth
    //getScalarProperty(properties, "depth_d_double_prime", E->data.zd_double_prime);

    E->viscosity.zlith = zlith / E->data.radius_km;
    E->viscosity.z410 = z410 / E->data.radius_km;
    E->viscosity.zlm = zlm / E->data.radius_km;
    E->viscosity.zcmb = zcmb / E->data.radius_km;

    // convert meter to kilometer
    E->data.layer_km = E->data.layer_km / 1e3;
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
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:IC_set_properties", &properties))
        return NULL;

    std::cerr << "IC.inventories:" << std::endl;

    int num_perturb;
    const int max_perturb = 32;

    getScalarProperty(properties, "num_perturbations", num_perturb);
    if(num_perturb > max_perturb) {
	// max. allowed perturberial types = 40
	std::cerr << "'num_perturb' greater than allowed value, set to "
		  << max_perturb << std::endl;
	num_perturb = max_perturb;
    }
    E->number_of_perturbations = num_perturb;

    getVectorProperty(properties, "perturbl", E->perturb_ll, num_perturb);
    getVectorProperty(properties, "perturbm", E->perturb_mm, num_perturb);
    getVectorProperty(properties, "perturblayer", E->load_depth, num_perturb);
    getVectorProperty(properties, "perturbmag", E->perturb_mag, num_perturb);

    if (PyErr_Occurred())
      return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Parallel_set_properties__doc__[] = "";
char pyCitcom_Parallel_set_properties__name__[] = "Parallel_set_properties";

PyObject * pyCitcom_Parallel_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Parallel_set_properties", &properties))
        return NULL;

    std::cerr << "Parallel.inventories:" << std::endl;

    getScalarProperty(properties, "nproc_surf", E->parallel.nprocxy);
    getScalarProperty(properties, "nprocx", E->parallel.nprocxl);
    getScalarProperty(properties, "nprocy", E->parallel.nprocyl);
    getScalarProperty(properties, "nprocz", E->parallel.nproczl);

    if (E->parallel.nprocxy == 12)
	if (E->parallel.nprocxl != E->parallel.nprocyl) {
	    char errmsg[] = "!!!! nprocx must equal to nprocy";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
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
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Param_set_properties", &properties))
        return NULL;

    std::cerr << "Param.inventories:" << std::endl;

    getStringProperty(properties, "datafile", E->control.data_file);

    getScalarProperty(properties, "file_vbcs", E->control.vbcs_file);
    getStringProperty(properties, "vel_bound_file", E->control.velocity_boundary_file);

    getScalarProperty(properties, "mat_control", E->control.mat_control);
    getStringProperty(properties, "mat_file", E->control.mat_file);

    getScalarProperty(properties, "lith_age", E->control.lith_age);
    getStringProperty(properties, "lith_age_file", E->control.lith_age_file);
    getScalarProperty(properties, "lith_age_time", E->control.lith_age_time);
    getScalarProperty(properties, "lith_age_depth", E->control.lith_age_depth);
    getScalarProperty(properties, "mantle_temp", E->control.lith_age_mantle_temp);

    getScalarProperty(properties, "tracer", E->control.tracer);
    getStringProperty(properties, "tracer_file", E->control.tracer_file);

    getScalarProperty(properties, "restart", E->control.restart);
    getScalarProperty(properties, "post_p", E->control.post_p);
    getStringProperty(properties, "datafile_old", E->control.old_P_file);
    getScalarProperty(properties, "solution_cycles_init", E->monitor.solution_cycles_init);
    getScalarProperty(properties, "zero_elapsed_time", E->control.zero_elapsed_time);

    getScalarProperty(properties, "minstep", E->advection.min_timesteps);
    getScalarProperty(properties, "maxstep", E->advection.max_timesteps);
    getScalarProperty(properties, "maxtotstep", E->advection.max_total_timesteps);
    getScalarProperty(properties, "storage_spacing", E->control.record_every);
    getScalarProperty(properties, "cpu_limits_in_seconds", E->control.record_all_until);

    getScalarProperty(properties, "stokes_flow_only", E->control.stokes);

    getScalarProperty(properties, "inputdiffusivity", E->control.inputdiff);

    getScalarProperty(properties, "rayleigh", E->control.Atemp);

    getScalarProperty(properties, "Q0", E->control.Q0);

    getScalarProperty(properties, "verbose", E->control.verbose);
    getScalarProperty(properties, "see_convergence", E->control.print_convergence);

    getScalarProperty(properties, "start_age", E->control.start_age);
    getScalarProperty(properties, "reset_startage", E->control.reset_startage);


    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_Phase_set_properties__doc__[] = "";
char pyCitcom_Phase_set_properties__name__[] = "Phase_set_properties";

PyObject * pyCitcom_Phase_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Phase_set_properties", &properties))
        return NULL;

    std::cerr << "Phase.inventories:" << std::endl;

    getScalarProperty(properties, "Ra_410", E->control.Ra_410 );
    getScalarProperty(properties, "clapeyron410", E->control.clapeyron410);
    getScalarProperty(properties, "transT410", E->control.transT410);
    getScalarProperty(properties, "width410", E->control.width410);

    if (E->control.width410!=0.0)
	E->control.width410 = 1.0/E->control.width410;

    getScalarProperty(properties, "Ra_670", E->control.Ra_670 );
    getScalarProperty(properties, "clapeyron670", E->control.clapeyron670);
    getScalarProperty(properties, "transT670", E->control.transT670);
    getScalarProperty(properties, "width670", E->control.width670);

    if (E->control.width670!=0.0)
	E->control.width670 = 1.0/E->control.width670;

    getScalarProperty(properties, "Ra_cmb", E->control.Ra_cmb);
    getScalarProperty(properties, "clapeyroncmb", E->control.clapeyroncmb);
    getScalarProperty(properties, "transTcmb", E->control.transTcmb);
    getScalarProperty(properties, "widthcmb", E->control.widthcmb);

    if (E->control.widthcmb!=0.0)
	E->control.widthcmb = 1.0/E->control.widthcmb;

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_Sphere_set_properties__doc__[] = "";
char pyCitcom_Sphere_set_properties__name__[] = "Sphere_set_properties";

PyObject * pyCitcom_Sphere_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Sphere_set_properties", &properties))
        return NULL;

    std::cerr << "Sphere.inventories:" << std::endl;

    getScalarProperty(properties, "nproc_surf", E->parallel.nprocxy);
    getScalarProperty(properties, "nprocx", E->parallel.nprocxl);
    getScalarProperty(properties, "nprocy", E->parallel.nprocyl);
    getScalarProperty(properties, "nprocz", E->parallel.nproczl);

    if (E->parallel.nprocxy == 12)
	if (E->parallel.nprocxl != E->parallel.nprocyl) {
	    char errmsg[] = "!!!! nprocx must equal to nprocy";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
    }

    getScalarProperty(properties, "coor", E->control.coor);
    getStringProperty(properties, "coor_file", E->control.coor_file);

    getScalarProperty(properties, "nodex", E->mesh.nox);
    getScalarProperty(properties, "nodey", E->mesh.noy);
    getScalarProperty(properties, "nodez", E->mesh.noz);
    getScalarProperty(properties, "mgunitx", E->mesh.mgunitx);
    getScalarProperty(properties, "mgunity", E->mesh.mgunity);
    getScalarProperty(properties, "mgunitz", E->mesh.mgunitz);
    getScalarProperty(properties, "levels", E->mesh.levels);

    if (E->parallel.nprocxy == 12) {
	if (E->mesh.nox != E->mesh.noy) {
	    char errmsg[] = "!!!! nodex must equal to nodey";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
	}
	if (E->mesh.mgunitx != E->mesh.mgunity) {
	    char errmsg[] = "!!!! mgunitx must equal to mgunity";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
	}
    }

    getScalarProperty(properties, "radius_outer", E->sphere.ro);
    getScalarProperty(properties, "radius_inner", E->sphere.ri);

    getScalarProperty(properties, "theta_min", E->control.theta_min);
    getScalarProperty(properties, "theta_max", E->control.theta_max);
    getScalarProperty(properties, "fi_min", E->control.fi_min);
    getScalarProperty(properties, "fi_max", E->control.fi_max);

    E->sphere.cap[1].theta[1] = E->control.theta_min;
    E->sphere.cap[1].theta[2] = E->control.theta_max;
    E->sphere.cap[1].theta[3] = E->control.theta_max;
    E->sphere.cap[1].theta[4] = E->control.theta_min;
    E->sphere.cap[1].fi[1] = E->control.fi_min;
    E->sphere.cap[1].fi[2] = E->control.fi_min;
    E->sphere.cap[1].fi[3] = E->control.fi_max;
    E->sphere.cap[1].fi[4] = E->control.fi_max;

    E->mesh.nsd = 3;
    E->mesh.dof = 3;
    E->sphere.max_connections = 6;
    if (E->parallel.nprocxy == 12)
	E->sphere.caps = 12;
    else
	E->sphere.caps = 1;

    getScalarProperty(properties, "dimenx", E->mesh.layer[1]);
    getScalarProperty(properties, "dimeny", E->mesh.layer[2]);
    getScalarProperty(properties, "dimenz", E->mesh.layer[3]);

    getScalarProperty(properties, "ll_max", E->sphere.llmax);
    getScalarProperty(properties, "nlong", E->sphere.noy);
    getScalarProperty(properties, "nlati", E->sphere.nox);
    getScalarProperty(properties, "output_ll_max", E->sphere.output_llmax);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Visc_set_properties__doc__[] = "";
char pyCitcom_Visc_set_properties__name__[] = "Visc_set_properties";

PyObject * pyCitcom_Visc_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Visc_set_properties", &properties))
        return NULL;

    std::cerr << "Visc.inventories:" << std::endl;

    getStringProperty(properties, "Viscosity", E->viscosity.STRUCTURE);
    if ( strcmp(E->viscosity.STRUCTURE,"system") == 0)
	E->viscosity.FROM_SYSTEM = 1;
    else
	E->viscosity.FROM_SYSTEM = 0;

    getScalarProperty(properties, "rheol", E->viscosity.RHEOL);


    getScalarProperty(properties, "visc_smooth_method", E->viscosity.smooth_cycles);
    getScalarProperty(properties, "VISC_UPDATE", E->viscosity.update_allowed);

    int num_mat;
    const int max_mat = 40;

    getScalarProperty(properties, "num_mat", num_mat);
    if(num_mat > max_mat) {
	// max. allowed material types = 40
	std::cerr << "'num_mat' greater than allowed value, set to "
		  << max_mat << std::endl;
	num_mat = max_mat;
    }
    E->viscosity.num_mat = num_mat;

    getVectorProperty(properties, "visc0",
			E->viscosity.N0, num_mat);

    getScalarProperty(properties, "TDEPV", E->viscosity.TDEPV);
    getVectorProperty(properties, "viscE",
			E->viscosity.E, num_mat);
    getVectorProperty(properties, "viscT",
			E->viscosity.T, num_mat);

    getScalarProperty(properties, "SDEPV", E->viscosity.SDEPV);
    getScalarProperty(properties, "sdepv_misfit", E->viscosity.sdepv_misfit);
    getVectorProperty(properties, "sdepv_expt",
			E->viscosity.sdepv_expt, num_mat);

    getScalarProperty(properties, "VMIN", E->viscosity.MIN);
    getScalarProperty(properties, "visc_min", E->viscosity.min_value);

    getScalarProperty(properties, "VMAX", E->viscosity.MAX);
    getScalarProperty(properties, "visc_max", E->viscosity.max_value);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_Stokes_solver_set_properties__doc__[] = "";
char pyCitcom_Stokes_solver_set_properties__name__[] = "Stokes_solver_set_properties";

PyObject * pyCitcom_Stokes_solver_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Stokes_solver_set_properties", &properties))
        return NULL;

    std::cerr << "Stokes_solver.inventories:" << std::endl;

    getStringProperty(properties, "Solver", E->control.SOLVER_TYPE);
    getScalarProperty(properties, "node_assemble", E->control.NASSEMBLE);
    getScalarProperty(properties, "precond", E->control.precondition);

    getScalarProperty(properties, "mg_cycle", E->control.mg_cycle);
    getScalarProperty(properties, "down_heavy", E->control.down_heavy);
    getScalarProperty(properties, "up_heavy", E->control.up_heavy);

    getScalarProperty(properties, "vlowstep", E->control.v_steps_low);
    getScalarProperty(properties, "vhighstep", E->control.v_steps_high);
    getScalarProperty(properties, "piterations", E->control.p_iterations);

    getScalarProperty(properties, "accuracy", E->control.accuracy);
    getScalarProperty(properties, "tole_compressibility", E->control.tole_comp);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}




//==========================================================
// helper functions


void getStringProperty(PyObject* properties, char* attribute, char* value)
{
    std::cerr << '\t' << attribute << " = ";

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
    std::cerr << '"' << value << '"' << std::endl;

    return;
}



template <class T>
void getScalarProperty(PyObject* properties, char* attribute, T& value)
{
    std::cerr << '\t' << attribute << " = ";

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
    std::cerr << value << std::endl;

    return;
}



template <class T>
void getVectorProperty(PyObject* properties, char* attribute,
		       T* vector, const int len)
{
    std::cerr << '\t' << attribute << " = ";

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
	std::cerr << warnmsg << std::endl;
    }

    std::cerr << "[ ";
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
	std::cerr << vector[i] << ", ";
    }
    std::cerr << ']' << std::endl;

    return;
}


// version
// $Id: setProperties.cc,v 1.10 2003/08/01 22:53:50 tan2 Exp $

// End of file
