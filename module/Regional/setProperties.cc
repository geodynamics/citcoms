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
#include <fstream>
#include <string>

#include "setProperties.h"

extern "C" {
#include "global_defs.h"
#include "citcom_init.h"

}


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
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Advection_diffusion_set_properties", &properties))
        return NULL;

    int m = E->parallel.me;
    if (not m)
	std::cerr << "Advection_diffusion.inventories:" << std::endl;

    getScalarProperty(properties, "ADV", E->advection.ADVECTION, m);
    getScalarProperty(properties, "fixed_timestep", E->advection.fixed_timestep, m);
    getScalarProperty(properties, "finetunedt", E->advection.fine_tune_dt, m);

    getScalarProperty(properties, "adv_sub_iterations", E->advection.temp_iterations, m);
    getScalarProperty(properties, "maxadvtime", E->advection.max_dimensionless_time, m);

    getScalarProperty(properties, "aug_lagr", E->control.augmented_Lagr, m);
    getScalarProperty(properties, "aug_number", E->control.augmented, m);

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

    int m = E->parallel.me;
    if (not m)
	std::cerr << "BC.inventories:" << std::endl;

    getScalarProperty(properties, "topvbc", E->mesh.topvbc, m);
    getScalarProperty(properties, "topvbxval", E->control.VBXtopval, m);
    getScalarProperty(properties, "topvbyval", E->control.VBYtopval, m);

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
    PyObject *properties;
    float zlith, z410, zlm, zcmb;

    if (!PyArg_ParseTuple(args, "O:Const_set_properties", &properties))
        return NULL;

    int m = E->parallel.me;
    if (not m)
	std::cerr << "Const.inventories:" << std::endl;

    getScalarProperty(properties, "layerd", E->data.layer_km, m);
    getScalarProperty(properties, "density", E->data.density, m);
    getScalarProperty(properties, "thermdiff", E->data.therm_diff, m);
    getScalarProperty(properties, "gravacc", E->data.grav_acc, m);
    getScalarProperty(properties, "thermexp", E->data.therm_exp, m);
    getScalarProperty(properties, "refvisc", E->data.ref_viscosity, m);
    getScalarProperty(properties, "cp", E->data.Cp, m);
    getScalarProperty(properties, "wdensity", E->data.density_above, m);

    E->data.therm_cond = E->data.therm_diff * E->data.density * E->data.Cp;

    getScalarProperty(properties, "depth_lith", zlith, m);
    getScalarProperty(properties, "depth_410", z410, m);
    getScalarProperty(properties, "depth_660", zlm, m);
    getScalarProperty(properties, "depth_cmb", zcmb, m); //this is used as the D" phase change depth
    //getScalarProperty(properties, "depth_d_double_prime", E->data.zd_double_prime, m);

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

    int m = E->parallel.me;
    if (not m)
	std::cerr << "IC.inventories:" << std::endl;

    int num_perturb;
    const int max_perturb = 32;

    getScalarProperty(properties, "num_perturbations", num_perturb, m);
    if(num_perturb > max_perturb) {
	// max. allowed perturbations = 32
	std::cerr << "'num_perturb' greater than allowed value, set to "
		  << max_perturb << std::endl;
	num_perturb = max_perturb;
    }
    E->convection.number_of_perturbations = num_perturb;

    getVectorProperty(properties, "perturbl", E->convection.perturb_ll, num_perturb, m);
    getVectorProperty(properties, "perturbm", E->convection.perturb_mm, num_perturb, m);
    getVectorProperty(properties, "perturblayer", E->convection.load_depth, num_perturb, m);
    getVectorProperty(properties, "perturbmag", E->convection.perturb_mag, num_perturb, m);

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

    int m = E->parallel.me;
    if (not m)
	std::cerr << "Parallel.inventories:" << std::endl;

    getScalarProperty(properties, "nproc_surf", E->parallel.nprocxy, m);
    getScalarProperty(properties, "nprocx", E->parallel.nprocxl, m);
    getScalarProperty(properties, "nprocy", E->parallel.nprocyl, m);
    getScalarProperty(properties, "nprocz", E->parallel.nproczl, m);

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

    int m = E->parallel.me;
    if (not m)
	std::cerr << "Param.inventories:" << std::endl;

    getStringProperty(properties, "datafile", E->control.data_file, m);

    getScalarProperty(properties, "file_vbcs", E->control.vbcs_file, m);
    getStringProperty(properties, "vel_bound_file", E->control.velocity_boundary_file, m);

    getScalarProperty(properties, "mat_control", E->control.mat_control, m);
    getStringProperty(properties, "mat_file", E->control.mat_file, m);

    getScalarProperty(properties, "lith_age", E->control.lith_age, m);
    getStringProperty(properties, "lith_age_file", E->control.lith_age_file, m);
    getScalarProperty(properties, "lith_age_time", E->control.lith_age_time, m);
    getScalarProperty(properties, "lith_age_depth", E->control.lith_age_depth, m);
    getScalarProperty(properties, "mantle_temp", E->control.lith_age_mantle_temp, m);

    getScalarProperty(properties, "tracer", E->control.tracer, m);
    getStringProperty(properties, "tracer_file", E->control.tracer_file, m);

    getScalarProperty(properties, "restart", E->control.restart, m);
    getScalarProperty(properties, "post_p", E->control.post_p, m);
    getStringProperty(properties, "datafile_old", E->control.old_P_file, m);
    getScalarProperty(properties, "solution_cycles_init", E->monitor.solution_cycles_init, m);
    getScalarProperty(properties, "zero_elapsed_time", E->control.zero_elapsed_time, m);

    getScalarProperty(properties, "minstep", E->advection.min_timesteps, m);
    getScalarProperty(properties, "maxstep", E->advection.max_timesteps, m);
    getScalarProperty(properties, "maxtotstep", E->advection.max_total_timesteps, m);
    getScalarProperty(properties, "storage_spacing", E->control.record_every, m);
    getScalarProperty(properties, "cpu_limits_in_seconds", E->control.record_all_until, m);

    getScalarProperty(properties, "stokes_flow_only", E->control.stokes, m);

    getScalarProperty(properties, "inputdiffusivity", E->control.inputdiff, m);

    getScalarProperty(properties, "rayleigh", E->control.Atemp, m);

    getScalarProperty(properties, "Q0", E->control.Q0, m);

    getScalarProperty(properties, "verbose", E->control.verbose, m);
    getScalarProperty(properties, "see_convergence", E->control.print_convergence, m);

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
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Phase_set_properties", &properties))
        return NULL;

    int m = E->parallel.me;
    if (not m)
	std::cerr << "Phase.inventories:" << std::endl;

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



char pyCitcom_Sphere_set_properties__doc__[] = "";
char pyCitcom_Sphere_set_properties__name__[] = "Sphere_set_properties";

PyObject * pyCitcom_Sphere_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Sphere_set_properties", &properties))
        return NULL;

    int m = E->parallel.me;
    if (not m)
	std::cerr << "Sphere.inventories:" << std::endl;

    getScalarProperty(properties, "nproc_surf", E->parallel.nprocxy, m);
    getScalarProperty(properties, "nprocx", E->parallel.nprocx, m);
    getScalarProperty(properties, "nprocy", E->parallel.nprocy, m);
    getScalarProperty(properties, "nprocz", E->parallel.nprocz, m);

    E->parallel.nprocxl = E->parallel.nprocx;
    E->parallel.nprocyl = E->parallel.nprocy;
    E->parallel.nproczl = E->parallel.nprocz;

    if (E->parallel.nprocxy == 12)
	if (E->parallel.nprocxl != E->parallel.nprocyl) {
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

    getScalarProperty(properties, "dimenx", E->mesh.layer[1], m);
    getScalarProperty(properties, "dimeny", E->mesh.layer[2], m);
    getScalarProperty(properties, "dimenz", E->mesh.layer[3], m);

    getScalarProperty(properties, "ll_max", E->sphere.llmax, m);
    getScalarProperty(properties, "nlong", E->sphere.noy, m);
    getScalarProperty(properties, "nlati", E->sphere.nox, m);
    getScalarProperty(properties, "output_ll_max", E->sphere.output_llmax, m);

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

    int m = E->parallel.me;
    if (not m)
	std::cerr << "Visc.inventories:" << std::endl;

    getStringProperty(properties, "Viscosity", E->viscosity.STRUCTURE, m);
    if ( strcmp(E->viscosity.STRUCTURE,"system") == 0)
	E->viscosity.FROM_SYSTEM = 1;
    else
	E->viscosity.FROM_SYSTEM = 0;

    getScalarProperty(properties, "rheol", E->viscosity.RHEOL, m);


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
    getVectorProperty(properties, "viscE",
			E->viscosity.E, num_mat, m);
    getVectorProperty(properties, "viscT",
			E->viscosity.T, num_mat, m);

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


char pyCitcom_Stokes_solver_set_properties__doc__[] = "";
char pyCitcom_Stokes_solver_set_properties__name__[] = "Stokes_solver_set_properties";

PyObject * pyCitcom_Stokes_solver_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Stokes_solver_set_properties", &properties))
        return NULL;

    int m = E->parallel.me;
    if (not m)
	std::cerr << "Stokes_solver.inventories:" << std::endl;

    getStringProperty(properties, "Solver", E->control.SOLVER_TYPE, m);
    getScalarProperty(properties, "node_assemble", E->control.NASSEMBLE, m);
    getScalarProperty(properties, "precond", E->control.precondition, m);

    getScalarProperty(properties, "mg_cycle", E->control.mg_cycle, m);
    getScalarProperty(properties, "down_heavy", E->control.down_heavy, m);
    getScalarProperty(properties, "up_heavy", E->control.up_heavy, m);

    getScalarProperty(properties, "vlowstep", E->control.v_steps_low, m);
    getScalarProperty(properties, "vhighstep", E->control.v_steps_high, m);
    getScalarProperty(properties, "piterations", E->control.p_iterations, m);

    getScalarProperty(properties, "accuracy", E->control.accuracy, m);
    getScalarProperty(properties, "tole_compressibility", E->control.tole_comp, m);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}




//==========================================================
// helper functions


void getStringProperty(PyObject* properties, char* attribute, char* value, int mute)
{
    std::ofstream out;

    if (mute)
	out.open("/dev/null");
    else
	out.open("/dev/stderr");

    out << '\t' << attribute << " = ";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	out.close();
	return;
    }

    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PyString_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a string", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	out.close();
	return;
    }

    strcpy(value, PyString_AsString(prop));
    out << '"' << value << '"' << std::endl;

    out.close();

    return;
}



template <class T>
void getScalarProperty(PyObject* properties, char* attribute, T& value, int mute)
{
    std::ofstream out;

    if (mute)
	out.open("/dev/null");
    else
	out.open("/dev/stderr");

    out << '\t' << attribute << " = ";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	out.close();
	return;
    }

    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PyNumber_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a number", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	out.close();
	return;
    }

    value = static_cast<T>(PyFloat_AsDouble(prop));
    out << value << std::endl;

    out.close();

    return;
}



template <class T>
void getVectorProperty(PyObject* properties, char* attribute,
		       T* vector, const int len, int mute)
{
    std::ofstream out;

    if (mute)
	out.open("/dev/null");
    else
	out.open("/dev/stderr");

    out << '\t' << attribute << " = ";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	out.close();
	return;
    }

    // is it a sequence?
    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PySequence_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a sequence", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	out.close();
	return;
    }

    // is it of length len?
    int n = PySequence_Size(prop);
    if(n < len) {
	char errmsg[255];
	sprintf(errmsg, "length of '%s' < %d", attribute, len);
	PyErr_SetString(PyExc_IndexError, errmsg);
	out.close();
	return;
    } else if (n > len) {
	char warnmsg[255];
	sprintf(warnmsg, "WARNING: length of '%s' > %d", attribute, len);
	out << warnmsg << std::endl;
    }

    out << "[ ";
    for (int i=0; i<len; i++) {
	PyObject* item = PySequence_GetItem(prop, i);
	if(!item) {
	    char errmsg[255];
	    sprintf(errmsg, "can't get %s[%d]", attribute, i);
	    PyErr_SetString(PyExc_IndexError, errmsg);
	    out.close();
	    return;
	}

	if(PyNumber_Check(item)) {
	    vector[i] = static_cast<T>(PyFloat_AsDouble(item));
	} else {
	    char errmsg[255];
	    sprintf(errmsg, "'%s[%d]' is not a number ", attribute, i);
	    PyErr_SetString(PyExc_TypeError, errmsg);
	    out.close();
	    return;
	}
	out << vector[i] << ", ";
    }
    out << ']' << std::endl;

    out.close();

    return;
}


// version
// $Id: setProperties.cc,v 1.13 2003/08/07 21:34:26 tan2 Exp $

// End of file
