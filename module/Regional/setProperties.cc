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

char pyRegional_BC_set_properties__doc__[] = "";
char pyRegional_BC_set_properties__name__[] = "BC_set_properties";

PyObject * pyRegional_BC_set_properties(PyObject *self, PyObject *args)
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



char pyRegional_Visc_set_properties__doc__[] = "";
char pyRegional_Visc_set_properties__name__[] = "Visc_set_properties";

PyObject * pyRegional_Visc_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Visc_set_properties", &properties))
        return NULL;

    std::cerr << "Visc.inventories:" << std::endl;

    getStringProperty(properties, "Viscosity", E->viscosity.STRUCTURE);

    getScalarProperty(properties, "rheol", E->viscosity.RHEOL);


    getScalarProperty(properties, "visc_smooth_method", E->viscosity.smooth_cycles);
    getScalarProperty(properties, "VISC_UPDATE", E->viscosity.update_allowed);

    int num_mat;
    getScalarProperty(properties, "num_mat", num_mat);
    if(num_mat > 40) {
	// max. allowed material types = 40
	std::cerr << "'num_mat' greater than allowed value, set to 40.";
	num_mat = 40;
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


char pyRegional_Const_set_properties__doc__[] = "";
char pyRegional_Const_set_properties__name__[] = "Const_set_properties";

PyObject * pyRegional_Const_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;
    float zlith, z410, zlm, zcmb;

    if (!PyArg_ParseTuple(args, "O:Const_set_properties", &properties))
        return NULL;

    std::cerr << "Const.inventories:" << std::endl;

    getScalarProperty(properties, "radius", E->data.radius_km);
    getScalarProperty(properties, "ref_density", E->data.density);
    getScalarProperty(properties, "thermdiff", E->data.therm_diff);
    getScalarProperty(properties, "gravacc", E->data.grav_acc);
    getScalarProperty(properties, "thermexp", E->data.therm_exp);
    getScalarProperty(properties, "ref_visc", E->data.ref_viscosity);
    getScalarProperty(properties, "heatcapacity", E->data.Cp);
    getScalarProperty(properties, "water_density", E->data.density_above);
    getScalarProperty(properties, "depth_lith", zlith);
    getScalarProperty(properties, "depth_410", z410);
    getScalarProperty(properties, "depth_660", zlm);
    getScalarProperty(properties, "depth_cmb", zcmb);
    
    E->viscosity.zlith=zlith/E->data.radius_km;
    E->viscosity.z410=z410/E->data.radius_km;
    E->viscosity.zlm=zlm/E->data.radius_km;
    E->viscosity.zcmb=zcmb/E->data.radius_km;
    // getScalarProperty(properties, "depth_d_double_prime", E->data.zd_double_prime);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}

char pyRegional_Parallel_set_properties__doc__[] = "";
char pyRegional_Parallel_set_properties__name__[] = "Parallel_set_properties";

PyObject * pyRegional_Parallel_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Parallel_set_properties", &properties))
        return NULL;

    std::cerr << "Parallel.inventories:" << std::endl;

    getScalarProperty(properties, "nproc_surf", E->parallel.nprocxy);
    getScalarProperty(properties, "nprocx", E->parallel.nprocxl);
    getScalarProperty(properties, "nprocy", E->parallel.nprocyl);
    getScalarProperty(properties, "nprocz", E->parallel.nproczl);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}

char pyRegional_Param_set_properties__doc__[] = "";
char pyRegional_Param_set_properties__name__[] = "Param_set_properties";

PyObject * pyRegional_Param_set_properties(PyObject *self, PyObject *args)
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
    
    getScalarProperty(properties, "DESCRIBE", E->control.DESCRIBE);
    getScalarProperty(properties, "BEGINNER", E->control.BEGINNER);
    getScalarProperty(properties, "VERBOSE", E->control.VERBOSE);
    getScalarProperty(properties, "verbose", E->control.verbose);
    getScalarProperty(properties, "see_convergence", E->control.print_convergence);
    
    getScalarProperty(properties, "start_age", E->control.start_age);
    getScalarProperty(properties, "reset_startage", E->control.reset_startage);
    
      
    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}

char pyRegional_Phase_set_properties__doc__[] = "";
char pyRegional_Phase_set_properties__name__[] = "Phase_set_properties";

PyObject * pyRegional_Phase_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Phase_set_properties", &properties))
        return NULL;

    std::cerr << "Phase.inventories:" << std::endl;

    getScalarProperty(properties, "Ra_410", E->control.Ra_410 );
    getScalarProperty(properties, "clapeyron410", E->control.clapeyron410);
    getScalarProperty(properties, "transT410", E->control.transT410);
    getScalarProperty(properties, "width410", E->control.width410);
    
    getScalarProperty(properties, "Ra_670", E->control.Ra_670 );
    getScalarProperty(properties, "clapeyron670", E->control.clapeyron670);
    getScalarProperty(properties, "transT670", E->control.transT670);
    getScalarProperty(properties, "width670", E->control.width670);
    
    getScalarProperty(properties, "Ra_cmb", E->control.Ra_cmb);
    getScalarProperty(properties, "clapeyroncmb", E->control.clapeyroncmb);
    getScalarProperty(properties, "transTcmb", E->control.transTcmb);
    getScalarProperty(properties, "widthcmb", E->control.widthcmb);
    

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}


char pyRegional_IC_set_properties__doc__[] = "";
char pyRegional_IC_set_properties__name__[] = "IC_set_properties";

PyObject * pyRegional_IC_set_properties(PyObject *self, PyObject *args)
{
    PyObject *properties;
    int num_perturb;

    if (!PyArg_ParseTuple(args, "O:IC_set_properties", &properties))
        return NULL;

    std::cerr << "IC.inventories:" << std::endl;

    getScalarProperty(properties, "num_perturbations", E->number_of_perturbations);

    num_perturb=E->number_of_perturbations;

    getVectorProperty(properties, "perturb_ll", E->perturb_ll,num_perturb);
    getVectorProperty(properties, "perturb_mm", E->perturb_mm,num_perturb);
    getVectorProperty(properties, "load_depth", E->load_depth,num_perturb);
    getVectorProperty(properties, "perturb_mag", E->perturb_mag,num_perturb);
    
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
    std::cerr << value << std::endl;

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
    } else if(n > len) {
	char warnmsg[255];
	sprintf(warnmsg, "length of '%s' > %d", attribute, len);
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

