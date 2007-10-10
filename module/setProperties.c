/*
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
*/

#include <Python.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "global_defs.h"
#include "parallel_related.h"
#include "setProperties.h"


/* See PEP 353. */
#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#endif


/*==============================================================*/
/* functions and macros which fetch properties from 'inventory' */


FILE *get_output_stream(PyObject *out, struct All_variables*E);
#define PUTS(s) if (fp) fprintf(fp, s)

int _getStringProperty(PyObject* properties, char* attribute,
                       char* value, size_t valueSize, FILE* fp);
#define getStringProperty(p, a, v, o) if (-1 == _getStringProperty(p, a, v, sizeof(v), o)) return NULL

int _getIntProperty(PyObject* properties, char* attribute, int *value, FILE* fp);
#define getIntProperty(p, a, v, o) if (-1 == _getIntProperty(p, a, &(v), o)) return NULL

int _getFloatProperty(PyObject* properties, char* attribute, float *value, FILE* fp);
#define getFloatProperty(p, a, v, o) if (-1 == _getFloatProperty(p, a, &(v), o)) return NULL

int _getDoubleProperty(PyObject* properties, char* attribute, double *value, FILE* fp);
#define getDoubleProperty(p, a, v, o) if (-1 == _getDoubleProperty(p, a, &(v), o)) return NULL

int _getIntVectorProperty(PyObject* properties, char* attribute,
                          int* vector, int len, FILE* fp);
#define getIntVectorProperty(p, a, v, l, o) if (-1 == _getIntVectorProperty(p, a, v, l, o)) return NULL

int _getFloatVectorProperty(PyObject* properties, char* attribute,
                            float* vector, int len, FILE* fp);
#define getFloatVectorProperty(p, a, v, l, o) if (-1 == _getFloatVectorProperty(p, a, v, l, o)) return NULL

int _getDoubleVectorProperty(PyObject* properties, char* attribute,
                             double* vector, int len, FILE* fp);
#define getDoubleVectorProperty(p, a, v, l, o) if (-1 == _getDoubleVectorProperty(p, a, v, l, o)) return NULL


void myerror(struct All_variables *,char *);
void report(struct All_variables *,char *);

/*==============================================================*/


char pyCitcom_Advection_diffusion_set_properties__doc__[] = "";
char pyCitcom_Advection_diffusion_set_properties__name__[] = "Advection_diffusion_set_properties";

PyObject * pyCitcom_Advection_diffusion_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "OOO:Advection_diffusion_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.tsolver]\n"));

    getIntProperty(properties, "ADV", E->advection.ADVECTION, fp);
    getIntProperty(properties, "filter_temp", E->advection.filter_temperature, fp);
    getIntProperty(properties, "monitor_max_T", E->advection.monitor_max_T, fp);

    getFloatProperty(properties, "finetunedt", E->advection.fine_tune_dt, fp);
    getFloatProperty(properties, "fixed_timestep", E->advection.fixed_timestep, fp);
    getFloatProperty(properties, "adv_gamma", E->advection.gamma, fp);
    getIntProperty(properties, "adv_sub_iterations", E->advection.temp_iterations, fp);

    getFloatProperty(properties, "inputdiffusivity", E->control.inputdiff, fp);


    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_BC_set_properties__doc__[] = "";
char pyCitcom_BC_set_properties__name__[] = "BC_set_properties";

PyObject * pyCitcom_BC_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "OOO:BC_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.bc]\n"));

    getIntProperty(properties, "side_sbcs", E->control.side_sbcs, fp);
    getIntProperty(properties, "pseudo_free_surf", E->control.pseudo_free_surf, fp);

    getIntProperty(properties, "topvbc", E->mesh.topvbc, fp);
    getFloatProperty(properties, "topvbxval", E->control.VBXtopval, fp);
    getFloatProperty(properties, "topvbyval", E->control.VBYtopval, fp);

    getIntProperty(properties, "botvbc", E->mesh.botvbc, fp);
    getFloatProperty(properties, "botvbxval", E->control.VBXbotval, fp);
    getFloatProperty(properties, "botvbyval", E->control.VBYbotval, fp);

    getIntProperty(properties, "toptbc", E->mesh.toptbc, fp);
    getFloatProperty(properties, "toptbcval", E->control.TBCtopval, fp);

    getIntProperty(properties, "bottbc", E->mesh.bottbc, fp);
    getFloatProperty(properties, "bottbcval", E->control.TBCbotval, fp);

    getIntProperty(properties, "temperature_bound_adj", E->control.temperature_bound_adj, fp);
    getFloatProperty(properties, "depth_bound_adj", E->control.depth_bound_adj, fp);
    getFloatProperty(properties, "width_bound_adj", E->control.width_bound_adj, fp);


    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Const_set_properties__doc__[] = "";
char pyCitcom_Const_set_properties__name__[] = "Const_set_properties";

PyObject * pyCitcom_Const_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;
    float radius;

    if (!PyArg_ParseTuple(args, "OOO:Const_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.const]\n"));

    getFloatProperty(properties, "radius", radius, fp);
    getFloatProperty(properties, "density", E->data.density, fp);
    getFloatProperty(properties, "thermdiff", E->data.therm_diff, fp);
    getFloatProperty(properties, "gravacc", E->data.grav_acc, fp);
    getFloatProperty(properties, "thermexp", E->data.therm_exp, fp);
    getFloatProperty(properties, "refvisc", E->data.ref_viscosity, fp);
    getFloatProperty(properties, "cp", E->data.Cp, fp);
    getFloatProperty(properties, "density_above", E->data.density_above, fp);
    getFloatProperty(properties, "density_below", E->data.density_below, fp);

    E->data.therm_cond = E->data.therm_diff * E->data.density * E->data.Cp;
    E->data.ref_temperature = E->control.Atemp * E->data.therm_diff
	* E->data.ref_viscosity / (radius * radius * radius)
	/ (E->data.density * E->data.grav_acc * E->data.therm_exp);

    getFloatProperty(properties, "z_lith", E->viscosity.zlith, fp);
    getFloatProperty(properties, "z_410", E->viscosity.z410, fp);
    getFloatProperty(properties, "z_lmantle", E->viscosity.zlm, fp);
    getFloatProperty(properties, "z_cmb", E->viscosity.zcmb, fp); /* this is used as the D" phase change depth */

    /* convert meter to kilometer */
    E->data.radius_km = radius / 1e3;

    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_IC_set_properties__doc__[] = "";
char pyCitcom_IC_set_properties__name__[] = "IC_set_properties";

PyObject * pyCitcom_IC_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "OOO:IC_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.ic]\n"));

    getIntProperty(properties, "restart", E->control.restart, fp);
    getIntProperty(properties, "post_p", E->control.post_p, fp);
    getIntProperty(properties, "solution_cycles_init", E->monitor.solution_cycles_init, fp);
    getIntProperty(properties, "zero_elapsed_time", E->control.zero_elapsed_time, fp);

    getIntProperty(properties, "tic_method", E->convection.tic_method, fp);

    if (E->convection.tic_method == 0 || E->convection.tic_method == 3) {
	int num_perturb;

	getIntProperty(properties, "num_perturbations", num_perturb, fp);
	if(num_perturb > PERTURB_MAX_LAYERS) {
	    fprintf(stderr, "'num_perturb' greater than allowed value, set to %d\n", PERTURB_MAX_LAYERS);
	    num_perturb = PERTURB_MAX_LAYERS;
	}
	E->convection.number_of_perturbations = num_perturb;

	getIntVectorProperty(properties, "perturbl", E->convection.perturb_ll,
                             num_perturb, fp);
	getIntVectorProperty(properties, "perturbm", E->convection.perturb_mm,
                             num_perturb, fp);
	getIntVectorProperty(properties, "perturblayer", E->convection.load_depth,
                             num_perturb, fp);
	getFloatVectorProperty(properties, "perturbmag", E->convection.perturb_mag,
                               num_perturb, fp);
    }
    else if (E->convection.tic_method == 1) {
        getFloatProperty(properties, "half_space_age", E->convection.half_space_age, fp);
    }
    else if (E->convection.tic_method == 2) {
        getFloatProperty(properties, "half_space_age", E->convection.half_space_age, fp);
        getFloatVectorProperty(properties, "blob_center", E->convection.blob_center, 3, fp);
        if( E->convection.blob_center[0] == -999.0 && E->convection.blob_center[1] == -999.0 && E->convection.blob_center[2] == -999.0 ) {
            E->convection.blob_center[0] = 0.5*(E->control.theta_min+E->control.theta_max);
            E->convection.blob_center[1] = 0.5*(E->control.fi_min+E->control.fi_max);
            E->convection.blob_center[2] = 0.5*(E->sphere.ri+E->sphere.ro);
        }
        getFloatProperty(properties, "blob_radius", E->convection.blob_radius, fp);
        getFloatProperty(properties, "blob_dT", E->convection.blob_dT, fp);
    }

    PUTS(("\n"));

    if (PyErr_Occurred())
      return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Output_set_properties__doc__[] = "";
char pyCitcom_Output_set_properties__name__[] = "Output_set_properties";

PyObject * pyCitcom_Output_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "OOO:Output_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.output]\n"));

    getStringProperty(properties, "output_format", E->output.format, fp);
    getStringProperty(properties, "output_optional", E->output.optional, fp);

    getIntProperty(properties, "gzdir_vtkio", E->output.gzdir.vtk_io, fp);
    getIntProperty(properties, "gzdir_rnr", E->output.gzdir.rnr, fp);
    E->output.gzdir.vtk_base_init = 0;
    /* should we save the basis vectors? (memory!) */
    E->output.gzdir.vtk_base_save = 1;

    getIntProperty(properties, "output_ll_max", E->output.llmax, fp);

    getIntProperty(properties, "cb_block_size", E->output.cb_block_size, fp);
    getIntProperty(properties, "cb_buffer_size", E->output.cb_buffer_size, fp);

    getIntProperty(properties, "sieve_buf_size", E->output.sieve_buf_size, fp);

    getIntProperty(properties, "output_alignment", E->output.alignment, fp);
    getIntProperty(properties, "output_alignment_threshold", E->output.alignment_threshold, fp);

    getIntProperty(properties, "cache_mdc_nelmts", E->output.cache_mdc_nelmts, fp);
    getIntProperty(properties, "cache_rdcc_nelmts", E->output.cache_rdcc_nelmts, fp);
    getIntProperty(properties, "cache_rdcc_nbytes", E->output.cache_rdcc_nbytes, fp);

    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_Param_set_properties__doc__[] = "";
char pyCitcom_Param_set_properties__name__[] = "Param_set_properties";

PyObject * pyCitcom_Param_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "OOO:Param_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.param]\n"));

    getIntProperty(properties, "reference_state", E->refstate.choice, fp);
    if(E->refstate.choice == 0) {
        getStringProperty(properties, "refstate_file", E->refstate.filename, fp);
    }

    getIntProperty(properties, "file_vbcs", E->control.vbcs_file, fp);
    getStringProperty(properties, "vel_bound_file", E->control.velocity_boundary_file, fp);

    getIntProperty(properties, "mat_control", E->control.mat_control, fp);
    getStringProperty(properties, "mat_file", E->control.mat_file, fp);

    getIntProperty(properties, "lith_age", E->control.lith_age, fp);
    getStringProperty(properties, "lith_age_file", E->control.lith_age_file, fp);
    getIntProperty(properties, "lith_age_time", E->control.lith_age_time, fp);
    getFloatProperty(properties, "lith_age_depth", E->control.lith_age_depth, fp);
    getFloatProperty(properties, "mantle_temp", E->control.lith_age_mantle_temp, fp);

    getFloatProperty(properties, "start_age", E->control.start_age, fp);
    getIntProperty(properties, "reset_startage", E->control.reset_startage, fp);

    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_Phase_set_properties__doc__[] = "";
char pyCitcom_Phase_set_properties__name__[] = "Phase_set_properties";

PyObject * pyCitcom_Phase_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "OOO:Phase_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.phase]\n"));

    getFloatProperty(properties, "Ra_410", E->control.Ra_410, fp);
    getFloatProperty(properties, "clapeyron410", E->control.clapeyron410, fp);
    getFloatProperty(properties, "transT410", E->control.transT410, fp);
    getFloatProperty(properties, "width410", E->control.width410, fp);

    if (E->control.width410!=0.0)
	E->control.width410 = 1.0/E->control.width410;

    getFloatProperty(properties, "Ra_670", E->control.Ra_670 , fp);
    getFloatProperty(properties, "clapeyron670", E->control.clapeyron670, fp);
    getFloatProperty(properties, "transT670", E->control.transT670, fp);
    getFloatProperty(properties, "width670", E->control.width670, fp);

    if (E->control.width670!=0.0)
	E->control.width670 = 1.0/E->control.width670;

    getFloatProperty(properties, "Ra_cmb", E->control.Ra_cmb, fp);
    getFloatProperty(properties, "clapeyroncmb", E->control.clapeyroncmb, fp);
    getFloatProperty(properties, "transTcmb", E->control.transTcmb, fp);
    getFloatProperty(properties, "widthcmb", E->control.widthcmb, fp);

    if (E->control.widthcmb!=0.0)
	E->control.widthcmb = 1.0/E->control.widthcmb;

    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;

}



char pyCitcom_Solver_set_properties__doc__[] = "";
char pyCitcom_Solver_set_properties__name__[] = "Solver_set_properties";

PyObject * pyCitcom_Solver_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;
    float tmp;

    if (!PyArg_ParseTuple(args, "OOO:Solver_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver]\n"));

    getStringProperty(properties, "datadir", E->control.data_dir, fp);
    getStringProperty(properties, "datafile", E->control.data_prefix, fp);
    getStringProperty(properties, "datadir_old", E->control.data_dir_old, fp);
    getStringProperty(properties, "datafile_old", E->control.data_prefix_old, fp);

    getFloatProperty(properties, "rayleigh", E->control.Atemp, fp);
    getFloatProperty(properties, "dissipation_number", E->control.disptn_number, fp);
    getFloatProperty(properties, "gruneisen", tmp, fp);
     /* special case: if tmp==0, set gruneisen as inf */
     if(tmp != 0)
        E->control.inv_gruneisen = 1/tmp;
    else
        E->control.inv_gruneisen = 0;

    getFloatProperty(properties, "surfaceT", E->control.surface_temp, fp);
    /*getFloatProperty(properties, "adiabaticT0", E->control.adiabaticT0, fp);*/
    getFloatProperty(properties, "Q0", E->control.Q0, fp);

    getIntProperty(properties, "stokes_flow_only", E->control.stokes, fp);

    getIntProperty(properties, "verbose", E->control.verbose, fp);
    getIntProperty(properties, "see_convergence", E->control.print_convergence, fp);

    /* parameters not used in pyre version,
       assigned value here to prevent uninitialized access */
    E->advection.min_timesteps = 1;
    E->advection.max_timesteps = 1;
    E->advection.max_total_timesteps = 1;
    E->control.checkpoint_frequency = 1;
    E->control.record_every = 1;
    E->control.record_all_until = 1;

    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Sphere_set_properties__doc__[] = "";
char pyCitcom_Sphere_set_properties__name__[] = "Sphere_set_properties";

PyObject * pyCitcom_Sphere_set_properties(PyObject *self, PyObject *args)
{
    void full_set_3dsphere_defaults2(struct All_variables *);
    void regional_set_3dsphere_defaults2(struct All_variables *);

    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "OOO:Sphere_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.mesher]\n"));

    getIntProperty(properties, "nproc_surf", E->parallel.nprocxy, fp);

    getIntProperty(properties, "nprocx", E->parallel.nprocx, fp);
    getIntProperty(properties, "nprocy", E->parallel.nprocy, fp);
    getIntProperty(properties, "nprocz", E->parallel.nprocz, fp);

    if (E->parallel.nprocxy == 12)
	if (E->parallel.nprocx != E->parallel.nprocy) {
	    char errmsg[] = "!!!! nprocx must equal to nprocy";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
    }

    getIntProperty(properties, "coor", E->control.coor, fp);
    getFloatVectorProperty(properties, "coor_refine", E->control.coor_refine, 4, fp);
    getStringProperty(properties, "coor_file", E->control.coor_file, fp);

    getIntProperty(properties, "nodex", E->mesh.nox, fp);
    getIntProperty(properties, "nodey", E->mesh.noy, fp);
    getIntProperty(properties, "nodez", E->mesh.noz, fp);
    getIntProperty(properties, "levels", E->mesh.levels, fp);

    E->mesh.mgunitx = (E->mesh.nox - 1) / E->parallel.nprocx /
	(int) pow(2.0, E->mesh.levels - 1);
    E->mesh.mgunity = (E->mesh.noy - 1) / E->parallel.nprocy /
	(int) pow(2.0, E->mesh.levels - 1);
    E->mesh.mgunitz = (E->mesh.noz - 1) / E->parallel.nprocz /
	(int) pow(2.0, E->mesh.levels - 1);

    if (E->parallel.nprocxy == 12) {
	if (E->mesh.nox != E->mesh.noy) {
	    char errmsg[] = "!!!! nodex must equal to nodey";
	    PyErr_SetString(PyExc_SyntaxError, errmsg);
	    return NULL;
	}
    }

    getDoubleProperty(properties, "radius_outer", E->sphere.ro, fp);
    getDoubleProperty(properties, "radius_inner", E->sphere.ri, fp);


    if (E->parallel.nprocxy == 12) {
        full_set_3dsphere_defaults2(E);
    }
    else {
	getDoubleProperty(properties, "theta_min", E->control.theta_min, fp);
	getDoubleProperty(properties, "theta_max", E->control.theta_max, fp);
	getDoubleProperty(properties, "fi_min", E->control.fi_min, fp);
	getDoubleProperty(properties, "fi_max", E->control.fi_max, fp);

        regional_set_3dsphere_defaults2(E);
    }

    E->mesh.layer[1] = 1;
    E->mesh.layer[2] = 1;
    E->mesh.layer[3] = 1;

    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Tracer_set_properties__doc__[] = "";
char pyCitcom_Tracer_set_properties__name__[] = "Tracer_set_properties";

PyObject * pyCitcom_Tracer_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;
    double tmp;
    char message[100];

    if (!PyArg_ParseTuple(args, "OOO:Tracer_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.tracer]\n"));

    getIntProperty(properties, "tracer", E->control.tracer, fp);

    getIntProperty(properties, "tracer_enriched", E->control.tracer_enriched, fp);
    if(E->control.tracer_enriched) {
        if(!E->control.tracer)
            myerror(E,"need to switch on tracers for tracer_enriched");

        getFloatProperty(properties, "Q0_enriched", E->control.Q0ER, fp);
        snprintf(message,100,"using compositionally enriched heating: C = 0: %g C = 1: %g (only one composition!)",
                 E->control.Q0,E->control.Q0ER);
        report(E,message);
    }

    getIntProperty(properties, "tracer_ic_method",
                   E->trace.ic_method, fp);

    if (E->trace.ic_method==0) {
        getIntProperty(properties, "tracers_per_element",
                       E->trace.itperel, fp);
    }
    else if (E->trace.ic_method==1) {
        getStringProperty(properties, "tracer_file",
                          E->trace.tracer_file, fp);
    }
    else if (E->trace.ic_method==2) {
    }
    else {
        fprintf(stderr,"Sorry, tracer_ic_method only 0, 1 and 2 available\n");
        fflush(stderr);
        parallel_process_termination();
    }

    getIntProperty(properties, "tracer_flavors", E->trace.nflavors, fp);

    getIntProperty(properties, "ic_method_for_flavors", E->trace.ic_method_for_flavors, fp);

    if (E->trace.nflavors > 1) {
        switch(E->trace.ic_method_for_flavors){
        case 0:			/* layer */
            E->trace.z_interface = (double*) malloc((E->trace.nflavors-1)
                                                    *sizeof(double));

            getDoubleVectorProperty(properties, "z_interface", E->trace.z_interface, E->trace.nflavors-1, fp);
            break;
        case 1:			/* from grid in top n materials */
            /* file from which to read */
            getStringProperty(properties, "ictracer_grd_file", E->trace.ggrd_file, fp);
            /* which top layers to use */
            getIntProperty(properties, "ictracer_grd_layers", E->trace.ggrd_layers, fp);
            break;
        default:
            fprintf(stderr,"ic_method_for_flavors %i undefined\n",E->trace.ic_method_for_flavors);
            parallel_process_termination();
            break;
        }
    }

    getIntProperty(properties, "itracer_warnings", E->trace.itracer_warnings, fp);

    getIntProperty(properties, "chemical_buoyancy",
                   E->composition.ichemical_buoyancy, fp);

    if (E->control.tracer && E->composition.ichemical_buoyancy==1) {
        getIntProperty(properties, "buoy_type", E->composition.ibuoy_type, fp);

        if (E->composition.ibuoy_type==0)
            E->composition.ncomp = E->trace.nflavors;
        else if (E->composition.ibuoy_type==1)
            E->composition.ncomp = E->trace.nflavors - 1;

        E->composition.buoyancy_ratio = (double*) malloc(E->composition.ncomp
                                                         *sizeof(double));

        getDoubleVectorProperty(properties, "buoyancy_ratio", E->composition.buoyancy_ratio, E->composition.ncomp, fp);
    }


    if(E->parallel.nprocxy == 12) {

        getDoubleProperty(properties, "regular_grid_deltheta", tmp, fp);
        E->trace.deltheta[0] = tmp;
        getDoubleProperty(properties, "regular_grid_delphi", tmp, fp);
        E->trace.delphi[0] = tmp;

        E->trace.ianalytical_tracer_test = 0;
        //getIntProperty(properties, "analytical_tracer_test", E->trace.ianalytical_tracer_test, fp);


        E->composition.icompositional_rheology = 0;
        /*
        getIntProperty(properties, "compositional_rheology", E->composition.icompositional_rheology, fp);

        if (E->composition.icompositional_rheology==1) {
            getDoubleProperty(properties, "compositional_prefactor", E->composition.compositional_rheology_prefactor, fp);
        }
        */
    }
    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_Visc_set_properties__doc__[] = "";
char pyCitcom_Visc_set_properties__name__[] = "Visc_set_properties";

PyObject * pyCitcom_Visc_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;
    int num_mat, i;

    if (!PyArg_ParseTuple(args, "OOO:Visc_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.visc]\n"));

    getStringProperty(properties, "Viscosity", E->viscosity.STRUCTURE, fp);
    if (strcmp(E->viscosity.STRUCTURE,"system") == 0)
	E->viscosity.FROM_SYSTEM = 1;
    else
	E->viscosity.FROM_SYSTEM = 0;

    getIntProperty(properties, "visc_smooth_method", E->viscosity.smooth_cycles, fp);
    getIntProperty(properties, "VISC_UPDATE", E->viscosity.update_allowed, fp);

#define MAX_MAT 40

    getIntProperty(properties, "num_mat", num_mat, fp);
    if(num_mat > MAX_MAT) {
	/* max. allowed material types = 40 */
	fprintf(stderr, "'num_mat' greater than allowed value, set to %d\n", MAX_MAT);
	num_mat = MAX_MAT;
    }
    E->viscosity.num_mat = num_mat;

    getFloatVectorProperty(properties, "visc0",
                           E->viscosity.N0, num_mat, fp);

    getIntProperty(properties, "TDEPV", E->viscosity.TDEPV, fp);
    getIntProperty(properties, "rheol", E->viscosity.RHEOL, fp);
    getFloatVectorProperty(properties, "viscE",
                           E->viscosity.E, num_mat, fp);
    getFloatVectorProperty(properties, "viscT",
                           E->viscosity.T, num_mat, fp);
    getFloatVectorProperty(properties, "viscZ",
                           E->viscosity.Z, num_mat, fp);

    getIntProperty(properties, "SDEPV", E->viscosity.SDEPV, fp);
    getFloatVectorProperty(properties, "sdepv_expt",
                           E->viscosity.sdepv_expt, num_mat, fp);

    getIntProperty(properties, "PDEPV", E->viscosity.PDEPV, fp);
    if (E->viscosity.PDEPV) {
        E->viscosity.pdepv_visited = 0;
        getIntProperty(properties, "pdepv_eff", E->viscosity.pdepv_eff, fp);
        getFloatVectorProperty(properties, "pdepv_a",
                               E->viscosity.pdepv_a, num_mat, fp);
        getFloatVectorProperty(properties, "pdepv_b",
                               E->viscosity.pdepv_b, num_mat, fp);
        getFloatVectorProperty(properties, "pdepv_y",
                               E->viscosity.pdepv_y, num_mat, fp);
        getFloatProperty(properties, "pdepv_offset", E->viscosity.pdepv_offset, fp);
    }
    if(E->viscosity.PDEPV || E->viscosity.SDEPV)
        getFloatProperty(properties, "sdepv_misfit", E->viscosity.sdepv_misfit, fp);

    getIntProperty(properties, "CDEPV", E->viscosity.CDEPV, fp);
    if(E->viscosity.CDEPV){	/* compositional viscosity */
        if(!E->control.tracer)
            myerror(E,"error: CDEPV requires tracers, but tracer is off");
        if(E->trace.nflavors > 10)
            myerror(E,"error: too many flavors for CDEPV");
        /* read in flavor factors */
        getFloatVectorProperty(properties, "cdepv_ff",
                               E->viscosity.cdepv_ff, E->trace.nflavors, fp);
        /* and take the log because we're using a geometric avg */
        for(i=0;i<E->trace.nflavors;i++)
            E->viscosity.cdepv_ff[i] = log(E->viscosity.cdepv_ff[i]);
    }

    getIntProperty(properties, "low_visc_channel", E->viscosity.channel, fp);
    getIntProperty(properties, "low_visc_wedge", E->viscosity.wedge, fp);

    getFloatProperty(properties, "lv_min_radius", E->viscosity.lv_min_radius, fp);
    getFloatProperty(properties, "lv_max_radius", E->viscosity.lv_max_radius, fp);
    getFloatProperty(properties, "lv_channel_thickness", E->viscosity.lv_channel_thickness, fp);
    getFloatProperty(properties, "lv_reduction", E->viscosity.lv_reduction, fp);

    getIntProperty(properties, "VMIN", E->viscosity.MIN, fp);
    getFloatProperty(properties, "visc_min", E->viscosity.min_value, fp);

    getIntProperty(properties, "VMAX", E->viscosity.MAX, fp);
    getFloatProperty(properties, "visc_max", E->viscosity.max_value, fp);

    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_Incompressible_set_properties__doc__[] = "";
char pyCitcom_Incompressible_set_properties__name__[] = "Incompressible_set_properties";

PyObject * pyCitcom_Incompressible_set_properties(PyObject *self, PyObject *args)
{
    PyObject *obj, *properties, *out;
    struct All_variables *E;
    FILE *fp;

    if (!PyArg_ParseTuple(args, "OOO:Incompressible_set_properties",
			  &obj, &properties, &out))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));
    fp = get_output_stream(out, E);

    PUTS(("[CitcomS.solver.vsolver]\n"));

    getStringProperty(properties, "Solver", E->control.SOLVER_TYPE, fp);
    getIntProperty(properties, "node_assemble", E->control.NASSEMBLE, fp);
    getIntProperty(properties, "precond", E->control.precondition, fp);

    getDoubleProperty(properties, "accuracy", E->control.accuracy, fp);
    getFloatProperty(properties, "tole_compressibility", E->control.tole_comp, fp);

    getIntProperty(properties, "mg_cycle", E->control.mg_cycle, fp);
    getIntProperty(properties, "down_heavy", E->control.down_heavy, fp);
    getIntProperty(properties, "up_heavy", E->control.up_heavy, fp);

    getIntProperty(properties, "vlowstep", E->control.v_steps_low, fp);
    getIntProperty(properties, "vhighstep", E->control.v_steps_high, fp);
    getIntProperty(properties, "piterations", E->control.p_iterations, fp);

    getIntProperty(properties, "aug_lagr", E->control.augmented_Lagr, fp);
    getDoubleProperty(properties, "aug_number", E->control.augmented, fp);

    getIntProperty(properties, "remove_rigid_rotation", E->control.remove_rigid_rotation, fp);

    if(E->control.inv_gruneisen != 0) {
        /* which compressible solver to use: "cg" or "bicg" */
        getStringProperty(properties, "uzawa", E->control.uzawa, fp);
        if(strcmp(E->control.uzawa, "cg") == 0) {
            /* more convergence parameters for "cg" */
            getIntProperty(properties, "compress_iter_maxstep", E->control.compress_iter_maxstep, fp);
            getFloatProperty(properties, "relative_err_accuracy", E->control.relative_err_accuracy, fp);
        }
    }

    PUTS(("\n"));

    Py_INCREF(Py_None);
    return Py_None;
}




/*==========================================================*/
/* helper functions */

FILE *get_output_stream(PyObject *out, struct All_variables*E)
{
    if (PyFile_Check(out)) {
        return PyFile_AsFile(out);
    }
    return NULL;
}


int _getStringProperty(PyObject* properties, char* attribute,
                       char* value, size_t valueSize, FILE* fp)
{
    PyObject *prop;
    char *buffer;
    Py_ssize_t length;

    if (!(prop = PyObject_GetAttrString(properties, attribute)))
        return -1;
    if (-1 == PyString_AsStringAndSize(prop, &buffer, &length))
        return -1;

    if (length >= (Py_ssize_t)valueSize) {
        PyErr_Format(PyExc_ValueError,
                     "value of '%s' cannot exceed %zu characters in length",
                     attribute, valueSize);
        return -1;
    }
    strcpy(value, buffer);

    if (fp)
        fprintf(fp, "%s=%s\n", attribute, value);

    return 0;
}


#define getTYPEProperty _getIntProperty
#define getTYPEVectorProperty _getIntVectorProperty
#define PyTYPE_Check PyInt_Check
#define CTYPE int
#define PyTYPE_AsCTYPE PyInt_AsLong
#define MESSAGE "an integer is required"
#define FORMAT "%d"
#include "getProperty.h"

#undef getTYPEProperty
#undef getTYPEVectorProperty
#undef PyTYPE_Check
#undef CTYPE
#undef PyTYPE_AsCTYPE
#undef MESSAGE
#undef FORMAT

#define getTYPEProperty _getFloatProperty
#define getTYPEVectorProperty _getFloatVectorProperty
#define PyTYPE_Check PyFloat_Check
#define CTYPE float
#define PyTYPE_AsCTYPE PyFloat_AsDouble
#define MESSAGE "a float is required"
#define FORMAT "%g"
#include "getProperty.h"


#undef getTYPEProperty
#undef getTYPEVectorProperty
#undef PyTYPE_Check
#undef CTYPE
#undef PyTYPE_AsCTYPE
#undef MESSAGE
#undef FORMAT

#define getTYPEProperty _getDoubleProperty
#define getTYPEVectorProperty _getDoubleVectorProperty
#define PyTYPE_Check PyFloat_Check
#define CTYPE double
#define PyTYPE_AsCTYPE PyFloat_AsDouble
#define MESSAGE "a float is required"
#define FORMAT "%g"
#include "getProperty.h"


/* $Id$ */

/* End of file */
