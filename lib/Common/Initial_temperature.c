
#include <math.h>

#include "global_defs.h"
#include "lith_age.h"
#include "parsing.h"

void parallel_process_termination();
void temperatures_conform_bcs();
void construct_tic_from_input(struct All_variables *);

#include "initial_temperature.h"
void restart_tic(struct All_variables *);
void construct_tic(struct All_variables *);
void debug_tic(struct All_variables *);
void restart_tic_from_file(struct All_variables *);



void tic_input(struct All_variables *E)
{

  int m = E->parallel.me;
  int noz = E->lmesh.noz;
  int n;

  /* This part put a temperature anomaly at depth where the global
     node number is equal to load_depth. The horizontal pattern of
     the anomaly is given by spherical harmonic ll & mm. */

  input_int("num_perturbations", &n, "0,0,32", m);

  if (n > 0) {
    E->convection.number_of_perturbations = n;

    if (! input_float_vector("perturbmag", n, E->convection.perturb_mag, m) ) {
      fprintf(stderr,"Missing input parameter: 'perturbmag'\n");
      parallel_process_termination();
    }
    if (! input_int_vector("perturbm", n, E->convection.perturb_mm, m) ) {
      fprintf(stderr,"Missing input parameter: 'perturbm'\n");
      parallel_process_termination();
    }
    if (! input_int_vector("perturbl", n, E->convection.perturb_ll, m) ) {
      fprintf(stderr,"Missing input parameter: 'perturbl'\n");
      parallel_process_termination();
    }
    if (! input_int_vector("perturblayer", n, E->convection.load_depth, m) ) {
      fprintf(stderr,"Missing input parameter: 'perturblayer'\n");
      parallel_process_termination();
    }
  }
  else {
    E->convection.number_of_perturbations = 1;
    E->convection.perturb_mag[0] = 1;
    E->convection.perturb_mm[0] = 2;
    E->convection.perturb_ll[0] = 2;
    E->convection.load_depth[0] = (noz+1)/2;
  }

  return;
}



void convection_initial_temperature(struct All_variables *E)
{
  if (E->control.restart)
    restart_tic(E);
  else if (E->control.post_p)
    restart_tic(E);
  else
    construct_tic(E);

  /* Note: it is the callee's responsibility to conform tbc. */
  /* like a call to temperatures_conform_bcs(E); */

  if (E->control.verbose)
    debug_tic(E);

  return;
}



void restart_tic(struct All_variables *E)
{
  if (E->control.lith_age)
    lith_age_restart_tic(E);
  else
    restart_tic_from_file(E);

  return;
}


void construct_tic(struct All_variables *E)
{
  if (E->control.lith_age)
    lith_age_construct_tic(E);
  else
    construct_tic_from_input(E);

  return;
}


void debug_tic(struct All_variables *E)
{
  int m, j;

  fprintf(E->fp_out,"output_temperature\n");
  for(m=1;m<=E->sphere.caps_per_proc;m++)        {
    fprintf(E->fp_out,"for cap %d\n",E->sphere.capid[m]);
    for (j=1;j<=E->lmesh.nno;j++)
      fprintf(E->fp_out,"X = %.6e Z = %.6e Y = %.6e T[%06d] = %.6e \n",E->sx[m][1][j],E->sx[m][2][j],E->sx[m][3][j],j,E->T[m][j]);
  }
  fflush(E->fp_out);

  return;
}



void restart_tic_from_file(struct All_variables *E)
{
  int ii, ll, mm;
  float notusedhere;
  int i, m;
  char output_file[255], input_s[1000];
  FILE *fp;

  float v1, v2, v3, g;

  ii = E->monitor.solution_cycles_init;
  sprintf(output_file,"%s.velo.%d.%d",E->control.old_P_file,E->parallel.me,ii);
  fp=fopen(output_file,"r");
  if (fp == NULL) {
    fprintf(E->fp,"(Initial_temperature.c #1) Cannot open %s\n",output_file);
    parallel_process_termination();
  }

  fgets(input_s,1000,fp);
  sscanf(input_s,"%d %d %f",&ll,&mm,&notusedhere);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    fgets(input_s,1000,fp);
    sscanf(input_s,"%d %d",&ll,&mm);
    for(i=1;i<=E->lmesh.nno;i++)  {
      fgets(input_s,1000,fp);
      sscanf(input_s,"%g %g %g %f",&(v1),&(v2),&(v3),&(g));

      /*  E->sphere.cap[m].V[1][i] = d;
	  E->sphere.cap[m].V[1][i] = e;
	  E->sphere.cap[m].V[1][i] = f;  */
      E->T[m][i] = max(0.0,min(g,1.0));
    }
  }
  fclose (fp);

  temperatures_conform_bcs(E);
  return;
}


