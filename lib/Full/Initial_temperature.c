
#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h> /* for "system" command */
#include <strings.h>

#include "lith_age.h"
#include "parsing.h"

#include "initial_temperature.h"
void restart_tic(struct All_variables *);
void restart_tic_from_file(struct All_variables *);
void construct_tic(struct All_variables *);
void construct_tic_full(struct All_variables *);
void debug_tic(struct All_variables *);

void parallel_process_termination();
void temperatures_conform_bcs();




void tic_input(struct All_variables *E)
{

  int m = E->parallel.me;
  int noz = E->lmesh.noz;

  /* This part put a temperature anomaly at depth where the global
     node number is equal to load_depth. The horizontal pattern of
     the anomaly is given by spherical harmonic ll & mm. */

  input_int("num_perturbations",&(E->number_of_perturbations),"0,0,32",m);

  if (E->number_of_perturbations > 0) {
    if (! input_float_vector("perturbmag",E->number_of_perturbations,E->perturb_mag,m) ) {
      fprintf(stderr,"Missing input parameter: 'perturbmag'\n");
      parallel_process_termination();
    }
    if (! input_int_vector("perturbm",E->number_of_perturbations,E->perturb_mm,m) ) {
      fprintf(stderr,"Missing input parameter: 'perturbm'\n");
      parallel_process_termination();
    }
    if (! input_int_vector("perturbl",E->number_of_perturbations,E->perturb_ll,m) ) {
      fprintf(stderr,"Missing input parameter: 'perturbml'\n");
      parallel_process_termination();
    }
    if (! input_int_vector("perturblayer",E->number_of_perturbations,E->load_depth,m) ) {
      fprintf(stderr,"Missing input parameter: 'perturblayer'\n");
      parallel_process_termination();
    }
  }
  else {
    E->number_of_perturbations = 1;
    E->perturb_mag[0] = 0.05 * (E->control.TBCbotval - E->control.TBCtopval);
    E->perturb_mm[0] = 2;
    E->perturb_ll[0] = 2;
    E->load_depth[0] = (noz+1)/2;
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
    construct_tic_full(E);

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


void construct_tic_full(struct All_variables *E)
{
  int i, j, k, kk, m, p, node;
  int nox, noy, noz, gnoz;
  double r1, f1, t1;
  int mm, ll;
  double con;
  double modified_plgndr_a(int, int, double);

  noy=E->lmesh.noy;
  nox=E->lmesh.nox;
  noz=E->lmesh.noz;
  gnoz=E->mesh.noz;
  
  /* set up a linear temperature profile first */
  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=noy;i++)
      for(j=1;j<=nox;j++)
	for(k=1;k<=noz;k++) {
	  node=k+(j-1)*noz+(i-1)*nox*noz;
	  r1=E->sx[m][3][node];
	  E->T[m][node] = E->control.TBCbotval - (E->control.TBCtopval + E->control.TBCbotval)*(r1 - E->sphere.ri)/(E->sphere.ro - E->sphere.ri);
	}

  /* This part put a temperature anomaly at depth where the global
     node number is equal to load_depth. The horizontal pattern of
     the anomaly is given by spherical harmonic ll & mm. */
  
  for (p=0; p<E->number_of_perturbations; p++) {
    mm = E->perturb_mm[p];
    ll = E->perturb_ll[p];
    con = E->perturb_mag[p];
    kk = E->load_depth[p];

    if ( kk<=0 || kk>gnoz ) continue;

    k = kk - E->lmesh.nzs + 1;
    if ( k<=0 || k>noz ) continue; // if layer k is not inside this proc.

    if (E->parallel.me_loc[1] == 0 && E->parallel.me_loc[2] == 0 
	&& E->sphere.capid[1] == 1 ) 
      fprintf(stderr,"Initial temperature perturbation:  layer=%d  mag=%g  l=%d  m=%d\n", kk, con, ll, mm);
    
    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=noy;i++)
	for(j=1;j<=nox;j++) {
	  node=k+(j-1)*noz+(i-1)*nox*noz;
	  t1=E->sx[m][1][node];
	  f1=E->sx[m][2][node];

	  E->T[m][node] += con*modified_plgndr_a(ll,mm,t1)*cos(mm*f1);
	  E->T[m][node] = max( min(E->T[m][node], E->control.TBCbotval),
			       E->control.TBCtopval);
	}
  }

  temperatures_conform_bcs(E);

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


#if 0

void convection_initial_temperature(E)
     struct All_variables *E;
{
    int i,j,k,p,node,ii,jj,m,mm,ll;
    int in1,in2,in3,instance,nox,noy,noz,noz2;
    int load_depth;
    char output_file[255],input_s[1000];

    float global_fmax(),global_fmin(),amaxx,aminx;
    double con,temp,t1,f1,r1,sphere_h();
    double modified_plgndr_a(),multis();
    double temp1,global_tdot_d(),drand48();
    float rad,beta; 
    float a,b,c,d,e,f,g; 
    FILE *fp;

    float v1,v2,v3;

    void temperatures_conform_bcs();
    void thermal_buoyancy();
    void parallel_process_termination();
    void sphere_harmonics_layer();
    void inv_sphere_harmonics();
    
    const int dims=E->mesh.nsd;
    rad = 180.0/M_PI;

    noy=E->lmesh.noy;  
    nox=E->lmesh.nox;  
    noz=E->lmesh.noz;  

    noz2=(E->mesh.noz-1)/2+1;  

    if ((E->control.restart || E->control.post_p))    {
/* used if restarting from a previous run. CPC 1/28/00 */
/*
        E->monitor.solution_cycles=(E->control.restart)?E->control.restart:E->advection.max_timesteps;
*/
        ii = E->monitor.solution_cycles_init;
        sprintf(output_file,"%s.velo.%d.%d",E->control.old_P_file,E->parallel.me,ii);
        fp=fopen(output_file,"r");
	if (fp == NULL) {
          fprintf(E->fp,"(Initial_temperature.c #1) Cannot open %s\n",output_file);
          exit(8);
	}
        fgets(input_s,1000,fp);
        sscanf(input_s,"%d %d %f",&ll,&mm,&E->monitor.elapsed_time);

        for(m=1;m<=E->sphere.caps_per_proc;m++)  {
          fgets(input_s,1000,fp);
          sscanf(input_s,"%d %d",&ll,&mm);
          for(i=1;i<=E->lmesh.nno;i++)  {
            fgets(input_s,1000,fp);
            sscanf(input_s,"%g %g %g %f",&(v1),&(v2),&(v3),&(g));
/*            E->sphere.cap[m].V[1][i] = d;
            E->sphere.cap[m].V[1][i] = e;
            E->sphere.cap[m].V[1][i] = f;  */
	    E->T[m][i] = max(0.0,min(g,1.0));
            }
          }

        fclose (fp);

        }

    else   {
        int number_of_perturbations;
        int perturb_ll[32], perturb_mm[32], load_depth[32];
	float perturb_mag[32];

        m = E->parallel.me;

      /* This part put a temperature anomaly at depth where the global 
	 node number is equal to load_depth. The horizontal pattern of
	 the anomaly is given by spherical harmonic ll & mm. The amplitude
	 of the anomaly is such that its integral with depth is 1  */

	input_int("num_perturbations",&number_of_perturbations,"0,0,32",m);

	if (number_of_perturbations > 0) {
	  if (! input_float_vector("perturbmag",number_of_perturbations,perturb_mag,m) ) {
	    fprintf(stderr,"Missing input parameter: 'perturbmag'\n");
	    parallel_process_termination();
	  }
	  if (! input_int_vector("perturbm",number_of_perturbations,perturb_mm,m) ) {
	    fprintf(stderr,"Missing input parameter: 'perturbm'\n");
	    parallel_process_termination();
	  }
	  if (! input_int_vector("perturbl",number_of_perturbations,perturb_ll,m) ) {;
	    fprintf(stderr,"Missing input parameter: 'perturbml'\n");
	    parallel_process_termination();
	  }
	  if (! input_int_vector("perturblayer",number_of_perturbations,load_depth,m) ) {
	    fprintf(stderr,"Missing input parameter: 'perturblayer'\n");
	    parallel_process_termination();
	  }
	}
	else {
	  number_of_perturbations = 1;
          perturb_mag[0] = E->mesh.elz/(E->sphere.ro-E->sphere.ri);
	  perturb_mm[0] = 2;
	  perturb_ll[0] = 2;
	  load_depth[0] = noz/2;
	}

	for(m=1;m<=E->sphere.caps_per_proc;m++)
	  for(i=1;i<=noy;i++)  
	    for(j=1;j<=nox;j++) 
	      for(k=1;k<=noz;k++)  {
		ii = k + E->lmesh.nzs - 1;
		node=k+(j-1)*noz+(i-1)*nox*noz;
		t1=E->sx[m][1][node];
		f1=E->sx[m][2][node];
		r1=E->sx[m][3][node];
		E->T[m][node] = (E->control.TBCtopval + E->control.TBCbotval)*(noz-k)/(noz-1.0);

		for (p=0; p<number_of_perturbations; p++) {
		  mm = perturb_mm[p];
		  ll = perturb_ll[p];
		  con = perturb_mag[p];

		  if (ii == load_depth[p]) {
		    E->T[m][node] += con*modified_plgndr_a(ll,mm,t1)*cos(mm*f1);
		    E->T[m][node] = max(min(E->T[m][node], 1.0), 0.0);
		  }
		}
	}
    }
  temperatures_conform_bcs(E);

  if (E->control.verbose)  {
    fprintf(E->fp_out,"output_temperature\n");
    for(m=1;m<=E->sphere.caps_per_proc;m++)        {
      fprintf(E->fp_out,"for cap %d\n",E->sphere.capid[m]);
      for (j=1;j<=E->lmesh.nno;j++)
         fprintf(E->fp_out,"X = %.6e Z = %.6e Y = %.6e T[%06d] = %.6e \n",E->sx[m][1][j],E->sx[m][2][j],E->sx[m][3][j],j,E->T[m][j]);
      }
    fflush(E->fp_out);
    }

    return; 
    }


#endif
