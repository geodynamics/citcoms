/* Routine to process the output of the finite element cycles
   and to turn them into a coherent suite  files  */


#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "output.h"

void output_coord(struct All_variables *);
void output_mat(struct All_variables *);
void output_velo(struct All_variables *, int);
void output_visc_prepare(struct All_variables *, float **);
void output_visc(struct All_variables *, int);
void output_surf_botm(struct All_variables *, int);
void output_ave_r(struct All_variables *, int);

extern void parallel_process_termination();
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**,
			 float**, float**, int);

/**********************************************************************/


void output(struct All_variables *E, int cycles)
{

  if (cycles == 0) {
    output_coord(E);
    output_mat(E);
  }

  output_velo(E, cycles);
  output_visc(E, cycles);
  output_surf_botm(E, cycles);

  /* disable horizontal average output   by Tan2 */
  /* output_ave_r(E, cycles); */

  return;
}


FILE* output_open(char *filename)
{
  FILE *fp1;

  // if filename is empty, output to stderr.
  if (*filename) {
    fp1 = fopen(filename,"w");
    if (!fp1) {
      fprintf(stderr,"Cannot open file '%s'\n",filename);
      parallel_process_termination();
    }
  }
  else
    fp1 = stderr;

  return fp1;
}


void output_coord(struct All_variables *E)
{
  int i, j;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.coord.%d",E->control.data_file,E->parallel.me);
  fp1 = output_open(output_file);

  for(j=1;j<=E->sphere.caps_per_proc;j++)     {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.6e %.6e %.6e\n",E->sx[j][1][i],E->sx[j][2][i],E->sx[j][3][i]);
  }

  fclose(fp1);

  return;
}


void output_visc(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  FILE *fp1;
  int lev = E->mesh.levmax;

  sprintf(output_file,"%s.visc.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = output_open(output_file);


  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.4e\n",E->VI[lev][j][i]);
  }

  fclose(fp1);

  return;
}


void output_velo(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.velo.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = output_open(output_file);

  fprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.6e %.6e %.6e %.6e\n",E->sphere.cap[j].V[1][i],E->sphere.cap[j].V[2][i],E->sphere.cap[j].V[3][i],E->T[j][i]);
  }

  fclose(fp1);

  return;
}


void output_surf_botm(struct All_variables *E, int cycles)
{
  int i, j, s;
  char output_file[255];
  FILE* fp2;


  heat_flux(E);
  get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,cycles);

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1) {
    sprintf(output_file,"%s.surf.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp2 = output_open(output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)   {
	s = i*E->lmesh.noz;
        fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpg[j][i],E->slice.shflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
      }
    }
    fclose(fp2);
  }


  if (E->parallel.me_loc[3]==0)      {
    sprintf(output_file,"%s.botm.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp2 = output_open(output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)  {
	s = (i-1)*E->lmesh.noz + 1;
        fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpgb[j][i],E->slice.bhflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
      }
    }
    fclose(fp2);
  }

  return;
}


void output_avg_r(struct All_variables *E, int cycles)
{
  /* horizontal average output of temperature and rms velocity*/
  void return_horiz_ave_f();

  int m, i, j;
  float *S1[NCS],*S2[NCS],*S3[NCS];
  char output_file[255];
  FILE *fp1;

  // compute horizontal average here....

  for(m=1;m<=E->sphere.caps_per_proc;m++)      {
    S1[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    S2[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    S3[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  }

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    for(i=1;i<=E->lmesh.nno;i++) {
      S1[m][i] = E->T[m][i];
      S2[m][i] = E->sphere.cap[m].V[1][i]*E->sphere.cap[m].V[1][i]
          	+ E->sphere.cap[m].V[2][i]*E->sphere.cap[m].V[2][i];
      S3[m][i] = E->sphere.cap[m].V[3][i]*E->sphere.cap[m].V[3][i];
    }
  }

  return_horiz_ave_f(E,S1,E->Have.T);
  return_horiz_ave_f(E,S2,E->Have.V[1]);
  return_horiz_ave_f(E,S3,E->Have.V[2]);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    free((void *)S1[m]);
    free((void *)S2[m]);
    free((void *)S3[m]);
  }

  for (i=1;i<=E->lmesh.noz;i++) {
      E->Have.V[1][i] = sqrt(E->Have.V[1][i]);
      E->Have.V[2][i] = sqrt(E->Have.V[2][i]);
  }

  // only the first nprocz processors need to output

  if (E->parallel.me<E->parallel.nprocz)  {
    sprintf(output_file,"%s.ave_r.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp1=fopen(output_file,"w");
    for(j=1;j<=E->lmesh.noz;j++)  {
        fprintf(fp1,"%.4e %.4e %.4e %.4e\n",E->sx[1][3][j],E->Have.T[j],E->Have.V[1][j],E->Have.V[2][j]);
    }
    fclose(fp1);
  }

  return;
}



void output_mat(struct All_variables *E)
{
  int m, el;
  char output_file[255];
  FILE* fp;

  sprintf(output_file,"%s.mat.%d",E->control.data_file,E->parallel.me);
  fp = output_open(output_file);

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(el=1;el<=E->lmesh.nel;el++)
      fprintf(fp,"%d %d %f\n", el,E->mat[m][el],E->VIP[m][el]);

  fclose(fp);

  return;
}



void output_pressure(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.pressure.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = output_open(output_file);

  fprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.6e\n",E->NP[j][i]);
  }

  fclose(fp1);

  return;
}





