/* Routine to process the output of the finite element cycles
   and to turn them into a coherent suite  files  */


#include <fcntl.h>
#include <math.h>
#include <stdlib.h>             /* for "system" command */
#ifndef __sunos__               /* string manipulations */
#include <strings.h>
#else
#include <string.h>
#endif

#include "element_definitions.h"
#include "global_defs.h"
#include "output.h"

FILE* output_init(char *output_file)
{
  FILE *fp1;

  if (*output_file)
    fp1 = fopen(output_file,"w");
  else
    fp1 = stderr;

  return fp1;
}


void output_close(FILE* fp1)
{
  fclose(fp1);
  return;
}


void output_coord(struct All_variables *E, FILE *fp1)
{
  int i, j;

  for(j=1;j<=E->sphere.caps_per_proc;j++)     {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.3e %.3e %.3e\n",E->sx[j][1][i],E->sx[j][2][i],E->sx[j][3][i]);
  }
  return;
}


void output_visc_prepare(struct All_variables *E, float **VE)
{
  void get_ele_visc ();
  void visc_from_ele_to_gint();
  void visc_from_gint_to_nodes();

  float *EV, *VN[NCS];
  const int lev = E->mesh.levmax;
  const int nsd = E->mesh.nsd;
  const int vpts = vpoints[nsd];
  int i, m;


  // Here is a bug in the original code. EV is not allocated for each
  // E->sphere.caps_per_proc. Later, when elemental viscosity is written
  // to it (in get_ele_visc()), viscosity in high cap number will overwrite
  // that in a lower cap number.
  //
  // Since current CitcomS only support 1 cap per processor, this bug won't
  // manifest itself. So, I will leave it here.
  // by Tan2 5/22/2003
  int size2 = (E->lmesh.nel+1)*sizeof(float);
  EV = (float *) malloc (size2);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    VN[m]=(float *)malloc((1+E->lmesh.nel*vpts)*sizeof(float));
  }

  get_ele_visc(E,EV,1);

  for(i=1;i<=E->lmesh.nel;i++)
    VE[1][i]=EV[i];

  visc_from_ele_to_gint(E, VE, VN, lev);
  visc_from_gint_to_nodes(E, VN, VE, lev);

  free((void *) EV);
  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    free((void *) VN[m]);
  }

  return;
}


void output_visc(struct All_variables *E, FILE *fp1, float **VE)
{
  int i, j, m;

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.3e\n",VE[1][i]);
  }

  return;
}


void output_velo_header(struct All_variables *E, FILE *fp1, int file_number)
{
  fprintf(fp1,"%d %d %.5e\n",file_number,E->lmesh.nno,E->monitor.elapsed_time);
  return;
}


void output_velo(struct All_variables *E, FILE *fp1)
{
  int i, j;

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.6e %.6e %.6e %.6e\n",E->sphere.cap[j].V[1][i],E->sphere.cap[j].V[2][i],E->sphere.cap[j].V[3][i],E->T[j][i]);
  }

  return;
}



/*********************************************************************/


void output_velo_related(E,file_number)
  struct All_variables *E;
  int file_number;
{
  int el,els,i,j,k,m,node,fd;
  int s,nox,noz,noy,size1,size2,size3;

  char output_file[255];
  FILE *fp1,*fp2;
  float *VE[NCS];

  void parallel_process_termination();

  sprintf(output_file,"%s.coord.%d",E->control.data_file,E->parallel.me);
  fp1 = output_init(output_file);
  if (!fp1) {
    fprintf(E->fp,"Cannot open file '%s'\n",output_file);
    parallel_process_termination();
  }
  output_coord(E, fp1);
  output_close(fp1);

  sprintf(output_file,"%s.visc.%d.%d",E->control.data_file,E->parallel.me,file_number);
  fp1 = output_init(output_file);
  if (!fp1) {
    fprintf(E->fp,"Cannot open file '%s'\n",output_file);
    parallel_process_termination();
  }

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    VE[m]=(float *)malloc((1+E->lmesh.nno)*sizeof(float));
  }
  output_visc_prepare(E, VE);
  output_visc(E, fp1, VE);
  output_close(fp1);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    free((void*) VE[m]);
  }

  sprintf(output_file,"%s.velo.%d.%d",E->control.data_file,E->parallel.me,file_number);
  fp1 = output_init(output_file);
  if (!fp1) {
    fprintf(E->fp,"Cannot open file '%s'\n",output_file);
    parallel_process_termination();
  }
  output_velo_header(E, fp1, file_number);
  output_velo(E, fp1);
  output_close(fp1);


  if (E->parallel.me_locl[3]==E->parallel.nproczl-1)      {
    sprintf(output_file,"%s.surf.%d.%d",E->control.data_file,E->parallel.me,file_number);
    fp2 = output_init(output_file);
    if (!fp2) {
      fprintf(E->fp,"Cannot open file '%s'\n",output_file);
      parallel_process_termination();
    }
    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)   {
	s = i*E->lmesh.noz;
        fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpg[j][i],E->slice.shflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
	}
      }
    output_close(fp2);

    }

  if (E->parallel.me_locl[3]==0)      {
    sprintf(output_file,"%s.botm.%d.%d",E->control.data_file,E->parallel.me,file_number);
    fp2 = output_init(output_file);
    if (!fp2) {
      fprintf(E->fp,"Cannot open file '%s'\n",output_file);
      parallel_process_termination();
    }
    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)  {
	s = (i-1)*E->lmesh.noz + 1;
        fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpgb[j][i],E->slice.bhflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
	}
      }
    output_close(fp2);
    }

  /* remove horizontal average output   by Tan2 Mar. 1 2002  */
/*    if (E->parallel.me<E->parallel.nproczl)  { */
/*      sprintf(output_file,"%s.ave_r.%d.%d",E->control.data_file,E->parallel.me,file_number); */
/*      fp2 = output_open(output_file); */
/*  	if (fp2 == NULL) { */
/*            fprintf(E->fp,"(Output.c #6) Cannot open %s\n",output_file); */
/*            exit(8); */
/*  	} */
/*      for(j=1;j<=E->lmesh.noz;j++)  { */
/*          fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->sx[1][3][j],E->Have.T[j],E->Have.V[1][j],E->Have.V[2][j]); */
/*  	} */
/*      output_close(fp2); */
/*      } */

  return;
  }

/* ====================================================================== */

void output_temp(E,file_number)
  struct All_variables *E;
  int file_number;
{
  int m,nno,i,j,fd;
  char output_file[255];
  void parallel_process_sync();

  return;
}

