/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *
 * CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 * Clint Conrad, Michael Gurnis, and Eun-seo Choi.
 * Copyright (C) 1994-2005, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *</LicenseText>
 * 
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

/*=======================================================================
  read velocity vectors at the top surface from files
=========================================================================*/

void read_velocity_boundary_from_file(E)
     struct All_variables *E;
{
    (E->solver.read_input_files_for_timesteps)(E,1,1); /* read velocity(1) and output(1) */
    return;
}

/*=======================================================================
  construct material array
=========================================================================*/

void read_mat_from_file(E)
     struct All_variables *E;
{
  float find_age_in_MY();

  int nn,m,i,j,k,kk,el,lev,els;
  int elx,ely,elz,e,elg,emax,gmax;
  float *VIP1,*VIP2;

  float age1,newage1,newage2;
  int nodea,nage;

  int llayer;
  int layers();
  FILE *fp,*fp1,*fp2,*fp3,*fp4;
  char output_file[255];

  const int dims=E->mesh.nsd,dofs=E->mesh.dof;
  const int ends=enodes[dims];

  elx=E->lmesh.elx;
  elz=E->lmesh.elz;
  ely=E->lmesh.ely;

  emax=E->mesh.elx*E->mesh.elz*E->mesh.ely;
  gmax=E->mesh.elx*E->mesh.ely;

  VIP1 = (float*) malloc ((gmax+1)*sizeof(float));
  VIP2 = (float*) malloc ((gmax+1)*sizeof(float));



  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (el=1; el<=elx*ely*elz; el++)  {
      nodea = E->ien[m][el].node[2];
      llayer = layers(E,m,nodea);
      if (llayer)  { /* for layers:1-lithosphere,2-upper, 3-trans, and 4-lower mantle */
        E->mat[m][el] = llayer;
      }
    }


  if(E->control.mat_control==1)  {

    age1 = find_age_in_MY(E);

    nage=age1/1.;
    newage1=1.*nage;

    sprintf(output_file,"%s%0.0f",E->control.mat_file,newage1);
    if(E->parallel.me==0)
      fprintf(E->fp,"%s %f %s\n","newage1",newage1,output_file);
    fp1=fopen(output_file,"r");
    if (fp1 == NULL) {
      fprintf(E->fp,"(Problem_related #1) Cannot open %s\n",output_file);
      exit(8);
    }

    newage2=newage1+1.;
    sprintf(output_file,"%s%0.0f",E->control.mat_file,newage2);
    if(E->parallel.me==0)
      fprintf(E->fp,"%s %f %s\n","newage2",newage2,output_file);
    fp2=fopen(output_file,"r");
    if (fp2 == NULL) {
      fprintf(E->fp,"(Problem_related #2) Cannot open %s\n",output_file);
      exit(8);
    }

    for(i=1;i<=gmax;i++)  {
      fscanf(fp1,"%d %f", &nn,&(VIP1[i]));
      fscanf(fp2,"%d %f", &nn,&(VIP2[i]));
    }

    fclose(fp1);
    fclose(fp2);


    for (m=1;m<=E->sphere.caps_per_proc;m++)
      for (k=1;k<=ely;k++)
	for (i=1;i<=elx;i++)   {
	  elg = E->lmesh.exs+i + (E->lmesh.eys+k-1)*E->mesh.elx;

	  for (j=1;j<=elz;j++)  {
	    el = j + (i-1)*E->lmesh.elz + (k-1)*E->lmesh.elz*E->lmesh.elx;

	    if(E->sx[m][3][E->ien[m][el].node[2]]>=E->sphere.ro-E->viscosity.zlith)
	      E->VIP[m][el] = VIP1[elg]+(VIP2[elg]-VIP1[elg])/(newage2-newage1)*(age1-newage1);

	  }   /* end for j  */

	}     /*  end for m  */

  }     /* end for E->control.mat==1  */


  /* mat output moved to Output.c */

  free ((void *) VIP1);
  free ((void *) VIP2);

  return;

}


/*=======================================================================
  Open restart file to get initial elapsed time, or calculate the right value
=========================================================================*/

void get_initial_elapsed_time(E)
  struct All_variables *E;
{
    FILE *fp;
    int ll, mm;
    char output_file[255],input_s[1000];

    E->monitor.elapsed_time = 0.0;
    if ((E->control.restart || E->control.post_p))    {
	sprintf(output_file, "%s.velo.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);
        fp=fopen(output_file,"r");
	if (fp == NULL) {
          fprintf(E->fp,"(Problem_related #8) Cannot open %s\n",output_file);
          exit(8);
	}
        fgets(input_s,1000,fp);
        sscanf(input_s,"%d %d %f",&ll,&mm,&E->monitor.elapsed_time);
     fclose(fp);
      } /* end control.restart */

   return;
}

/*=======================================================================
  Sets the elapsed time to zero, if desired.
=========================================================================*/

void set_elapsed_time(E)
  struct All_variables *E;
{

    if (E->control.zero_elapsed_time) /* set elapsed_time to zero */
	E->monitor.elapsed_time = 0.0;

   return;
}

/*=======================================================================
  Resets the age at which to start time (startage) to the end of the previous
  run, if desired.
=========================================================================*/

void set_starting_age(E)
  struct All_variables *E;
{
/* remember start_age is in MY */
    if (E->control.reset_startage)
	E->control.start_age = E->monitor.elapsed_time*E->data.scalet;

   return;
}


/*=======================================================================
  Returns age at which to open an input file (velocity, material, age)
  NOTE: Remember that ages are positive, but going forward in time means
  making ages SMALLER!
=========================================================================*/

  float find_age_in_MY(E)

  struct All_variables *E;
{
   float age_in_MY, e_4;


   e_4=1.e-4;

   if (E->data.timedir >= 0) { /* forward convection */
      age_in_MY = E->control.start_age - E->monitor.elapsed_time*E->data.scalet;
   }
   else { /* backward convection */
      age_in_MY = E->control.start_age + E->monitor.elapsed_time*E->data.scalet;
   }

      if (((age_in_MY+e_4) < 0.0) && (E->monitor.solution_cycles <= 1)) {
        if (E->parallel.me == 0) fprintf(stderr,"Age = %g Ma, Initial age should not be negative!\n",age_in_MY);
	exit(11);
      }

   return(age_in_MY);
}
