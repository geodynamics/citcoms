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
  Calculate ages (MY) for opening input files -> material, ages, velocities
  Open these files, read in results, and average if necessary
=========================================================================*/

void regional_read_input_files_for_timesteps(E,action,output)
    struct All_variables *E;
    int action, output;
{
    float find_age_in_MY();

    FILE *fp1, *fp2;
    float age, newage1, newage2;
    char output_file1[255],output_file2[255];
    float *VB1[4],*VB2[4], inputage1, inputage2;
    int nox,noz,noy,nnn,nox1,noz1,noy1,lev;
    int i,ii,ll,mm,j,k,n,nodeg,nodel,node;
    int intage, pos_age;
    int nodea;
    int nn, el;

    const int dims=E->mesh.nsd;

    int elx,ely,elz,elg,emax;
    float *VIP1,*VIP2;
    int *LL1, *LL2;

    int llayer;
    int layers();

    if( E->parallel.me == 0) fprintf(stderr, "\nINSIDE regional_read_input_files_for_timesteps   action=%d\n",action);

    nox=E->mesh.nox;
    noy=E->mesh.noy;
    noz=E->mesh.noz;
    nox1=E->lmesh.nox;
    noz1=E->lmesh.noz;
    noy1=E->lmesh.noy;
    lev=E->mesh.levmax;

    elx=E->lmesh.elx;
    elz=E->lmesh.elz;
    ely=E->lmesh.ely;

    emax=E->mesh.elx*E->mesh.elz*E->mesh.ely;

    age=find_age_in_MY(E);

    if (age < 0.0) { /* age is negative -> use age=0 for input files */
      intage = 0;
      newage2 = newage1 = 0.0;
      pos_age = 0;
    }
    else {
      intage = age;
      newage1 = 1.0*intage;
      newage2 = 1.0*intage + 1.0;
      pos_age = 1;
    }

    switch (action) { /* set up files to open */

    case 1:  /* read velocity boundary conditions */
      sprintf(output_file1,"%s%0.0f",E->control.velocity_boundary_file,newage1);
      sprintf(output_file2,"%s%0.0f",E->control.velocity_boundary_file,newage2);
      fp1=fopen(output_file1,"r");
	if (fp1 == NULL) {
          fprintf(E->fp,"(Problem_related #4) Cannot open %s\n",output_file1);
          exit(8);
	}
      if (pos_age) {
        fp2=fopen(output_file2,"r");
	 if (fp2 == NULL) {
          fprintf(E->fp,"(Problem_related #5) Cannot open %s\n",output_file2);
          exit(8);
	 }
      }
      if((E->parallel.me==0) && (output==1))   {
         fprintf(E->fp,"Velocity: Starting Age = %g, Elapsed time = %g, Current Age = %g\n",E->control.start_age,E->monitor.elapsed_time,age);
         fprintf(E->fp,"Velocity: File1 = %s\n",output_file1);
        if (pos_age)
           fprintf(E->fp,"Velocity: File2 = %s\n",output_file2);
        else
           fprintf(E->fp,"Velocity: File2 = No file inputted (negative age)\n");
      }
      break;

      case 3:  /* read element materials */

        sprintf(output_file1,"%s%0.0f.0",E->control.mat_file,newage1);
        sprintf(output_file2,"%s%0.0f.0",E->control.mat_file,newage2);
        fp1=fopen(output_file1,"r");
        if (fp1 == NULL) {
          fprintf(E->fp,"(Problem_related #8) Cannot open %s\n",output_file1);
          exit(8);
        }
        if (pos_age) {
          fp2=fopen(output_file2,"r");
          if (fp2 == NULL) {
            fprintf(E->fp,"(Problem_related #9) Cannot open %s\n",output_file2);
            exit(8);
          }
        }
        if((E->parallel.me==0) && (output==1))   {
          fprintf(E->fp,"Mat: Starting Age = %g, Elapsed time = %g, Current Age = %g\n",E->control.start_age,E->monitor.elapsed_time,age);
          fprintf(E->fp,"Mat: File1 = %s\n",output_file1);
          if (pos_age)
            fprintf(E->fp,"Mat: File2 = %s\n",output_file2);
          else
            fprintf(E->fp,"Mat: File2 = No file inputted (negative age)\n");
        }

    } /* end switch */



    switch (action) { /* Read the contents of files and average */

    case 1:  /* velocity boundary conditions */
      nnn=nox*noy;
      for(i=1;i<=dims;i++)  {
        VB1[i]=(float*) malloc ((nnn+1)*sizeof(float));
        VB2[i]=(float*) malloc ((nnn+1)*sizeof(float));
      }
      for(i=1;i<=nnn;i++)   {
         fscanf(fp1,"%f %f",&(VB1[1][i]),&(VB1[2][i]));
         VB1[1][i]=E->data.timedir*VB1[1][i];
         VB1[2][i]=E->data.timedir*VB1[2][i];
         if (pos_age) {
             fscanf(fp2,"%f %f",&(VB2[1][i]),&(VB2[2][i]));
             VB2[1][i]=E->data.timedir*VB2[1][i];
             VB2[2][i]=E->data.timedir*VB2[2][i];
         }
      }
      fclose(fp1);
      if (pos_age) fclose(fp2);

      if(E->parallel.me_loc[3]==E->parallel.nprocz-1 )  {
          for(k=1;k<=noy1;k++)
             for(i=1;i<=nox1;i++)    {
                nodeg = E->lmesh.nxs+i-1 + (E->lmesh.nys+k-2)*nox;
                nodel = (k-1)*nox1*noz1 + (i-1)*noz1+noz1;
		if (pos_age) { /* positive ages - we must interpolate */
                    E->sphere.cap[1].VB[1][nodel] = (VB1[1][nodeg] + (VB2[1][nodeg]-VB1[1][nodeg])/(newage2-newage1)*(age-newage1))*E->data.scalev;
                    E->sphere.cap[1].VB[2][nodel] = (VB1[2][nodeg] + (VB2[2][nodeg]-VB1[2][nodeg])/(newage2-newage1)*(age-newage1))*E->data.scalev;
                    E->sphere.cap[1].VB[3][nodel] = 0.0;
		}
		else { /* negative ages - don't do the interpolation */
                    E->sphere.cap[1].VB[1][nodel] = VB1[1][nodeg]*E->data.scalev;
                    E->sphere.cap[1].VB[2][nodel] = VB1[2][nodeg]*E->data.scalev;
                    E->sphere.cap[1].VB[3][nodel] = 0.0;
		}
             }
      }   /* end of E->parallel.me_loc[3]==E->parallel.nprocz-1   */
      for(i=1;i<=dims;i++) {
          free ((void *) VB1[i]);
          free ((void *) VB2[i]);
      }
      break;

      case 3:  /* read element materials */

        VIP1 = (float*) malloc ((emax+1)*sizeof(float));
        VIP2 = (float*) malloc ((emax+1)*sizeof(float));
        LL1 = (int*) malloc ((emax+1)*sizeof(int));
        LL2 = (int*) malloc ((emax+1)*sizeof(int));

        /* probably can be safely removed */
          for (el=1; el<=elx*ely*elz; el++)  {
            nodea = E->ien[1][el].node[2];
            llayer = layers(E,1,nodea);
            if (llayer)  { /* for layers:1-lithosphere,2-upper, 3-trans, and 4-lower mantle */
              E->mat[1][el] = llayer;
            }
          }
          for(i=1;i<=emax;i++)  {
               fscanf(fp1,"%d %d %f", &nn,&(LL1[i]),&(VIP1[i]));
               fscanf(fp2,"%d %d %f", &nn,&(LL2[i]),&(VIP2[i]));
          }

          fclose(fp1);
          fclose(fp2);

          for (k=1;k<=ely;k++)   {
            for (i=1;i<=elx;i++)   {
              for (j=1;j<=elz;j++)  {
                el = j + (i-1)*E->lmesh.elz + (k-1)*E->lmesh.elz*E->lmesh.elx;
                elg = E->lmesh.ezs+j + (E->lmesh.exs+i-1)*E->mesh.elz + (E->lmesh.eys+k-1)*E->mesh.elz*E->mesh.elx;

                E->VIP[1][el] = VIP1[elg]+(VIP2[elg]-VIP1[elg])/(newage2-newage1)*(age-newage1);
                E->mat[1][el] = LL1[elg];

              }     /* end for j  */
            }     /*  end for i */
          }     /*  end for k  */

         free ((void *) VIP1);
         free ((void *) VIP2);
         free ((void *) LL1);
         free ((void *) LL2);

    } /* end switch */

   return;
}
