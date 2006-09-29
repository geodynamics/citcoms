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


#include "global_defs.h"


void regional_lith_age_read_files(struct All_variables *E, int output)
{
  float find_age_in_MY();

  FILE *fp1, *fp2;
  float age, newage1, newage2;
  char output_file1[255],output_file2[255];
  float inputage1, inputage2;
  int nox,noz,noy,nox1,noz1,noy1,lev;
  int i,j,node;
  int intage, pos_age;

  nox=E->mesh.nox;
  noy=E->mesh.noy;
  noz=E->mesh.noz;
  nox1=E->lmesh.nox;
  noz1=E->lmesh.noz;
  noy1=E->lmesh.noy;
  lev=E->mesh.levmax;

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

  /* read ages for lithosphere tempperature boundary conditions */
  sprintf(output_file1,"%s%0.0f",E->control.lith_age_file,newage1);
  sprintf(output_file2,"%s%0.0f",E->control.lith_age_file,newage2);
  fp1=fopen(output_file1,"r");
  if (fp1 == NULL) {
    fprintf(E->fp,"(Problem_related #6) Cannot open %s\n",output_file1);
    parallel_process_termination();
  }
  if (pos_age) {
    fp2=fopen(output_file2,"r");
    if (fp2 == NULL) {
      fprintf(E->fp,"(Problem_related #7) Cannot open %s\n",output_file2);
    parallel_process_termination();
    }
  }
  if((E->parallel.me==0) && output) {
    fprintf(E->fp,"Lith_Age: Starting Age = %g, Elapsed time = %g, Current Age = %g\n",E->control.start_age,E->monitor.elapsed_time,age);
    fprintf(E->fp,"Lith_Age: File1 = %s\n",output_file1);
    if (pos_age)
      fprintf(E->fp,"Lith_Age: File2 = %s\n",output_file2);
    else
      fprintf(E->fp,"Lith_Age: File2 = No file inputted (negative age)\n");
  }

  for(i=1;i<=noy;i++)
    for(j=1;j<=nox;j++) {
      node=j+(i-1)*nox;
      fscanf(fp1,"%f",&inputage1);
      if (pos_age) { /* positive ages - we must interpolate */
	fscanf(fp2,"%f",&inputage2);
	E->age_t[node] = (inputage1 + (inputage2-inputage1)/(newage2-newage1)*(age-newage1))/E->data.scalet;
      }
      else { /* negative ages - don't do the interpolation */
	E->age_t[node] = inputage1;
      }
    }
  fclose(fp1);
  if (pos_age) fclose(fp2);

  return;
}


/* End of file */
