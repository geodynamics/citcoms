/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *=====================================================================
 *
 *                              CitcomS
 *                 ---------------------------------
 *
 *                              Authors:
 *           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 *           Clint Conrad, Michael Gurnis, and Eun-seo Choi
 *          (c) California Institute of Technology 1994-2005
 *
 *        By downloading and/or installing this software you have
 *       agreed to the CitcomS.py-LICENSE bundled with this software.
 *             Free for non-commercial academic research ONLY.
 *      This program is distributed WITHOUT ANY WARRANTY whatsoever.
 *
 *=====================================================================
 *
 *  Copyright June 2005, by the California Institute of Technology.
 *  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
 * 
 *  Any commercial use must be negotiated with the Office of Technology
 *  Transfer at the California Institute of Technology. This software
 *  may be subject to U.S. export control laws and regulations. By
 *  accepting this software, the user agrees to comply with all
 *  applicable U.S. export laws and regulations, including the
 *  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
 *  the Export Administration Regulations, 15 C.F.R. 730-744. User has
 *  the responsibility to obtain export licenses, or other export
 *  authority as may be required before exporting such information to
 *  foreign countries or providing access to foreign nationals.  In no
 *  event shall the California Institute of Technology be liable to any
 *  party for direct, indirect, special, incidental or consequential
 *  damages, including lost profits, arising out of the use of this
 *  software and its documentation, even if the California Institute of
 *  Technology has been advised of the possibility of such damage.
 * 
 *  The California Institute of Technology specifically disclaims any
 *  warranties, including the implied warranties or merchantability and
 *  fitness for a particular purpose. The software and documentation
 *  provided hereunder is on an "as is" basis, and the California
 *  Institute of Technology has no obligations to provide maintenance,
 *  support, updates, enhancements or modifications.
 *
 *=====================================================================
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

void read_input_files_for_timesteps(E,action,output)
    struct All_variables *E;
    int action, output;
{
    float find_age_in_MY();

    FILE *fp1, *fp2;
    float age, newage1, newage2;
    char output_file1[255],output_file2[255];
    float *VB1[4],*VB2[4], inputage1, inputage2;
    int nox,noz,noy,nnn,nox1,noz1,noy1,lev;
    int i,ii,ll,m,mm,j,k,n,nodeg,nodel,node,cap;
    int intage, pos_age;

    const int dims=E->mesh.nsd;

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

    for (m=1;m<=E->sphere.caps_per_proc;m++)  {
      cap = E->sphere.capid[m] - 1;  /* capid: 1-12 */

      switch (action) { /* set up files to open */

      case 1:  /* read velocity boundary conditions */
	sprintf(output_file1,"%s%0.0f.%d",E->control.velocity_boundary_file,newage1,cap);
	sprintf(output_file2,"%s%0.0f.%d",E->control.velocity_boundary_file,newage2,cap);
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

      case 2:  /* read ages for lithosphere tempperature boundary conditions */
	sprintf(output_file1,"%s%0.0f.%d",E->control.lith_age_file,newage1,cap);
	sprintf(output_file2,"%s%0.0f.%d",E->control.lith_age_file,newage2,cap);
	fp1=fopen(output_file1,"r");
	if (fp1 == NULL) {
          fprintf(E->fp,"(Problem_related #6) Cannot open %s\n",output_file1);
          exit(8);
	}
	if (pos_age) {
	  fp2=fopen(output_file2,"r");
	  if (fp2 == NULL) {
	    fprintf(E->fp,"(Problem_related #7) Cannot open %s\n",output_file2);
	    exit(8);
	  }
	}
	if((E->parallel.me==0) && (output==1))   {
	  fprintf(E->fp,"Age: Starting Age = %g, Elapsed time = %g, Current Age = %g\n",E->control.start_age,E->monitor.elapsed_time,age);
	  fprintf(E->fp,"Age: File1 = %s\n",output_file1);
	  if (pos_age)
	    fprintf(E->fp,"Age: File2 = %s\n",output_file2);
	  else
	    fprintf(E->fp,"Age: File2 = No file inputted (negative age)\n");
	}
	break;

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
	  VB1[1][i] *= E->data.timedir;
	  VB1[2][i] *= E->data.timedir;
	  if (pos_age) {
	    fscanf(fp2,"%f %f",&(VB2[1][i]),&(VB2[2][i]));
	    VB2[1][i] *= E->data.timedir;
	    VB2[2][i] *= E->data.timedir;
	  }
	  /* if( E->parallel.me ==0)
	     fprintf(stderr,"%d %f  %f  %f  %f\n",i,VB1[1][i],VB1[2][i],VB2[1][i],VB2[2][i]); */
	}
	fclose(fp1);
	if (pos_age) fclose(fp2);

	if(E->parallel.me_loc[3]==E->parallel.nprocz-1 )  {
          for(k=1;k<=noy1;k++)
	    for(i=1;i<=nox1;i++)    {
	      nodeg = E->lmesh.nxs+i-1 + (E->lmesh.nys+k-2)*nox;
	      nodel = (k-1)*nox1*noz1 + (i-1)*noz1+noz1;
	      if (pos_age) { /* positive ages - we must interpolate */
		E->sphere.cap[m].VB[1][nodel] = (VB1[1][nodeg] + (VB2[1][nodeg]-VB1[1][nodeg])/(newage2-newage1)*(age-newage1))*E->data.scalev;
		E->sphere.cap[m].VB[2][nodel] = (VB1[2][nodeg] + (VB2[2][nodeg]-VB1[2][nodeg])/(newage2-newage1)*(age-newage1))*E->data.scalev;
		E->sphere.cap[m].VB[3][nodel] = 0.0;
	      }
	      else { /* negative ages - don't do the interpolation */
		E->sphere.cap[m].VB[1][nodel] = VB1[1][nodeg];
		E->sphere.cap[m].VB[2][nodel] = VB1[2][nodeg];
		E->sphere.cap[m].VB[3][nodel] = 0.0;
	      }
	    }
	}   /* end of E->parallel.me_loc[3]==E->parallel.nproczl-1   */
	for(i=1;i<=dims;i++) {
          free ((void *) VB1[i]);
          free ((void *) VB2[i]);
	}
	break;

      case 2:  /* ages for lithosphere temperature boundary conditions */
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
	break;

      } /* end switch */
    } /* end for m */

    fflush(E->fp);

    return;
}
