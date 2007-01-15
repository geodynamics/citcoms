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
/*

  Tracer_advection.c

      A program which initiates the distribution of tracers
      and advects those tracers in a time evolving velocity field.
      Called and used from the CitCOM finite element code.
      Written 2/96 M. Gurnis for Citcom in cartesian geometry
      Modified by Lijie in 1998 and by Vlad and Eh in 2005 for CitcomS

*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <mpi.h>
#include <math.h>
#include <sys/types.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#include "element_definitions.h"
#include "global_defs.h"


void regional_tracer_setup(E)
     struct All_variables *E;
{
        int i,j,k,node;
	int m,ntr;
        int n_x,n_y,n_z;
        int node1,node2,node3,node4,node5,node6,node7,node8;
        int local_element;
        float THETA_LOC_ELEM,FI_LOC_ELEM,R_LOC_ELEM;
	float idummy,xdummy,ydummy,zdummy;
	FILE *fp;
        int nox,noy,noz;
        char output_file[255];
	MPI_Comm world;
	MPI_Status status;

        n_x=0;
        n_y=0;
        n_z=0;

        nox=E->lmesh.nox;
        noy=E->lmesh.noy;
        noz=E->lmesh.noz;

        sprintf(output_file,"%s",E->control.tracer_file);
	fp=fopen(output_file,"r");
	if (fp == NULL) {
          fprintf(E->fp,"(Tracer_advection #1) Cannot open %s\n", output_file);
          exit(8);
	}
	fscanf(fp,"%d",&(E->Tracer.NUM_TRACERS));

        E->Tracer.tracer_x=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
        E->Tracer.tracer_y=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
        E->Tracer.tracer_z=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
        E->Tracer.itcolor=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));

        E->Tracer.x_space=(float*) malloc((nox+1)*sizeof(float));
        E->Tracer.y_space=(float*) malloc((noy+1)*sizeof(float));
        E->Tracer.z_space=(float*) malloc((noz+1)*sizeof(float));

        E->Tracer.LOCAL_ELEMENT=(int*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(int));

                /* for rheology stuff */
/*         E->Tracer.THETA_LOC_ELEM=(float*) malloc((E->lmesh.nox*E->lmesh.noy*E->lmesh.noz+1)*sizeof(float)); */
/*         E->Tracer.FI_LOC_ELEM=(float*) malloc((E->lmesh.nox*E->lmesh.noy*E->lmesh.noz+1)*sizeof(float)); */
/*         E->Tracer.R_LOC_ELEM=(float*) malloc((E->lmesh.nox*E->lmesh.noy*E->lmesh.noz+1)*sizeof(float)); */

/*         E->Tracer.THETA_LOC_ELEM_T=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float)); */
/*         E->Tracer.FI_LOC_ELEM_T=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float)); */
/*         E->Tracer.R_LOC_ELEM_T=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float)); */



        /***comment by Vlad 03/15/2005
	    each processor holds its own number of tracers
        ***/

	ntr=1;
	for(i=1;i<=E->Tracer.NUM_TRACERS;i++) {
	  fscanf(fp,"%f %f %f %f", &idummy, &xdummy, &ydummy, &zdummy);
	  if(xdummy >= E->sx[1][1][1] && xdummy <= E->sx[1][1][nox*noy*noz]) {
	    if(ydummy >= E->sx[1][2][1] && ydummy <= E->sx[1][2][nox*noy*noz])  {
	      if(zdummy >= E->sx[1][3][1] && zdummy <= E->sx[1][3][nox*noy*noz])  {
		E->Tracer.itcolor[ntr]=idummy;
		E->Tracer.tracer_x[ntr]=xdummy;
		E->Tracer.tracer_y[ntr]=ydummy;
		E->Tracer.tracer_z[ntr]=zdummy;
		ntr++;
    	      }
	    }
	  }
	}

	/***comment by Vlad 3/30/2005
	    E->Tracer.LOCAL_NUM_TRACERS is the initial number
	    of tracers in each processor
	 ***/

	E->Tracer.LOCAL_NUM_TRACERS=ntr-1;



	/***comment by Vlad 1/26/2005
	    reading the local mesh coordinate
	***/



        for(m=1;m<=E->sphere.caps_per_proc;m++)  {
	  for(i=1;i<=nox;i++)
	    {
	      j=1;
	      k=1;
	      node=k+(i-1)*noz+(j-1)*nox*noz;
	      E->Tracer.x_space[i]=E->sx[m][1][node];
	    }

	  for(j=1;j<=noy;j++)
	    {
	      i=1;
	      k=1;
	      node=k+(i-1)*noz+(j-1)*nox*noz;
	      E->Tracer.y_space[j]=E->sx[m][2][node];
	    }

	  for(k=1;k<=noz;k++)
	    {
		    i=1;
		    j=1;
		    node=k+(i-1)*noz+(j-1)*nox*noz;
		    E->Tracer.z_space[k]=E->sx[m][3][node];
		  }

	}   /* end of m  */

        /***comment by Vlad 04/20/2006
            reading the local element eight nodes coordinate,
           then computing the mean element coordinates
        ***/



        for(m=1;m<=E->sphere.caps_per_proc;m++)  {
          for(j=1;j<=(noy-1);j++) {
           for(i=1;i<=(nox-1);i++) {
             for(k=1;k<=(noz-1);k++) {
               n_x=i;
               n_y=j;
               n_z=k;

               node1 = n_z + (n_x-1)*noz + (n_y-1)*noz*nox;
                node2 = n_z + n_x*noz + (n_y-1)*noz*nox;
                node3 = n_z+1  + (n_x-1)*noz + (n_y-1)*noz*nox;
                node4 = n_z+1  + n_x*noz + (n_y-1)*noz*nox;
                node5 = n_z + (n_x-1)*noz + n_y*noz*nox;
                node6 = n_z + n_x*noz + n_y*noz*nox;
                node7 = n_z+1 + (n_x-1)*noz + n_y*noz*nox;
                node8 = n_z+1 + n_x*noz + n_y*noz*nox;

                /* for rheology stuff */
/*                local_element=node1-(n_x-1)-(n_y-1)*(nox+noz-1); */

/*                E->Tracer.THETA_LOC_ELEM[local_element]=(E->sx[m][1][node1]+E->sx[m][1][node2])/2; */
/*                E->Tracer.FI_LOC_ELEM[local_element]=(E->sx[m][2][node1]+E->sx[m][2][node5])/2; */
/*                E->Tracer.R_LOC_ELEM[local_element]=(E->sx[m][3][node1]+E->sx[m][3][node3])/2; */


               //if(E->parallel.me == 55) fprintf(stderr,"%d %s %d %d %s %s %f %s %f %s %f\n", E->parallel.me,"The local element no.:", local_element, i*j*k,"has", "stinga=", E->sx[m][2][node1], "y=", E->Tracer.FI_LOC_ELEM[local_element], "dreapta=", E->sx[m][2][node5]);

               }
              }
           }
          }   /* end of m  */


        // if(E->parallel.me == 55) fprintf(stderr,"%d %d %d %d %d\n", E->parallel.me, nox, noy, noz, E->lmesh.nel);



        return;

}

void regional_tracer_advection(E)
     struct All_variables *E;
{
      int i,j,k,l,m,n,o,p;
      int n_x,n_y,n_z;
      int node1,node2,node3,node4,node5,node6,node7,node8;
      int nno,nox,noy,noz;
      int iteration;
      float x_tmp, y_tmp, z_tmp;
      float volume, tr_dx, tr_dy, tr_dz, dx, dy, dz;
      float w1,w2,w3,w4,w5,w6,w7,w8;
      float tr_v[NCS][4];
      MPI_Comm world;
      MPI_Status status[4];
      MPI_Status status_count;
      MPI_Request request[4];
      float xmin,xmax,ymin,ymax,zmin,zmax;

      float x_global_min,x_global_max,y_global_min,y_global_max,z_global_min,z_global_max;


      float *tr_color_1,*tr_x_1,*tr_y_1,*tr_z_1;
      float *tr_color_new[13],*tr_x_new[13],*tr_y_new[13],*tr_z_new[13];
      int *Left_proc_list,*Right_proc_list;
      int *jump_new,*count_new;
      int *jj;

      int proc;
      int Previous_num_tracers,Current_num_tracers;
      int locx,locy,locz;
      int left,right,up,down,back,front;
      int temp_tracers;


      world=E->parallel.world;


      nox=E->lmesh.nox;
      noy=E->lmesh.noy;
      noz=E->lmesh.noz;
      nno=nox*noy*noz;

      Left_proc_list=(int*) malloc(6*sizeof(int));
      Right_proc_list=(int*) malloc(6*sizeof(int));
      jump_new=(int*) malloc(6*sizeof(int));
      count_new=(int*) malloc(6*sizeof(int));
      jj=(int*) malloc(6*sizeof(int));

      tr_x_1=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
      tr_y_1=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
      tr_z_1=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
      tr_color_1=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));

      for(i=0;i<=11;i++){
	tr_color_new[i]=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
	tr_x_new[i]=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
	tr_y_new[i]=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
	tr_z_new[i]=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
      }


      /*** comment by Vlad 3/25/2005
	   This part of code gets the bounding box of the local mesh.
      ***/

      xmin=E->Tracer.x_space[1];
      xmax=E->Tracer.x_space[nox];
      ymin=E->Tracer.y_space[1];
      ymax=E->Tracer.y_space[noy];
      zmin=E->Tracer.z_space[1];
      zmax=E->Tracer.z_space[noz];

      /*fprintf(stderr,"%d %d\n", E->parallel.nprocx, E->parallel.loc2proc_map[0][0][1][0]);*/

      /*** comment by Tan2 1/25/2005
           Copy the velocity array.
      ***/


      for(m=1;m<=E->sphere.caps_per_proc;m++)   {
	for(i=1;i<=nno;i++)
	  for(j=1;j<=3;j++)   {
	    E->GV[m][j][i]=E->sphere.cap[m].V[j][i];
	  }
      }

      /*** comment by vlad 03/17/2005
	     advecting tracers in each processor
      ***/


      for(n=1;n<=E->Tracer.LOCAL_NUM_TRACERS;n++) {

	n_x=0;
	n_y=0;
	n_z=0;

	/*printf("%s %d %s %d %s %f\n","La pasul",E->monitor.solution_cycles, "Procesorul", E->parallel.me, "advecteaza tracerul",E->Tracer.tracer_y[n]);

	//printf("%d %d %d %f %f %f %f\n", E->monitor.solution_cycles, E->parallel.me,E->Tracer.LOCAL_NUM_TRACERS,E->Tracer.itcolor[n],E->Tracer.tracer_x[n],E->Tracer.tracer_y[n],E->Tracer.tracer_z[n]);
	//printf("%d %d %d %f %f %f %f\n", E->monitor.solution_cycles, E->parallel.me,E->Tracer.LOCAL_NUM_TRACERS,tr_color[n],tr_x[n],tr_y[n],tr_z[n]);
        */

	/*  mid point method uses 2 iterations */

        x_tmp=E->Tracer.tracer_x[n];
        y_tmp=E->Tracer.tracer_y[n];
        z_tmp=E->Tracer.tracer_z[n];

	for(iteration=1;iteration<=2;iteration++)
	  {


	    /*** comment by Tan2 1/25/2005
		 Find the element that contains the tracer.

		 nodex      n_x                 n_x+1
		 |           *                    |
		 <----------->
		    tr_dx

		 <-------------------------------->
		                 dx
	    ***/

	    for(i=1;i<nox;i++) {
	      if(x_tmp >= E->Tracer.x_space[i] && x_tmp <= E->Tracer.x_space[i+1]) {
		tr_dx=x_tmp-E->Tracer.x_space[i];
		dx=E->Tracer.x_space[i+1]-E->Tracer.x_space[i];
		n_x=i;
/*                 E->Tracer.THETA_LOC_ELEM_T[n]=(E->Tracer.x_space[i+1]+E->Tracer.x_space[i])/2; */
	      }
	    }

	    for(j=1;j<noy;j++) {
	      if(y_tmp >= E->Tracer.y_space[j] && y_tmp <= E->Tracer.y_space[j+1]) {
		tr_dy=y_tmp-E->Tracer.y_space[j];
		dy=E->Tracer.y_space[j+1]-E->Tracer.y_space[j];
	        n_y=j;
/*                 E->Tracer.FI_LOC_ELEM_T[n]=(E->Tracer.y_space[j+1]+E->Tracer.y_space[j])/2; */

	      }
	    }

	    for(k=1;k<noz;k++) {
	      if(z_tmp >= E->Tracer.z_space[k] && z_tmp <= E->Tracer.z_space[k+1]) {
		tr_dz=z_tmp-E->Tracer.z_space[k];
		dz=E->Tracer.z_space[k+1]-E->Tracer.z_space[k];
	        n_z=k;
/*                 E->Tracer.R_LOC_ELEM_T[n]=(E->Tracer.z_space[k+1]+E->Tracer.z_space[k])/2; */

	      }
	    }

            //fprintf(stderr,"%d %f %f %f\n",n,E->Tracer.THETA_LOC_ELEM_T[n],E->Tracer.FI_LOC_ELEM_T[n],E->Tracer.R_LOC_ELEM_T[n]);


	    /*** comment by Tan2 1/25/2005
		 Calculate shape functions from tr_dx, tr_dy, tr_dz
		 This assumes linear element
	    ***/


	    /* compute volumetic weighting functions */
	    w1=tr_dx*tr_dz*tr_dy;
	    w2=(dx-tr_dx)*tr_dz*tr_dy;
	    w3=tr_dx*(dz-tr_dz)*tr_dy;
	    w4=(dx-tr_dx)*(dz-tr_dz)*tr_dy;
	    w5=tr_dx*tr_dz*(dy-tr_dy);
	    w6=(dx-tr_dx)*tr_dz*(dy-tr_dy);
	    w7=tr_dx*(dz-tr_dz)*(dy-tr_dy);
	    w8=(dx-tr_dx)*(dz-tr_dz)*(dy-tr_dy);


	    volume=dx*dz*dy;


	    /*** comment by Tan2 1/25/2005
		 Calculate the 8 node numbers of current element
	     ***/

	    node1 = n_z + (n_x-1)*noz + (n_y-1)*noz*nox;
	    node2 = n_z + n_x*noz + (n_y-1)*noz*nox;
	    node3 = n_z+1  + (n_x-1)*noz + (n_y-1)*noz*nox;
	    node4 = n_z+1  + n_x*noz + (n_y-1)*noz*nox;
	    node5 = n_z + (n_x-1)*noz + n_y*noz*nox;
	    node6 = n_z + n_x*noz + n_y*noz*nox;
	    node7 = n_z+1 + (n_x-1)*noz + n_y*noz*nox;
	    node8 = n_z+1 + n_x*noz + n_y*noz*nox;


	    /*	printf("%d %d %d %d %d %d %d %d %d\n",E->parallel.me, node1,node2,node3,node4,node5,node6,node7,node8);
	    //printf("%d %f %f %f %f %f %f %f %f\n", E->parallel.me, E->GV[1][2][node1], E->GV[1][2][node2], E->GV[1][2][node3], E->GV[1][2][node4], E->GV[1][2][node5], E->GV[1][2][node6], E->GV[1][2][node7], E->GV[1][2][node8]);
            */

	    /*** comment by Tan2 1/25/2005
		 Interpolate the velocity on the tracer position
	    ***/

            for(m=1;m<=E->sphere.caps_per_proc;m++)   {
              for(j=1;j<=3;j++)   {
		tr_v[m][j]=w8*E->GV[m][j][node1]
		  +w7*E->GV[m][j][node2]
		  +w6*E->GV[m][j][node3]
		  +w5*E->GV[m][j][node4]
		  +w4*E->GV[m][j][node5]
		  +w3*E->GV[m][j][node6]
		  +w2*E->GV[m][j][node7]
		  +w1*E->GV[m][j][node8];
		tr_v[m][j]=tr_v[m][j]/volume;

	      }



              E->Tracer.LOCAL_ELEMENT[n]=node1-(n_x-1)-(n_y-1)*(nox+noz-1);




             //fprintf(stderr,"%s %d %s %d %f %f %f %f\n", "The tracer no:", n,"is in element no:", E->Tracer.LOCAL_ELEMENT[n], E->Tracer.y_space[n_y], E->Tracer.tracer_y[n], E->Tracer.FI_LOC_ELEM[515], E->Tracer.y_space[n_y+1]);





	      /*** comment by Tan2 1/25/2005
		   advect tracer using mid-point method (2nd order accuracy)
	       ***/

	      /* mid point method */

	      if(iteration == 1) {
		x_tmp = x_tmp + (E->advection.timestep/2.0)*tr_v[m][1]/E->Tracer.z_space[n_z];
		y_tmp = y_tmp + (E->advection.timestep/2.0)*tr_v[m][2]/(E->Tracer.z_space[n_z]*sin(E->Tracer.x_space[n_x]));
		z_tmp = z_tmp + (E->advection.timestep/2.0)*tr_v[m][3];
	      }
	      if( iteration == 2) {
		E->Tracer.tracer_x[n] += E->advection.timestep*tr_v[m][1]/E->Tracer.z_space[n_z];
		E->Tracer.tracer_y[n] += E->advection.timestep*tr_v[m][2]/(E->Tracer.z_space[n_z]*sin(E->Tracer.x_space[n_x]));
		E->Tracer.tracer_z[n] += E->advection.timestep*tr_v[m][3];


               //fprintf(stderr,"%d %d %f %f %f %f %f %f\n", E->parallel.me, E->monitor.solution_cycles, E->Tracer.tracer_x[n],E->Tracer.tracer_y[n],E->Tracer.tracer_z[n], tr_v[m][1],tr_v[m][2],tr_v[m][3]);


	  }



	    }   /*  end of m  */


	  } /* end of iteration loop */


      /*** Comment by Vlad 12/15/2005
           Put the tracers back in the box if they go out
      ***/

       /*** Comment by Vlad 12/15/2005
           get the bounding box of the global mesh
      ***/

        x_global_min = E->control.theta_min;
        x_global_max = E->control.theta_max;
        y_global_min = E->control.fi_min;
        y_global_max = E->control.fi_max;
        z_global_min = E->sphere.ri;
        z_global_max = E->sphere.ro;

       //printf("%f %f %f %f %f %f\n", E->sphere.cap[1].theta[1],E->sphere.cap[1].theta[3],E->sphere.cap[1].fi[1],E->sphere.cap[1].fi[3],E->sphere.ri,E->sphere.ro);

       if(E->Tracer.tracer_x[n] > x_global_max)
           E->Tracer.tracer_x[n] = x_global_max;
       if(E->Tracer.tracer_x[n] < x_global_min)
           E->Tracer.tracer_x[n] = x_global_min;
       if(E->Tracer.tracer_y[n] > y_global_max)
           E->Tracer.tracer_y[n] = y_global_max;
       if(E->Tracer.tracer_y[n] < y_global_min)
           E->Tracer.tracer_y[n] = y_global_min;
       if(E->Tracer.tracer_z[n] > z_global_max)
           E->Tracer.tracer_z[n] = z_global_max;
       if(E->Tracer.tracer_z[n] < z_global_min)
           E->Tracer.tracer_z[n] = z_global_min;



       }/* end of tracer loop */

      /*** Comment by Vlad 3/25/2005
           MPI for the tracer-advection code
      ***/


      m = 0;

      locx = E->parallel.me_loc[1];
      locy = E->parallel.me_loc[2];
      locz = E->parallel.me_loc[3];

      /* Am I the left-most proc.? If not, who is on my left? */
      if (locy == 0)
	left = -1;
      else
	left = E->parallel.loc2proc_map[m][locx][locy-1][locz];

      /* Am I the right-most proc.? If not, who is on my right? */
      if (locy == E->parallel.nprocy-1)
	right = -1;
      else
	right = E->parallel.loc2proc_map[m][locx][locy+1][locz];

      /* Am I the lower-most proc.? If not, who is beneath me? */
      if (locz == 0)
	down = -1;
      else
	down = E->parallel.loc2proc_map[m][locx][locy][locz-1];

      /* Am I the upper-most proc.? If not, who is above me? */
      if (locz == E->parallel.nprocz-1)
	up = -1;
      else
	up = E->parallel.loc2proc_map[m][locx][locy][locz+1];

      /* Am I the back-most proc.? If not, who is behind me? */
      if (locx == 0)
	back = -1;
       else
	 back = E->parallel.loc2proc_map[m][locx-1][locy][locz];

      /* Am I the front-most proc.? If not, who is in front of me? */
      if (locx == E->parallel.nprocx-1)
	front = -1;
      else
	front = E->parallel.loc2proc_map[m][locx+1][locy][locz];


      Left_proc_list[0]=left;
      Left_proc_list[1]=right;
      Left_proc_list[2]=down;
      Left_proc_list[3]=up;
      Left_proc_list[4]=back;
      Left_proc_list[5]=front;

      Right_proc_list[0]=right;
      Right_proc_list[1]=left;
      Right_proc_list[2]=up;
      Right_proc_list[3]=down;
      Right_proc_list[4]=front;
      Right_proc_list[5]=back;

      jump_new[0]=0;
      jump_new[1]=0;
      jump_new[2]=0;
      jump_new[3]=0;
      jump_new[4]=0;
      jump_new[5]=0;

      count_new[0]=0;
      count_new[1]=0;
      count_new[2]=0;
      count_new[3]=0;
      count_new[4]=0;
      count_new[5]=0;

      jj[0]=1;
      jj[1]=0;
      jj[2]=3;
      jj[3]=2;
      jj[4]=5;
      jj[5]=4;

      temp_tracers=0;
      Current_num_tracers=0;

      for(i=0;i<=11;i++){
        for(j=0;j<=E->Tracer.NUM_TRACERS;j++){
          tr_color_new[i][j]=999;
          tr_x_new[i][j]=999;
          tr_y_new[i][j]=999;
          tr_z_new[i][j]=999;

	  tr_color_1[j]=999;
	  tr_x_1[j]=999;
	  tr_y_1[j]=999;
	  tr_z_1[j]=999;
        }
      }


      i=0;
      j=0;
      k=0;
      l=0;
      m=0;
      o=0;
      p=0;


      for(n=1;n<=E->Tracer.LOCAL_NUM_TRACERS;n++){

        if(E->Tracer.tracer_y[n]>ymax) {
            /* excluding Nan */
          if(E->Tracer.tracer_y[n]+100 != 100) {
            tr_color_new[0][i]=E->Tracer.itcolor[n];
            tr_x_new[0][i]=E->Tracer.tracer_x[n];
            tr_y_new[0][i]=E->Tracer.tracer_y[n];
            tr_z_new[0][i]=E->Tracer.tracer_z[n];
            i++;
            jump_new[0]=i;
          }
        }
        else if(E->Tracer.tracer_y[n]<ymin) {
          if(E->Tracer.tracer_y[n]+100 != 100) {
            tr_color_new[1][j]=E->Tracer.itcolor[n];
            tr_x_new[1][j]=E->Tracer.tracer_x[n];
            tr_y_new[1][j]=E->Tracer.tracer_y[n];
            tr_z_new[1][j]=E->Tracer.tracer_z[n];
            j++;
            jump_new[1]=j;
          }
        }
        else if(E->Tracer.tracer_z[n]>zmax) {
          if(E->Tracer.tracer_z[n]+100 != 100) {
            tr_color_new[2][k]=E->Tracer.itcolor[n];
            tr_x_new[2][k]=E->Tracer.tracer_x[n];
            tr_y_new[2][k]=E->Tracer.tracer_y[n];
            tr_z_new[2][k]=E->Tracer.tracer_z[n];
            k++;
            jump_new[2]=k;
          }
        }
        else if(E->Tracer.tracer_z[n]<zmin) {
          if(E->Tracer.tracer_z[n]+100 != 100) {
            tr_color_new[3][l]=E->Tracer.itcolor[n];
            tr_x_new[3][l]=E->Tracer.tracer_x[n];
            tr_y_new[3][l]=E->Tracer.tracer_y[n];
            tr_z_new[3][l]=E->Tracer.tracer_z[n];
            l++;
            jump_new[3]=l;
          }
        }

        else if(E->Tracer.tracer_x[n]>xmax) {
          if(E->Tracer.tracer_x[n]+100 != 100) {
            tr_color_new[4][m]=E->Tracer.itcolor[n];
            tr_x_new[4][m]=E->Tracer.tracer_x[n];
            tr_y_new[4][m]=E->Tracer.tracer_y[n];
            tr_z_new[4][m]=E->Tracer.tracer_z[n];
            m++;
            jump_new[4]=m;
          }
        }
        else if(E->Tracer.tracer_x[n]<xmin) {
          if(E->Tracer.tracer_x[n]+100 != 100) {
            tr_color_new[5][o]=E->Tracer.itcolor[n];
            tr_x_new[5][o]=E->Tracer.tracer_x[n];
            tr_y_new[5][o]=E->Tracer.tracer_y[n];
            tr_z_new[5][o]=E->Tracer.tracer_z[n];
            o++;
            jump_new[5]=o;
          }
        }

        else {
          tr_color_1[p]=E->Tracer.itcolor[n];
          tr_x_1[p]=E->Tracer.tracer_x[n];
          tr_y_1[p]=E->Tracer.tracer_y[n];
          tr_z_1[p]=E->Tracer.tracer_z[n];
          p++;
        }
      }

      Previous_num_tracers=E->Tracer.LOCAL_NUM_TRACERS;
      Current_num_tracers=Previous_num_tracers-jump_new[0]-jump_new[1]-jump_new[2]-jump_new[3]-jump_new[4]-jump_new[5];

      /* compact the remaining tracer */
      for(p=1;p<=Current_num_tracers;p++){
	E->Tracer.itcolor[p]=tr_color_1[p-1];
	E->Tracer.tracer_x[p]=tr_x_1[p-1];
	E->Tracer.tracer_y[p]=tr_y_1[p-1];
	E->Tracer.tracer_z[p]=tr_z_1[p-1];
      }


      for(i=0;i<=5;i++){

	j=jj[i];

        if (Left_proc_list[i] >= 0) {
          proc=Left_proc_list[i];
          MPI_Irecv(tr_color_new[i+6], E->Tracer.NUM_TRACERS, MPI_FLOAT, proc, 11+i, world, &request[0]);
          MPI_Irecv(tr_x_new[i+6], E->Tracer.NUM_TRACERS, MPI_FLOAT, proc, 12+i, world, &request[1]);
          MPI_Irecv(tr_y_new[i+6], E->Tracer.NUM_TRACERS, MPI_FLOAT, proc, 13+i, world, &request[2]);
          MPI_Irecv(tr_z_new[i+6], E->Tracer.NUM_TRACERS, MPI_FLOAT, proc, 14+i, world, &request[3]);
        }

        if (Right_proc_list[i] >= 0) {
          proc=Right_proc_list[i];
          MPI_Send(tr_color_new[i], jump_new[i], MPI_FLOAT, proc, 11+i, world);
          MPI_Send(tr_x_new[i], jump_new[i], MPI_FLOAT, proc, 12+i, world);
          MPI_Send(tr_y_new[i], jump_new[i], MPI_FLOAT, proc, 13+i, world);
          MPI_Send(tr_z_new[i], jump_new[i], MPI_FLOAT, proc, 14+i, world);
        }

        if (Left_proc_list[i] >= 0) {
          MPI_Waitall(4, request, status);
          status_count = status[0];
          MPI_Get_count(&status_count, MPI_FLOAT, &count_new[i]);
        }


	temp_tracers=temp_tracers+count_new[i]-jump_new[i];
	E->Tracer.LOCAL_NUM_TRACERS=Previous_num_tracers+temp_tracers;


        /* append the tracers */

	if(i <= 1){
	  for(n=Current_num_tracers+count_new[j];n<=Current_num_tracers+count_new[i]+count_new[j]-1;n++) {
	    m=Current_num_tracers+count_new[j];
	    E->Tracer.itcolor[n+1]=tr_color_new[i+6][n-m];
	    E->Tracer.tracer_x[n+1]=tr_x_new[i+6][n-m];
	    E->Tracer.tracer_y[n+1]=tr_y_new[i+6][n-m];
	    E->Tracer.tracer_z[n+1]=tr_z_new[i+6][n-m];

	  }
	}


	else if (i <= 3) {
	  for(n=Current_num_tracers+count_new[0]+count_new[1]+count_new[j];n<=Current_num_tracers+count_new[0]+count_new[1]+count_new[i]+count_new[j]-1;n++) {
	    m=Current_num_tracers+count_new[0]+count_new[1]+count_new[j];
	    E->Tracer.itcolor[n+1]=tr_color_new[i+6][n-m];
	    E->Tracer.tracer_x[n+1]=tr_x_new[i+6][n-m];
	    E->Tracer.tracer_y[n+1]=tr_y_new[i+6][n-m];
	    E->Tracer.tracer_z[n+1]=tr_z_new[i+6][n-m];

	  }
	}

	else  {
	  for(n=Current_num_tracers+count_new[0]+count_new[1]+count_new[2]+count_new[3]+count_new[j];n<=E->Tracer.LOCAL_NUM_TRACERS-1;n++) {
	    m=Current_num_tracers+count_new[0]+count_new[1]+count_new[2]+count_new[3]+count_new[j];
	    E->Tracer.itcolor[n+1]=tr_color_new[i+6][n-m];
	    E->Tracer.tracer_x[n+1]=tr_x_new[i+6][n-m];
	    E->Tracer.tracer_y[n+1]=tr_y_new[i+6][n-m];
	    E->Tracer.tracer_z[n+1]=tr_z_new[i+6][n-m];

	  }
	}


      }


      free (tr_color_1);
      free (tr_x_1);
      free (tr_y_1);
      free (tr_z_1);
      for(i=0;i<=11;i++) {
	free (tr_color_new[i]);
	free (tr_x_new[i]);
	free (tr_y_new[i]);
	free (tr_z_new[i]);
      }


      return;
}
