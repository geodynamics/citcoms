/*
  
  Tracer_advection.c
      
      A program which initiates the distribution of tracers
      and advects those tracers in a time evolving velocity field.
      
      Called and used from the CitCOM finite element code.
      
      Written 2/96 M. Gurnis for Citcom in cartesian geometry
      
      Modified by Lijie in 1998 and by Vlad and Eh in 2005 for CitcomS
*/


#include <mpi.h>
#include <math.h>
#include <sys/types.h>
#include <malloc.h>
#include <stdlib.h> /* for "system" command */

#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"

void tracer_input(struct All_variables *E)
{
  int m=E->parallel.me;

  input_int("tracer",&(E->control.tracer),"0",m);
  input_string("tracer_file",E->control.tracer_file,"tracer.dat",m);
}


void tracer_initial_settings(E)
     struct All_variables *E;
{
   void tracer_setup();
   void tracer_advection();
   
   E->problem_tracer_setup=tracer_setup;
   E->problem_tracer_advection=tracer_advection;
 }


void tracer_setup(E)
     struct All_variables *E;
{
        int i,j,k,node;
	int m,ntr;
	float idummy,xdummy,ydummy,zdummy;
	FILE *fp;
        int nox,noy,noz,gnox,gnoy,gnoz;
        char output_file[255];
	MPI_Comm world;
	MPI_Status status;
	
        gnox=E->mesh.nox;
        gnoy=E->mesh.noy;
        gnoz=E->mesh.noz;
        nox=E->lmesh.nox;
        noy=E->lmesh.noy;
        noz=E->lmesh.noz;
	
        sprintf(output_file,"%s",E->control.tracer_file);
	fp=fopen(output_file,"r");
	if (fp == NULL) {
          fprintf(E->fp,"(Tracer_advection #1) Cannot open %s\n",output_file);
          exit(8);
	}
	fscanf(fp,"%d",&(E->Tracer.NUM_TRACERS));
	
        E->Tracer.tracer_x=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
        E->Tracer.tracer_y=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
        E->Tracer.tracer_z=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
        E->Tracer.itcolor=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
	
        E->Tracer.x_space=(float*) malloc((gnox+1)*sizeof(float));
        E->Tracer.y_space=(float*) malloc((gnoy+1)*sizeof(float));
        E->Tracer.z_space=(float*) malloc((gnoz+1)*sizeof(float));
	
	
	
        /***comment by Vlad 03/15/2005
	    each processor holds its own number of tracers 
        ***/
	
	ntr=1;	
	for(i=1;i<=E->Tracer.NUM_TRACERS;i++)
	  {
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
	
	
	return;

}

void tracer_advection(E)
     struct All_variables *E;
{
      int n;
      int i,j,k,m;
      int n_x,n_y,n_z;
      int node1,node2,node3,node4,node5,node6,node7,node8;
      int nno,nox,noy,noz,gnox,gnoy,gnoz;
      int iteration;
      int count;
      float x_tmp, y_tmp, z_tmp;
      float volume, tr_dx, tr_dy, tr_dz, dx, dy, dz;
      float w1,w2,w3,w4,w5,w6,w7,w8;
      float tr_v[NCS][4];
      MPI_Comm world;
      MPI_Status status;
      float xmin,xmax,ymin,ymax,zmin,zmax;
      float *tr_color,*tr_x,*tr_y,*tr_z;
      
      world=E->parallel.world;

      gnox=E->mesh.nox;
      gnoy=E->mesh.noy;
      gnoz=E->mesh.noz;
      nox=E->lmesh.nox;
      noy=E->lmesh.noy;
      noz=E->lmesh.noz;
      nno=nox*noy*noz;
      
      tr_x=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
      tr_y=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
      tr_z=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));
      tr_color=(float*) malloc((E->Tracer.NUM_TRACERS+1)*sizeof(float));  
      
      /*** comment by Vlad 3/25/2005
	   This part of code get the bounding box of the local mesh.
      ***/
      
      xmin=E->Tracer.x_space[1];
      xmax=E->Tracer.x_space[nox];
      ymin=E->Tracer.y_space[1];
      ymax=E->Tracer.y_space[noy];
      zmin=E->Tracer.z_space[1];
      zmax=E->Tracer.z_space[noz];
      
      //fprintf(stderr,"%d %d\n", E->parallel.nprocx, E->parallel.loc2proc_map[0][0][1][0]);
      
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

	
	//printf("%d %d %f %f %f\n", E->parallel.me,E->Tracer.NUM_TRACERS,E->Tracer.tracer_x[n],E->Tracer.tracer_y[n],E->Tracer.tracer_z[n]);

	/*  mid point method uses 2 iterations */
	
	for(iteration=1;iteration<=2;iteration++)
	  {
	    
	    if(iteration ==1) {
	      x_tmp=E->Tracer.tracer_x[n];
	      y_tmp=E->Tracer.tracer_y[n];
	      z_tmp=E->Tracer.tracer_z[n];
	    }
	    
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
	      }
	    }

	    for(j=1;j<noy;j++) {
	      if(y_tmp >= E->Tracer.y_space[j] && y_tmp <= E->Tracer.y_space[j+1]) {
		tr_dy=y_tmp-E->Tracer.y_space[j];
		dy=E->Tracer.y_space[j+1]-E->Tracer.y_space[j];
	        n_y=j;
	      }
	    }

	    for(k=1;k<noz;k++) {
	      if(z_tmp >= E->Tracer.z_space[k] && z_tmp <= E->Tracer.z_space[k+1]) {
		tr_dz=z_tmp-E->Tracer.z_space[k];
		dz=E->Tracer.z_space[k+1]-E->Tracer.z_space[k];
	        n_z=k;
	      }
	    }



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

	    
	    //	printf("%d %d %d %d %d %d %d %d %d\n",E->parallel.me, node1,node2,node3,node4,node5,node6,node7,node8);
	    //        printf("%d %f %f %f %f %f %f %f %f\n", E->parallel.me, E->GV[1][2][node1], E->GV[1][2][node2], E->GV[1][2][node3], E->GV[1][2][node4], E->GV[1][2][node5], E->GV[1][2][node6], E->GV[1][2][node7], E->GV[1][2][node8]);

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

	  }


	    }   /*  end of m  */
	    
	    
	  } /* end of iteration loop */


      } /* end of tracer loop */

  
     
	/*** Comment by Vlad 3/25/2005
	 Sending the tracer coordinates nad color to the next processor 
	 ***/
      
      
      i=1;

	for(n=1;n<=E->Tracer.LOCAL_NUM_TRACERS;n++) {

	  if(E->Tracer.tracer_y[n]<ymin) {  
	    
	    E->Tracer.itcolor[i]=E->Tracer.itcolor[n];
	    E->Tracer.tracer_x[i]=E->Tracer.tracer_x[n];
	    E->Tracer.tracer_y[i]=E->Tracer.tracer_y[n];
	    E->Tracer.tracer_z[i]=E->Tracer.tracer_z[n]; 
	    
	    i++;
	  } 

	}
	
	E->Tracer.TEMP_TRACERS=i-1;
	
	if(E->parallel.me == 1){
	  E->Tracer.LOCAL_NUM_TRACERS=E->Tracer.LOCAL_NUM_TRACERS-E->Tracer.TEMP_TRACERS;
	   
	}

	
	if(E->parallel.me == 1) { 
	  MPI_Send(E->Tracer.itcolor,  E->Tracer.TEMP_TRACERS, MPI_DOUBLE, 0, 16, world);
	  MPI_Send(E->Tracer.tracer_x, E->Tracer.TEMP_TRACERS, MPI_DOUBLE, 0, 17, world);
	  MPI_Send(E->Tracer.tracer_y, E->Tracer.TEMP_TRACERS, MPI_DOUBLE, 0, 18, world);
	  MPI_Send(E->Tracer.tracer_z, E->Tracer.TEMP_TRACERS, MPI_DOUBLE, 0, 19, world);
	}
	
	if(E->parallel.me == 0) { 
	  MPI_Recv(E->Tracer.itcolor,  E->Tracer.NUM_TRACERS, MPI_DOUBLE, 1, 16, world, &status);
	  MPI_Recv(E->Tracer.tracer_x, E->Tracer.NUM_TRACERS, MPI_DOUBLE, 1, 17, world, &status);
	  MPI_Recv(E->Tracer.tracer_y, E->Tracer.NUM_TRACERS, MPI_DOUBLE, 1, 18, world, &status);
	  MPI_Recv(E->Tracer.tracer_z, E->Tracer.NUM_TRACERS, MPI_DOUBLE, 1, 19, world, &status);
	} 
	
	MPI_Get_count(&status, MPI_DOUBLE, &count);
	if(E->parallel.me == 0) {
	  E->Tracer.LOCAL_NUM_TRACERS=E->Tracer.LOCAL_NUM_TRACERS+count;
	}
	
			
	return;
    }
