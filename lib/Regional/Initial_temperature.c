
#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h> /* for "system" command */
#include <strings.h>

void convection_initial_temperature(E)
     struct All_variables *E;
{
    float find_age_in_MY();

    int i,j,k,p,node,ii,jj,m,mm,ll;
    int nox,noy,noz,noz2;
    int gnox,gnoy,gnoz,nodeg;
    char output_file[255],input_s[1000];
    float lscale,ficenter,rcenter,dist2,lscalex,rscale,age;

    double con,temp,t1,f1,r1;
    float rad,beta,notusedhere; 
    float a,b,c,d,e,f,g; 
    FILE *fp,*fp1,*fp2;

    float v1,v2,v3;
    float e_4;


    void temperatures_conform_bcs();
    void thermal_buoyancy();
    void parallel_process_termination();
    void sphere_harmonics_layer();
    void inv_sphere_harmonics();
    
    const int dims=E->mesh.nsd;
    rad = 180.0/M_PI;
    e_4=1.e-4;


    noy=E->lmesh.noy;  
    nox=E->lmesh.nox;  
    noz=E->lmesh.noz;  

    noz2=(E->mesh.noz-1)/2+1;  
       
       gnox=E->mesh.nox;
       gnoy=E->mesh.noy;
       gnoz=E->mesh.noz;



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
        sscanf(input_s,"%d %d %f",&ll,&mm,&notusedhere);

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


/* use this section to include imposed ages at the surface CPC 4/27/00 */
     if(E->control.lith_age==1)   {

if(E->control.lith_age_time==1)   { 
  /* if opening lithosphere age info every timestep - naming is different*/
  age=find_age_in_MY(E);
  sprintf(output_file,"%s%0.0f",E->control.lith_age_file,age);
  if(E->parallel.me==0)  {
    fprintf(E->fp,"%s %s\n","Initial Lithosphere age info:",output_file);
  }
}
else {     /* just open lithosphere age info here*/
  sprintf(output_file,"%s",E->control.lith_age_file);
}


       fp1=fopen(output_file,"r");
	if (fp1 == NULL) {
          fprintf(E->fp,"(Convection.c #2) Cannot open %s\n",output_file);
          exit(8);
	}

          for(i=1;i<=gnoy;i++)  
            for(j=1;j<=gnox;j++) {
             node=j+(i-1)*gnox;
             fscanf(fp1,"%f",&(E->age_t[node]));

             E->age_t[node]=E->age_t[node]/E->data.scalet;

       }
        fclose(fp1);

        for(m=1;m<=E->sphere.caps_per_proc;m++)
          for(i=1;i<=noy;i++)  
            for(j=1;j<=nox;j++) 
              for(k=1;k<=noz;k++)  {
                nodeg=E->lmesh.nxs-1+j+(E->lmesh.nys+i-2)*gnox;
                node=k+(j-1)*noz+(i-1)*nox*noz;
                r1=E->sx[m][3][node];
                   if(  r1 >= E->sphere.ro-E->control.lith_age_depth ) 
                   { /* if closer than (lith_age_depth) from top */
                     temp = (E->sphere.ro-r1) *0.5 /sqrt(E->age_t[nodeg]);
                     E->T[m][node]  = E->control.mantle_temp * erf(temp);
                   }


            }     /* end k   */




     } /* end lith age */





        for(m=1;m<=E->sphere.caps_per_proc;m++)
          for(i=1;i<=noy;i++)  
            for(j=1;j<=nox;j++) 
              for(k=1;k<=noz;k++)  {
                node=k+(j-1)*noz+(i-1)*nox*noz;
                  r1=E->sx[m][3][node];

                     if(fabs(r1-E->sphere.ro)>=e_4 && fabs(r1-E->sphere.ri)>=e_4)  {
                      E->sphere.cap[m].TB[1][node]=E->T[m][node];
                      E->sphere.cap[m].TB[2][node]=E->T[m][node];
                      E->sphere.cap[m].TB[3][node]=E->T[m][node];
                    }
               }

        }

    else   { /* if control.restart=0 => not using a restart file */

      
     if(E->control.lith_age==1)   {
/* used if the lithosphere age is given in an input file. CPC 1/28/00 */

       gnox=E->mesh.nox;
       gnoy=E->mesh.noy;
       gnoz=E->mesh.noz;

if(E->control.lith_age_time==1)   { 
  /* if opening lithosphere age info every timestep - naming is different*/
  age=find_age_in_MY(E);
  sprintf(output_file,"%s%0.0f",E->control.lith_age_file,age);
  if(E->parallel.me==0)  {
    fprintf(E->fp,"%s %s\n","Initial Lithosphere age info:",output_file);
  }
}
else {     /* just open lithosphere age info here*/
  sprintf(output_file,"%s",E->control.lith_age_file);
}

       fp1=fopen(output_file,"r");
	if (fp1 == NULL) {
          fprintf(E->fp,"(Convection.c #3) Cannot open %s\n",output_file);
          exit(8);
	}


          for(i=1;i<=gnoy;i++)  
            for(j=1;j<=gnox;j++) {
             node=j+(i-1)*gnox;
             fscanf(fp1,"%f",&(E->age_t[node]));

             E->age_t[node]=E->age_t[node]/E->data.scalet;

       }

        fclose(fp1);
       

        for(m=1;m<=E->sphere.caps_per_proc;m++)
          for(i=1;i<=noy;i++)  
            for(j=1;j<=nox;j++) 
              for(k=1;k<=noz;k++)  {
                nodeg=E->lmesh.nxs-1+j+(E->lmesh.nys+i-2)*gnox;
                node=k+(j-1)*noz+(i-1)*nox*noz;
                E->T[m][node] = 1.0;
                  t1=E->sx[m][1][node];
                  f1=E->sx[m][2][node];
                  r1=E->sx[m][3][node];

                     temp = (E->sphere.ro-r1) *0.5 /sqrt(E->age_t[nodeg]);
                     E->T[m][node]  = E->control.mantle_temp * erf(temp);

                     if(fabs(r1-E->sphere.ro)>=e_4 && fabs(r1-E->sphere.ri)>=e_4)  {

                      E->sphere.cap[m].TB[1][node]=E->T[m][node];
                      E->sphere.cap[m].TB[2][node]=E->T[m][node];
                      E->sphere.cap[m].TB[3][node]=E->T[m][node];

                    }


            }     /* end k   */

          }  /* end of lith-age  */

       else  
/* used otherwise - Set your own initial temperature initial conditions! */
          {
        int number_of_perturbations;
        int perturb_ll[32], perturb_mm[32], load_depth[32];
	float perturb_mag[32];
	float tlen, flen;

        m = E->parallel.me;
	tlen = M_PI / (E->control.theta_max - E->control.theta_min);
	flen = M_PI / (E->control.fi_max - E->control.fi_min);

      /* This part put a temperature anomaly at depth where the global 
	 node number is equal to load_depth. The horizontal pattern of
	 the anomaly is given by spherical harmonic ll & mm. */

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
/* 	  if (! input_int_vector("perturblayer",number_of_perturbations,load_depth,m) ) { */
/* 	    fprintf(stderr,"Missing input parameter: 'perturblayer'\n"); */
/* 	    parallel_process_termination(); */
/* 	  } */
	}
	else {
	  number_of_perturbations = 1;
          perturb_mag[0] = E->mesh.elz/(E->sphere.ro-E->sphere.ri);
	  perturb_mm[0] = 2;
	  perturb_ll[0] = 2;
/* 	  load_depth[0] = noz/2; */
	}

	for(m=1;m<=E->sphere.caps_per_proc;m++)
	  for(i=1;i<=noy;i++)  
	    for(j=1;j<=nox;j++) 
	      for(k=1;k<=noz;k++)  {
		ii = k + E->lmesh.nzs - 1;
		node=k+(j-1)*noz+(i-1)*nox*noz;
		t1 = (E->sx[m][1][node] - E->control.theta_min) * tlen;
		f1 = (E->sx[m][2][node] - E->control.fi_min) * flen;
		r1 = E->sx[m][3][node] - E->sphere.ri;
		E->T[m][node] = E->control.TBCbotval - (E->control.TBCtopval + E->control.TBCbotval)*r1/(E->sphere.ro - E->sphere.ri);

		for (p=0; p<number_of_perturbations; p++) {
		  mm = perturb_mm[p];
		  ll = perturb_ll[p];
		  con = perturb_mag[p];
		  
		  E->T[m][node] += con*cos(ll*f1)*cos(mm*t1)*sin(M_PI*r1/(E->sphere.ro - E->sphere.ri));
		  E->T[m][node] = max(min(E->T[m][node], 1.0), 0.0);
		}
	      }

         }   /* end for else  */

       }   /* end for else  */

     

  temperatures_conform_bcs(E);

  if (E->control.verbose==1)  {
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

