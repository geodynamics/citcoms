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
    void read_input_files_for_timesteps();

    read_input_files_for_timesteps(E,1,1); /* read velocity(1) and output(1) */

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

/*
  sprintf(output_file,"mat.%d",E->parallel.me);
  fp=fopen(output_file,"w");
	if (fp == NULL) {
          fprintf(E->fp,"(Problem_related #3) Cannot open %s\n",output_file);
          exit(8);
	}
  for (m=1;m<=E->sphere.caps_per_proc;m++)   
      for(el=1;el<=E->lmesh.nel;el++)  
         fprintf(fp,"%d %d %f\n", el,E->mat[m][el],E->VIP[m][el]);
  fclose(fp); 
*/

     free ((void *) VIP1);
     free ((void *) VIP2);


   return;

}


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
    int i,ii,ll,mm,j,k,n,nodeg,nodel,node;
    int intage, pos_age;

    const int dims=E->mesh.nsd;
    pos_age = 1;


    nox=E->mesh.nox;
    noy=E->mesh.noy;
    noz=E->mesh.noz;
    nox1=E->lmesh.nox;
    noz1=E->lmesh.noz;
    noy1=E->lmesh.noy;
    lev=E->mesh.levmax;

    
    age=find_age_in_MY(E);

    intage = age;
    newage1 = 1.0*intage;
    newage2 = 1.0*intage + 1.0;
    if (newage1 < 0.0) { /* age is negative -> use age=0 for input files */
        newage1 = 0.0;
        pos_age = 0;
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

    case 2:  /* read ages for lithosphere tempperature boundary conditions */
      sprintf(output_file1,"%s%0.0f",E->control.lith_age_file,newage1);
      sprintf(output_file2,"%s%0.0f",E->control.lith_age_file,newage2);
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
         VB1[1][i]=E->data.timedir*VB1[1][i];
         VB1[2][i]=E->data.timedir*VB1[2][i];
         if (pos_age) {
             fscanf(fp2,"%f %f",&(VB2[1][i]),&(VB2[2][i]));
             VB2[1][i]=E->data.timedir*VB2[1][i];
             VB2[2][i]=E->data.timedir*VB2[2][i];
         }
	/* if( E->parallel.me ==0)
	  fprintf(stderr,"%d %f  %f  %f  %f\n",i,VB1[1][i],VB1[2][i],VB2[1][i],VB2[2][i]); */
      }
      fclose(fp1);
      if (pos_age) fclose(fp2);

      if(E->parallel.me_locl[3]==E->parallel.nproczl-1 )  {
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
                    E->sphere.cap[1].VB[1][nodel] = VB1[1][nodeg];
                    E->sphere.cap[1].VB[2][nodel] = VB1[2][nodeg];
                    E->sphere.cap[1].VB[3][nodel] = 0.0;
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
      else {
        age_in_MY = fabs(age_in_MY);
      }

   return(age_in_MY);
}
