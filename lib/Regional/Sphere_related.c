/* Functions relating to the building and use of mesh locations ... */

#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

void coord_of_cap(E,icap)
   struct All_variables *E;
   int icap;
  {

  int m;
  int i,j,k,jjj,lev,temp,elx,ely,nox,noy,noz,node,nodes;
  int nprocxl,nprocyl,nproczl;
  int nnproc;
  int gnox,gnoy,gnoz;
  int nodesx,nodesy;
  char output_file[255];
  char a[100];
  int nn,step;
  FILE *fp,*fp1;
  double *x[5],*y[5],*z[5],xx[5],yy[5],zz[5];
  float *theta1[MAX_LEVELS],*fi1[MAX_LEVELS];
  double *SX[2];
  double *tt,*ff;
  double dt,df;
  double myatan();
  void parallel_process_termination();

  void even_divide_arc12();

  m=1;

  gnox=E->mesh.nox;
  gnoy=E->mesh.noy;
  gnoz=E->mesh.noz;
  nox=E->lmesh.nox;
  noy=E->lmesh.noy;
  noz=E->lmesh.noz;

  nprocxl=E->parallel.nprocxl;
  nprocyl=E->parallel.nprocyl;
  nproczl=E->parallel.nproczl;
  nnproc=nprocyl*nprocxl*nproczl;
  temp = max(E->mesh.NOY[E->mesh.levmax],E->mesh.NOX[E->mesh.levmax]);


if(E->control.coor==1)   {
  for(i=E->mesh.gridmin;i<=E->mesh.gridmax;i++)  {
  theta1[i] = (float *)malloc((temp+1)*sizeof(float));
  fi1[i]    = (float *)malloc((temp+1)*sizeof(float));
  }

  temp = E->mesh.NOY[E->mesh.levmax]*E->mesh.NOX[E->mesh.levmax];
 
  sprintf(output_file,"%s",E->control.coor_file);
  fp=fopen(output_file,"r");
	if (fp == NULL) {
          fprintf(E->fp,"(Sphere_related #1) Cannot open %s\n",output_file);
          exit(8);
	}

  fscanf(fp,"%s%d",&a,&nn);
   for(i=1;i<=gnox;i++) {
     fscanf(fp,"%d%f",&nn,&theta1[E->mesh.gridmax][i]);
   }

  fscanf(fp,"%s%d",&a,&nn);
   for(i=1;i<=gnoy;i++)  {
     fscanf(fp,"%d%f",&nn,&fi1[E->mesh.gridmax][i]);
    }

  fclose(fp);


  for (lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)  {

  if (E->control.NMULTIGRID||E->control.EMULTIGRID)
        step = (int) pow(2.0,(double)(E->mesh.levmax-lev));
    else
        step = 1;

     for (i=1;i<=E->mesh.NOX[lev];i++)
         theta1[lev][i] = theta1[E->mesh.gridmax][(i-1)*step+1];

     for (i=1;i<=E->mesh.NOY[lev];i++)
         fi1[lev][i] = fi1[E->mesh.gridmax][(i-1)*step+1];

  }


  for (lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)  {
     elx = E->lmesh.ELX[lev];
     ely = E->lmesh.ELY[lev];
     nox = E->lmesh.NOX[lev];
     noy = E->lmesh.NOY[lev];
     noz = E->lmesh.NOZ[lev];
        /* evenly divide arc linking 1 and 2, and the arc linking 4 and 3 */

                /* get the coordinates for the entire cap  */

        for (j=1;j<=nox;j++)
          for (k=1;k<=noy;k++)          {
             nodesx = E->lmesh.NXS[lev]+j-1;
             nodesy = E->lmesh.NYS[lev]+k-1;

             for (i=1;i<=noz;i++)          {
                node = i + (j-1)*noz + (k-1)*nox*noz;

                     /*   theta,fi,and r coordinates   */
                E->SX[lev][m][1][node] = theta1[lev][nodesx];
                E->SX[lev][m][2][node] = fi1[lev][nodesy];
                E->SX[lev][m][3][node] = E->sphere.R[lev][i];

                     /*   x,y,and z oordinates   */
                E->X[lev][m][1][node] = 
                            E->sphere.R[lev][i]*sin(theta1[lev][nodesx])*cos(fi1[lev][nodesy]);
                E->X[lev][m][2][node] = 
                            E->sphere.R[lev][i]*sin(theta1[lev][nodesx])*sin(fi1[lev][nodesy]);
                E->X[lev][m][3][node] = 
                            E->sphere.R[lev][i]*cos(theta1[lev][nodesx]);
                }
             }

     }       /* end for lev */



  for (lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)  {
  free ((void *)theta1[lev]);
  free ((void *)fi1[lev]   );
   }

 }

else if(E->control.coor==0)   {
  

  for(i=1;i<=5;i++)  {
  x[i] = (double *) malloc((E->parallel.nproc+1)*sizeof(double));
  y[i] = (double *) malloc((E->parallel.nproc+1)*sizeof(double));
  z[i] = (double *) malloc((E->parallel.nproc+1)*sizeof(double));
  tt = (double *) malloc((4+1)*sizeof(double));
  ff = (double *) malloc((4+1)*sizeof(double));

  }

  temp = E->lmesh.NOY[E->mesh.levmax]*E->lmesh.NOX[E->mesh.levmax];

  SX[0]  = (double *)malloc((temp+1)*sizeof(double));
  SX[1]  = (double *)malloc((temp+1)*sizeof(double));


     tt[1] = E->sphere.cap[m].theta[1]+(E->sphere.cap[m].theta[2] -E->sphere.cap[m].theta[1])/nprocxl*(E->parallel.me_locl[1]);
     tt[2] = E->sphere.cap[m].theta[1]+(E->sphere.cap[m].theta[2] -E->sphere.cap[m].theta[1])/nprocxl*(E->parallel.me_locl[1]+1);
     tt[3] = tt[2];
     tt[4] = tt[1];
     ff[1] = E->sphere.cap[m].fi[1]+(E->sphere.cap[m].fi[4] -E->sphere.cap[1].fi[1])/nprocyl*(E->parallel.me_locl[2]);
     ff[2] = ff[1];
     ff[3] = E->sphere.cap[m].fi[1]+(E->sphere.cap[m].fi[4] -E->sphere.cap[1].fi[1])/nprocyl*(E->parallel.me_locl[2]+1);
     ff[4] = ff[3];


  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)  {

     elx = E->lmesh.ELX[lev];
     ely = E->lmesh.ELY[lev];
     nox = E->lmesh.NOX[lev];
     noy = E->lmesh.NOY[lev];
     noz = E->lmesh.NOZ[lev];
        /* evenly divide arc linking 1 and 2, and the arc linking 4 and 3 */

      for(j=1;j<=nox;j++) {
       dt=(tt[3]-tt[1])/elx;
       df=(ff[3]-ff[1])/ely;

        for (k=1;k<=noy;k++)   {
           nodes = j + (k-1)*nox;
           SX[0][nodes] = tt[1]+dt*(j-1);
           SX[1][nodes] = ff[1]+df*(k-1);
           }

        }       /* end for j */

                /* get the coordinates for the entire cap  */

        for (j=1;j<=nox;j++)
          for (k=1;k<=noy;k++)          {
             nodes = j + (k-1)*nox;
             for (i=1;i<=noz;i++)          {
                node = i + (j-1)*noz + (k-1)*nox*noz;

                     /*   theta,fi,and r coordinates   */
                E->SX[lev][m][1][node] = SX[0][nodes];
                E->SX[lev][m][2][node] = SX[1][nodes];
                E->SX[lev][m][3][node] = E->sphere.R[lev][i];

                     /*   x,y,and z oordinates   */
                E->X[lev][m][1][node] = 
                            E->sphere.R[lev][i]*sin(SX[0][nodes])*cos(SX[1][nodes]);
                E->X[lev][m][2][node] = 
                            E->sphere.R[lev][i]*sin(SX[0][nodes])*sin(SX[1][nodes]);
                E->X[lev][m][3][node] = 
                            E->sphere.R[lev][i]*cos(SX[0][nodes]);
                }
             }

     }       /* end for lev */



  free ((void *)SX[0]);
  free ((void *)SX[1]);
}

  return;
  }

/* =================================================
  this routine evenly divides the arc between points
  1 and 2 in a great cicle. The word "evenly" means 
  anglewise evenly.
 =================================================*/

void even_divide_arc12(elx,x1,y1,z1,x2,y2,z2,theta,fi)
 double x1,y1,z1,x2,y2,z2,*theta,*fi;
 int elx;
{
  double dx,dy,dz,xx,yy,zz,myatan();
  int j, nox;

  nox = elx+1;

  dx = (x2 - x1)/elx;
  dy = (y2 - y1)/elx;
  dz = (z2 - z1)/elx;
  for (j=1;j<=nox;j++)   {
      xx = x1 + dx*(j-1) + 5.0e-32;
      yy = y1 + dy*(j-1);
      zz = z1 + dz*(j-1);
      theta[j] = acos(zz/sqrt(xx*xx+yy*yy+zz*zz));
      fi[j]    = myatan(yy,xx);
      }

   return;
  }


/* =================================================
  rotate the mesh 
 =================================================*/
void rotate_mesh(E,m,icap)
   struct All_variables *E;
   int icap,m;
  {

  int i,lev;
  double t[4],myatan();

  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)  {
    for (i=1;i<=E->lmesh.NNO[lev];i++)  {
      t[0] = E->X[lev][m][1][i]*E->sphere.dircos[1][1]+ 
             E->X[lev][m][2][i]*E->sphere.dircos[1][2]+ 
             E->X[lev][m][3][i]*E->sphere.dircos[1][3]; 
      t[1] = E->X[lev][m][1][i]*E->sphere.dircos[2][1]+ 
             E->X[lev][m][2][i]*E->sphere.dircos[2][2]+ 
             E->X[lev][m][3][i]*E->sphere.dircos[2][3]; 
      t[2] = E->X[lev][m][1][i]*E->sphere.dircos[3][1]+ 
             E->X[lev][m][2][i]*E->sphere.dircos[3][2]+ 
             E->X[lev][m][3][i]*E->sphere.dircos[3][3]; 

      E->X[lev][m][1][i] = t[0];
      E->X[lev][m][2][i] = t[1];
      E->X[lev][m][3][i] = t[2];
      E->SX[lev][m][1][i] = acos(t[2]/E->SX[lev][m][3][i]);
      E->SX[lev][m][2][i] = myatan(t[1],t[0]);
      }
    }    /* lev */

  return;
  }
