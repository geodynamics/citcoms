/* Functions relating to the building and use of mesh locations ... */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

void coord_of_cap(E,m,icap)
   struct All_variables *E;
   int icap,m;
  {

  int i,j,k,lev,temp,elx,ely,nox,noy,noz,node,nodes;
  int lelx,lely,lnox,lnoy,lnoz;
  double x[5],y[5],z[5],xx[5],yy[5],zz[5];
  double *theta1,*fi1,*theta2,*fi2,*theta,*fi,*SX[2];
  double myatan();

  void even_divide_arc12();

  temp = max(E->mesh.NOY[E->mesh.levmax],E->mesh.NOX[E->mesh.levmax]);

  theta1 = (double *)malloc((temp+1)*sizeof(double));
  fi1    = (double *)malloc((temp+1)*sizeof(double));
  theta2 = (double *)malloc((temp+1)*sizeof(double));
  fi2    = (double *)malloc((temp+1)*sizeof(double));
  theta  = (double *)malloc((temp+1)*sizeof(double));
  fi     = (double *)malloc((temp+1)*sizeof(double));

  temp = E->mesh.NOY[E->mesh.levmax]*E->mesh.NOX[E->mesh.levmax]; /* allocate enough for the entire cap */

  SX[0]  = (double *)malloc((temp+1)*sizeof(double));
  SX[1]  = (double *)malloc((temp+1)*sizeof(double));

  for (i=1;i<=4;i++)    {
     x[i] = sin(E->sphere.cap[icap].theta[i])*cos(E->sphere.cap[icap].fi[i]);
     y[i] = sin(E->sphere.cap[icap].theta[i])*sin(E->sphere.cap[icap].fi[i]);
     z[i] = cos(E->sphere.cap[icap].theta[i]);
     }
  
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)  {

     elx = E->lmesh.ELX[lev]*E->parallel.nprocx;
     ely = E->lmesh.ELY[lev]*E->parallel.nprocy;
     nox = elx+1;
     noy = ely+1;

     lelx = E->lmesh.ELX[lev];
     lely = E->lmesh.ELY[lev];
     lnox = lelx+1;
     lnoy = lely+1;
     lnoz = E->lmesh.NOZ[lev];
        /* evenly divide arc linking 1 and 2, and the arc linking 4 and 3 */

     even_divide_arc12(elx,x[1],y[1],z[1],x[2],y[2],z[2],theta1,fi1);
     even_divide_arc12(elx,x[4],y[4],z[4],x[3],y[3],z[3],theta2,fi2);

     for (j=1;j<=nox;j++)   {

         /* pick up the two ends  */
        xx[1] = sin(theta1[j])*cos(fi1[j]);
        yy[1] = sin(theta1[j])*sin(fi1[j]);
        zz[1] = cos(theta1[j]);
        xx[2] = sin(theta2[j])*cos(fi2[j]);
        yy[2] = sin(theta2[j])*sin(fi2[j]);
        zz[2] = cos(theta2[j]);


        even_divide_arc12(ely,xx[1],yy[1],zz[1],xx[2],yy[2],zz[2],theta,fi);

        for (k=1;k<=noy;k++)   {
           nodes = j + (k-1)*nox;
           SX[0][nodes] = theta[k];
           SX[1][nodes] = fi[k];
           }


        }       /* end for j */

                /* get the coordinates for the entire cap  */

        for (j=1;j<=lnox;j++)
          for (k=1;k<=lnoy;k++)          {
             nodes = (j+(E->lmesh.NXS[lev]-1))+(k-1+(E->lmesh.NYS[lev]-1))*nox;
             for (i=1;i<=lnoz;i++)          {
                node = i + (j-1)*lnoz + (k-1)*lnox*lnoz;

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
  free ((void *)theta );
  free ((void *)theta1);
  free ((void *)theta2);
  free ((void *)fi    );
  free ((void *)fi1   );
  free ((void *)fi2   );

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
