/* Functions relating to the building and use of mesh locations ... */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h>

/*   ======================================================================
    ======================================================================  */

void set_sphere_harmonics(E)
   struct All_variables *E;

  {
  int node,ll,mm,i,j;
  double multis(),dth,dfi,sqrt_multis();
  void construct_interp_net();
  void compute_sphereh_table();

  E->sphere.sx[1] = (double *) malloc((E->sphere.nsf+1)*sizeof(double));
  E->sphere.sx[2] = (double *) malloc((E->sphere.nsf+1)*sizeof(double));
  E->sphere.int_cap = (int *) malloc((E->sphere.nsf+1)*sizeof(int));
  E->sphere.int_ele = (int *) malloc((E->sphere.nsf+1)*sizeof(int));

  E->sphere.radius=(float *) malloc((E->sphere.slab_layers+3)*sizeof(float));

   i=0;
   for (ll=0;ll<=E->sphere.llmax;ll++)
     for (mm=0;mm<=ll;mm++)   {
        E->sphere.hindex[ll][mm] = i;
        i++;
	}
  E->sphere.hindice = i;

  E->sphere.con = (double *)malloc((E->sphere.hindice+3)*sizeof(double));

  for (i=1;i<=E->sphere.lelx;i++)
    E->sphere.tableplm[i]= (double *)malloc((E->sphere.hindice+3)*sizeof(double));
  for (i=1;i<=E->sphere.lely;i++) {
    E->sphere.tablecosf[i]= (double *)malloc((E->sphere.output_llmax+3)*sizeof(double));
    E->sphere.tablesinf[i]= (double *)malloc((E->sphere.output_llmax+3)*sizeof(double));
    }

  for (i=1;i<=E->sphere.lnox;i++)
    E->sphere.tableplm_n[i]= (double *)malloc((E->sphere.hindice+3)*sizeof(double));
  for (i=1;i<=E->sphere.lnoy;i++) {
    E->sphere.tablecosf_n[i]= (double *)malloc((E->sphere.output_llmax+3)*sizeof(double));
    E->sphere.tablesinf_n[i]= (double *)malloc((E->sphere.output_llmax+3)*sizeof(double));
    }
  E->sphere.sien  = (struct SIEN *) malloc((E->sphere.lsnel+1)*sizeof(struct SIEN));



/*     ll = E->sphere.slab_layers+2; */

   for (i=0;i<=1;i++)   {
     E->sphere.harm_tpgt[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
     E->sphere.harm_tpgb[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
     E->sphere.harm_velp[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
     E->sphere.harm_velt[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
     E->sphere.harm_divg[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
     E->sphere.harm_vort[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
     E->sphere.harm_visc[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
/*       E->sphere.harm_slab[i]=(float*)malloc((ll*E->sphere.hindice+2)*sizeof(float)); */
     }



  for (ll=0;ll<=E->sphere.output_llmax;ll++)
     for (mm=0;mm<=ll;mm++)   {
        E->sphere.con[E->sphere.hindex[ll][mm]] = 
	     sqrt( (2.0-((mm==0)?1.0:0.0))*(2*ll+1)/(4.0*M_PI) )
	    *sqrt_multis(ll+mm,ll-mm);  /* which is sqrt((ll-mm)!/(ll+mm)!) */
	}

  dth = M_PI/E->sphere.elx;
  dfi = 2.0*M_PI/E->sphere.ely;

  for (j=1;j<=E->sphere.noy;j++)
  for (i=1;i<=E->sphere.nox;i++) {
    node = i+(j-1)*E->sphere.nox;
    E->sphere.sx[1][node] = dth*(i-1);
    E->sphere.sx[2][node] = dfi*(j-1);
    }


  construct_interp_net(E);


  compute_sphereh_table(E);

  return;
  }
/*   ======================================================================
    ======================================================================  */

void sphere_harmonics_layer(E,T,sphc,sphs,iprint,filen)
   struct All_variables *E;
   float **T,*sphc,*sphs;
   int iprint;
   char * filen;
 {
/*    void sphere_expansion(); */
/*    void sphere_interpolate(); */
/*    void parallel_process_termination(); */
/*    void parallel_process_sync(); */
/*    void print_field_spectral_regular(); */
/*    FILE *fp; */
/*    char output_file[255]; */
/*    int i,node,j,ll,mm,printt,proc_loc; */
/*    float minx,maxx,t,f,rad; */
/*    static int been_here=0; */
/*    float *TG; */

/*    rad=180.0/M_PI; */

/*    maxx=-1.e6; */
/*    minx=1.e6; */

/*    printt=0; */

/*    if(E->parallel.me_loc[3]==E->parallel.nprocz-1 && iprint==1) printt=1; */
/*    if(E->parallel.me_loc[3]==0 && iprint==0) printt=1; */

/*    TG  = (float *)malloc((E->sphere.nsf+1)*sizeof(float)); */

/*    proc_loc = E->parallel.me_loc[3]; */

/*       sphere_interpolate(E,T,TG); */

/*       sphere_expansion (E,TG,sphc,sphs); */

/*       if (printt) */
/* 	print_field_spectral_regular(E,TG,sphc,sphs,proc_loc,filen); */


/*    parallel_process_sync(); */

/*    free ((void *)TG); */

   return;
  }
  
/* ================================================
  compute angle and area  
 ================================================*/

void compute_angle_surf_area (E)
  struct All_variables *E;
{

 int es,el,m,i,j,ii,ia[5],lev;
 double aa,y1[4],y2[4],angle[6],xx[4][5],area_sphere_cap();
 void get_angle_sphere_cap();
 void parallel_process_termination();

 for (m=1;m<=E->sphere.caps_per_proc;m++)   {
   ia[1] = 1;
   ia[2] = E->lmesh.noz*E->lmesh.nox-E->lmesh.noz+1;
   ia[3] = E->lmesh.nno-E->lmesh.noz+1;
   ia[4] = ia[3]-E->lmesh.noz*(E->lmesh.nox-1);

   for (i=1;i<=4;i++)  {
     xx[1][i] = E->x[m][1][ia[i]]/E->sx[m][3][ia[1]];
     xx[2][i] = E->x[m][2][ia[i]]/E->sx[m][3][ia[1]];
     xx[3][i] = E->x[m][3][ia[i]]/E->sx[m][3][ia[1]];
     }

   get_angle_sphere_cap(xx,angle);

   for (i=1;i<=4;i++)         /* angle1: bet 1 & 2; angle2: bet 2 & 3 ..*/
      E->sphere.angle[m][i] = angle[i];

   E->sphere.area[m] = area_sphere_cap(angle);

   for (lev=E->mesh.levmax;lev>=E->mesh.levmin;lev--) 
     for (es=1;es<=E->lmesh.SNEL[lev];es++)              {
       el = (es-1)*E->lmesh.ELZ[lev]+1;
       for (i=1;i<=4;i++)
         ia[i] = E->IEN[lev][m][el].node[i];

       for (i=1;i<=4;i++)  {
         xx[1][i] = E->X[lev][m][1][ia[i]]/E->SX[lev][m][3][ia[1]];
         xx[2][i] = E->X[lev][m][2][ia[i]]/E->SX[lev][m][3][ia[1]];
         xx[3][i] = E->X[lev][m][3][ia[i]]/E->SX[lev][m][3][ia[1]];
         }

       get_angle_sphere_cap(xx,angle);

       for (i=1;i<=4;i++)         /* angle1: bet 1 & 2; angle2: bet 2 & 3 ..*/
         E->sphere.angle1[lev][m][i][es] = angle[i];

       E->sphere.area1[lev][m][es] = area_sphere_cap(angle);

 /*fprintf(E->fp_out,"lev%d %d %.6e %.6e %.6e %.6e %.6e\n",lev,es,angle[1],angle[2],angle[3],angle[4],E->sphere.area1[lev][m][es]);
*/
       }  /* end for lev and es */

   }  /* end for m */

 return;
 }

/* ================================================
 area of spherical rectangle 
 ================================================ */
double area_sphere_cap(angle)
 double angle[6];
 {

 double area,a,b,c;
 double area_of_sphere_triag();

   a = angle[1]; 
   b = angle[2]; 
   c = angle[5]; 
   area = area_of_sphere_triag(a,b,c);

   a = angle[3]; 
   b = angle[4]; 
   c = angle[5]; 
   area += area_of_sphere_triag(a,b,c);

   return (area);
   }

/* ================================================
 area of spherical triangle
 ================================================ */
double area_of_sphere_triag(a,b,c)
 double a,b,c;
 {

 double ss,ak,aa,bb,cc,area;
 const double e_16 = 1.0e-16;
 const double two = 2.0;
 const double pt5 = 0.5;

 ss = (a+b+c)*pt5;
 area=0.0;
 a = sin(ss-a);
 b = sin(ss-b);
 c = sin(ss-c);
 ak = a*b*c/sin(ss);   /* sin(ss-a)*sin(ss-b)*sin(ss-c)/sin(ss)  */
 if(ak<e_16) return (area);
 ak = sqrt(ak);
 aa = two*atan(ak/a);
 bb = two*atan(ak/b);
 cc = two*atan(ak/c);
 area = aa+bb+cc-M_PI;

 return (area);
 }

/*  =====================================================================
 get the area for given five points (4 nodes for a rectangle and one test node)
 angle [i]: angle bet test node and node i of the rectangle
 angle1[i]: angle bet nodes i and i+1 of the rectangle
 ====================================================================== */
double area_of_5points(E,lev,m,el,x,ne)
  struct All_variables *E;
  int lev,m,el,ne;
  double x[4];
 {
  int i,es,ia[5];
  double area1,get_angle(),area_of_sphere_triag();
  double xx[4],angle[5],angle1[5];

    for (i=1;i<=4;i++)
      ia[i] = E->IEN[lev][m][el].node[i];

    es = (el-1)/E->lmesh.ELZ[lev]+1;

    for (i=1;i<=4;i++)                 {
      xx[1] = E->X[lev][m][1][ia[i]]/E->SX[lev][m][3][ia[1]];
      xx[2] = E->X[lev][m][2][ia[i]]/E->SX[lev][m][3][ia[1]];
      xx[3] = E->X[lev][m][3][ia[i]]/E->SX[lev][m][3][ia[1]];
      angle[i] = get_angle(x,xx);  /* get angle bet (i,j) and other four*/
      angle1[i]= E->sphere.angle1[lev][m][i][es];
      }

    area1 = area_of_sphere_triag(angle[1],angle[2],angle1[1])
          + area_of_sphere_triag(angle[2],angle[3],angle1[2])
          + area_of_sphere_triag(angle[3],angle[4],angle1[3])
          + area_of_sphere_triag(angle[4],angle[1],angle1[4]);

   return (area1);
  }

/*  ================================
 get the angle for given four points spherical rectangle 
 ================================= */

void  get_angle_sphere_cap(xx,angle)
 double xx[4][5],angle[6];
 {

  int i,j,ii;
  double y1[4],y2[4],get_angle();;

   for (i=1;i<=4;i++)     {     /* angle1: bet 1 & 2; angle2: bet 2 & 3 ..*/
     for (j=1;j<=3;j++)     {
       ii=(i==4)?1:(i+1);
       y1[j] = xx[j][i];
       y2[j] = xx[j][ii];
       }
     angle[i] = get_angle(y1,y2);
     }

   for (j=1;j<=3;j++) {
       y1[j] = xx[j][1];
       y2[j] = xx[j][3];
       }

   angle[5] = get_angle(y1,y2);     /* angle5 for betw 1 and 3: diagonal */
   return;
   }

/*  ================================
 get the angle for given two points 
 ================================= */
double get_angle(x,xx)
  double x[4],xx[4];
  {
  double dist,angle;
 const double pt5 = 0.5;
 const double two = 2.0;

       dist=sqrt( (x[1]-xx[1])*(x[1]-xx[1])
                + (x[2]-xx[2])*(x[2]-xx[2])
                + (x[3]-xx[3])*(x[3]-xx[3]) )*pt5;
       angle = asin(dist)*two;

  return (angle);
  }

/* ================================================
 for a given node, this routine gives which cap and element
 the node is in.
 ================================================*/
void construct_interp_net(E)
  struct All_variables *E;
  {

/*    void parallel_process_termination(); */
/*    void parallel_process_sync(); */
/*    int ii,jj,es,i,j,m,el,node; */
/*    int locate_cap(),locate_element(); */
/*    double x[4],t,f; */

/*    const int ends=4; */

/*    for (i=1;i<=E->sphere.nox;i++) */
/*      for (j=1;j<=E->sphere.noy;j++)   { */
/*        node = i+(j-1)*E->sphere.nox; */
/*        E->sphere.int_cap[node]=0; */
/*        E->sphere.int_ele[node]=0; */
/*        } */


/*    for (i=1;i<=E->sphere.nox;i++) */
/*      for (j=1;j<=E->sphere.noy;j++)   { */
/*        node = i+(j-1)*E->sphere.nox */;

             /* first find which cap this node (i,j) is in  */
/*        t = E->sphere.sx[1][node]; */
/*        f = E->sphere.sx[2][node]; */

/*        x[1] = sin(t)*cos(f);  */ /* radius does not matter */
/*        x[2] = sin(t)*sin(f); */
/*        x[3] = cos(t); */

       /* locate_cap may not work correctly after my change in numbering of caps */
       /* but sphere.int_cap and int_ele are not used anywhere */
/*        m = locate_cap(E,x); */
/*        if (m>0)  { */
/*           el = locate_element(E,m,x,node); */        /* bottom element */

/*           if (el<=0)    { */
/*             fprintf(stderr,"!!! Processor %d cannot find the right element in cap %d\n",E->parallel.me,m); */
/*             parallel_process_termination(); */
/*             } */

/*           E->sphere.int_cap[node]=m; */
/*           E->sphere.int_ele[node]=el; */

/*           } */
/*      }     */    /* end for i and j */
  
/*    parallel_process_sync(); */

  return;
  }

/* ================================================
  locate the cap for node (i,j)
 ================================================*/

int locate_cap(E,x)
 struct All_variables *E;
 double x[4];
 {

  int ia[5],i,m,mm;
  double xx[4],angle[5],angle1[5];
  double get_angle();
  double area1,rr;
  const double e_7=1.e-7;

  mm = 0;

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ia[1] = 1;
    ia[2] = E->lmesh.noz*E->lmesh.nox-E->lmesh.noz+1;
    ia[3] = E->lmesh.nno-E->lmesh.noz+1;
    ia[4] = ia[3]-E->lmesh.noz*(E->lmesh.nox-1);

    for (i=1;i<=4;i++)  {
      xx[1] = E->x[m][1][ia[i]]/E->sx[m][3][ia[1]];
      xx[2] = E->x[m][2][ia[i]]/E->sx[m][3][ia[1]];
      xx[3] = E->x[m][3][ia[i]]/E->sx[m][3][ia[1]];
      angle[i] = get_angle(x,xx);    /* get angle between (i,j) and other four*/
      angle1[i]=E->sphere.angle[m][i];
      }

    area1 = area_of_sphere_triag(angle[1],angle[2],angle1[1])
          + area_of_sphere_triag(angle[2],angle[3],angle1[2])
          + area_of_sphere_triag(angle[3],angle[4],angle1[3])
          + area_of_sphere_triag(angle[4],angle[1],angle1[4]);

    if ( fabs ((area1-E->sphere.area[m])/E->sphere.area[m]) <e_7 ) {
        mm = m;
        return (mm);
        }
    }

 return (mm);
 }

/* ================================================
  locate the element containing the node (i,j) with coord x. 
  The radius is assumed to be 1 in computing the areas.
  NOTE:  The returned element el is for the bottom layer. 
 ================================================*/

int locate_element(E,m,x,ne)
 struct All_variables *E;
 double x[4];
 int m,ne;
 {

  int el_temp,el,es,el_located,level,lev,lev_plus,el_plus,es_plus,i,j,node;
  double area,area1,areamin;
  double area_of_5points();
  const double e_7=1.e-7;
  const double e_6=1.e6;
  
  el_located = 0;


  level=E->mesh.levmin;
  for (es=1;es<=E->lmesh.SNEL[level];es++)              {

    el = (es-1)*E->lmesh.ELZ[level]+1;
    area1 = area_of_5points (E,level,m,el,x,ne);
    area = E->sphere.area1[level][m][es];

    if(fabs ((area1-area)/area) <e_7 ) {
       for (lev=E->mesh.levmin;lev<E->mesh.levmax;lev++)  {
          lev_plus = lev + 1;
	  j=1;
	  areamin = e_6;
	  do {
             el_plus = E->EL[lev][m][el].sub[j];
               
             es_plus = (el_plus-1)/E->lmesh.ELZ[lev_plus]+1;

             area1 = area_of_5points(E,lev_plus,m,el_plus,x,ne);
             area = E->sphere.area1[lev_plus][m][es_plus];

	     if(fabs(area1-area)<areamin) {
		 areamin=fabs(area1-area);
		 el_temp = el_plus;
		 }
	     j++;
             }  while (j<5 && fabs((area1-area)/area) > e_7);
          el = el_plus;
                          /* if exit with ..>e_7, pick the best one*/
	  if (fabs((area1-area)/area) > e_7) el = el_temp;
          }      /* end for loop lev         */
       el_located = el;
       }    /* end for if */

     if(el_located)  break;
     }    /* end for es at the coarsest level  */

 return (el_located);
 }

/* ===============================================================
  interpolate nodal T's within cap m and element el onto node with
  coordinate x[3] which is derived from a regular mesh and within 
  the element el. NOTE the radius of x[3] is the inner radius.
 =============================================================== */
 
float sphere_interpolate_point(E,T,m,el,x,ne)
 struct All_variables *E;
 float **T;
 double x[4];
 int m,el,ne;
 {
/*  double to,fo,y[4],yy[4][5],dxdy[4][4]; */
/*  double a1,b1,c1,d1,a2,b2,c2,d2,a,b,c,xx1,yy1,y1,y2; */
 float ta,t[5];
/*  int es,snode,i,j,node; */

/*  const int oned=4; */
/*  const double e_7=1.e-7; */
/*  const double four=4.0; */
/*  const double two=2.0; */
/*  const double one=1.0; */
/*  const double pt25=0.25; */

       /* first rotate the coord such that the center of element is
           the pole   */

/*   es = (el-1)/E->lmesh.elz+1; */

/*   to = E->eco[m][el].centre[1]; */
/*   fo = E->eco[m][el].centre[2]; */
 
/*   dxdy[1][1] = cos(to)*cos(fo); */
/*   dxdy[1][2] = cos(to)*sin(fo); */
/*   dxdy[1][3] = -sin(to); */
/*   dxdy[2][1] = -sin(fo); */
/*   dxdy[2][2] = cos(fo); */
/*   dxdy[2][3] = 0.0; */
/*   dxdy[3][1] = sin(to)*cos(fo); */
/*   dxdy[3][2] = sin(to)*sin(fo); */
/*   dxdy[3][3] = cos(to); */

/*   for(i=1;i<=oned;i++) {  */        /* nodes */
/*      node = E->ien[m][el].node[i]; */
/*      snode = E->sien[m][es].node[i]; */
/*      t[i] = T[m][snode]; */
/*      for (j=1;j<=E->mesh.nsd;j++)  */
/*        yy[j][i] = E->x[m][1][node]*dxdy[j][1] */
/*                 + E->x[m][2][node]*dxdy[j][2] */
/*                 + E->x[m][3][node]*dxdy[j][3]; */
/*      } */

/*   for (j=1;j<=E->mesh.nsd;j++)  */
/*      y[j] = x[1]*dxdy[j][1] + x[2]*dxdy[j][2] + x[3]*dxdy[j][3]; */

       /* then for node y, determine its coordinates xx1,yy1 
        in the parental element in the isoparametric element system*/
 
/*   a1 = yy[1][1] + yy[1][2] + yy[1][3] + yy[1][4]; */
/*   b1 = yy[1][3] + yy[1][2] - yy[1][1] - yy[1][4]; */
/*   c1 = yy[1][3] + yy[1][1] - yy[1][2] - yy[1][4]; */
/*   d1 = yy[1][3] + yy[1][4] - yy[1][1] - yy[1][2]; */
/*   a2 = yy[2][1] + yy[2][2] + yy[2][3] + yy[2][4]; */
/*   b2 = yy[2][3] + yy[2][2] - yy[2][1] - yy[2][4]; */
/*   c2 = yy[2][3] + yy[2][1] - yy[2][2] - yy[2][4]; */
/*   d2 = yy[2][3] + yy[2][4] - yy[2][1] - yy[2][2]; */

/*   a = d2*c1; */
/*   b = a2*c1+b1*d2-d1*c2-d1*b2-four*c1*y[2]; */
/*   c=four*c2*y[1]-c2*a1-a1*b2+four*b2*y[1]-four*b1*y[2]+a2*b1; */

/*   if (fabs(a)<e_7)  { */
/*       yy1 = -c/b; */
/*       xx1 = (four*y[1]-a1-d1*yy1)/(b1+c1*yy1); */
/*       } */
/*   else  { */
/*       y1= (-b+sqrt(b*b-four*a*c))/(two*a); */
/*       y2= (-b-sqrt(b*b-four*a*c))/(two*a); */
/*       if (fabs(y1)>fabs(y2)) */
/*          yy1 = y2; */
/*       else */
/*          yy1 = y1; */
/*       xx1 = (four*y[1]-a1-d1*yy1)/(b1+c1*yy1); */
/*       } */

     /* now we can calculate T at x[4] using shape function */

/*     ta = ((one-xx1)*(one-yy1)*t[1]+(one+xx1)*(one-yy1)*t[2]+ */
/*           (one+xx1)*(one+yy1)*t[3]+(one-xx1)*(one+yy1)*t[4])*pt25; */

/*if(fabs(xx1)>1.5 || fabs(yy1)>1.5)fprintf(E->fp_out,"ME= %d %d %d %g %g %g %g %g %g %g\n",ne,m,es,t[1],t[2],t[3],t[4],ta,xx1,yy1);
 */
 return (ta);
 }

/* ===================================================================
  do the interpolation on sphere for data T, which is needed for both
  spherical harmonic expansion and graphics 
 =================================================================== */

void sphere_interpolate(E,T,TG)
    struct All_variables *E;
    float **T,*TG;
 {

/*    float sphere_interpolate_point(); */
/*    void gather_TG_to_me0(); */
/*    void parallel_process_termination(); */

/*    int ii,jj,es,i,j,m,el,node; */
/*    double x[4],t,f; */

/*    const int ends=4; */

/*    for (i=1;i<=E->sphere.nox;i++) */
/*      for (j=1;j<=E->sphere.noy;j++)   { */
/*        node = i+(j-1)*E->sphere.nox; */
/*        TG[node] = 0.0; */
             /* first find which cap this node (i,j) is in  */

/*        m = E->sphere.int_cap[node]; */
/*        el = E->sphere.int_ele[node]; */

/*        if (m>0 && el>0)   { */
/*           t = E->sphere.sx[1][node]; */
/*           f = E->sphere.sx[2][node]; */

/*           x[1] = E->sx[1][3][1]*sin(t)*cos(f); */
/*           x[2] = E->sx[1][3][1]*sin(t)*sin(f); */
/*           x[3] = E->sx[1][3][1]*cos(t); */

/*           TG[node] = sphere_interpolate_point(E,T,m,el,x,node); */

/*           } */

/*    }  */

/*      gather_TG_to_me0(E,TG); */

 return;
 }

/* =========================================================
  ========================================================= */
void sphere_expansion(E,TG,sphc,sphs)
 struct All_variables *E;
 float *TG,*sphc,*sphs;
 {
/*  int p,i,j,es,mm,ll,rand(); */
/*  double temp,area,t,f,sphere_h(); */
/*  const double pt25=0.25; */
/*  static int been_here=0; */
/*  void sum_across_surf_sph1(); */
/*  void sum_across_surf_sph(); */

/*    for (i=0;i<E->sphere.hindice;i++)    { */
/*       sphc[i] = 0.0; */
/*       sphs[i] = 0.0; */
/*       } */

/*    area = 2.0*M_PI*M_PI/(E->sphere.elx*E->sphere.ely); */

/*    for (ll=0;ll<=E->sphere.output_llmax;ll++) */
/*      for (mm=0; mm<=ll; mm++)   { */

/*        p = E->sphere.hindex[ll][mm]; */

/*        for (j=1;j<=E->sphere.lely;j++) */
/* 	 for (i=1;i<=E->sphere.lelx;i++) { */

/* 	   es = i+(j-1)*E->sphere.lelx; */

/*            temp=pt25*(TG[E->sphere.sien[es].node[1]] */
/* 		     +TG[E->sphere.sien[es].node[2]]  */
/* 		     +TG[E->sphere.sien[es].node[3]]  */
/* 		     +TG[E->sphere.sien[es].node[4]]);  */

/*            sphc[p]+=temp*E->sphere.tableplm[i][p]*E->sphere.tablecosf[j][mm]*E->sphere.tablesint[i];  */
/*            sphs[p]+=temp*E->sphere.tableplm[i][p]*E->sphere.tablesinf[j][mm]*E->sphere.tablesint[i];  */

/* 	 }   */

/* 	 sphc[p] *= area;  */
/* 	 sphs[p] *= area;  */

/*        }     */   /* end for ll and mm  */

/*     sum_across_surf_sph1(E,sphc,sphs); */

 return;
 }

 /* =========================================================== */
void inv_sphere_harmonics(E,sphc,sphs,TG,proc_loc)
 struct All_variables *E;
 float *TG,*sphc,*sphs;
 int proc_loc;
 {
/*  int k,ll,mm,node,i,j,p,noz,snode; */
/*  float t1,f1,rad; */
/*  void parallel_process_sync(); */
/*  void gather_TG_to_me0(); */

/*  if (E->parallel.me_loc[3]==proc_loc)   { */

/*    for (j=1;j<=E->sphere.noy;j++)   */
/*      for (i=1;i<=E->sphere.nox;i++)  { */
/*        node = i + (j-1)*E->sphere.nox; */
/*        TG[node]=0.0; */
/*        } */

/*    for (ll=0;ll<=E->sphere.output_llmax;ll++) */
/*    for (mm=0;mm<=ll;mm++)   { */

/*      p = E->sphere.hindex[ll][mm]; */

/*      for (i=1;i<=E->sphere.lnox;i++) */
/*      for (j=1;j<=E->sphere.lnoy;j++)  { */
/*        node = i + E->sphere.lexs + (j+E->sphere.leys-1)*E->sphere.nox; */
/*        TG[node] += */
/* 	  (sphc[p]*E->sphere.tableplm_n[i][p]*E->sphere.tablecosf_n[j][mm] */
/* 	  +sphs[p]*E->sphere.tableplm_n[i][p]*E->sphere.tablesinf_n[j][mm]); */
/*        } */
/*      } */

/*    gather_TG_to_me0(E,TG); */

/*   } */

/*   parallel_process_sync(); */

 return;
 }

/* ==================================================*/
void  compute_sphereh_table(E)
struct All_variables *E;
{

int rr,node,ends,ll,mm,es,i,j,p;
double t,f,plgndr_a(),modified_plgndr_a();
const double pt25=0.25;

 ends = 4;

/*   for (j=1;j<=E->sphere.lely;j++) */
/*   for (i=1;i<=E->sphere.lelx;i++) { */
/*     es = i+(j-1)*E->sphere.lelx; */
/*     node = E->sphere.lexs + i + (E->sphere.leys+j-1)*E->sphere.nox; */
/*     for (rr=1;rr<=ends;rr++) */
/*       E->sphere.sien[es].node[rr] = node  */
/* 		       + offset[rr].vector[1] */
/* 		       + offset[rr].vector[2]*E->sphere.nox; */
/*     } */

/*   for (j=1;j<=E->sphere.lely;j++) { */
/*     es = 1+(j-1)*E->sphere.lelx; */
/*     f=pt25*(E->sphere.sx[2][E->sphere.sien[es].node[1]] */
/* 	  +E->sphere.sx[2][E->sphere.sien[es].node[2]]  */
/* 	  +E->sphere.sx[2][E->sphere.sien[es].node[3]]  */
/* 	  +E->sphere.sx[2][E->sphere.sien[es].node[4]]);  */
/*     for (mm=0;mm<=E->sphere.output_llmax;mm++)   { */
/*        E->sphere.tablecosf[j][mm] = cos( (double)(mm)*f ); */
/*        E->sphere.tablesinf[j][mm] = sin( (double)(mm)*f ); */
/*        } */
/*     } */
      
/*   for (i=1;i<=E->sphere.lelx;i++) { */
/*     es = i+(1-1)*E->sphere.lelx; */
/*     t=pt25*(E->sphere.sx[1][E->sphere.sien[es].node[1]] */
/* 	  +E->sphere.sx[1][E->sphere.sien[es].node[2]]  */
/* 	  +E->sphere.sx[1][E->sphere.sien[es].node[3]]  */
/* 	  +E->sphere.sx[1][E->sphere.sien[es].node[4]]);  */
/*     E->sphere.tablesint[i] = sin(t); */
/*     for (ll=0;ll<=E->sphere.output_llmax;ll++) */
/*       for (mm=0;mm<=ll;mm++)  { */
/*          p = E->sphere.hindex[ll][mm]; */
/*          E->sphere.tableplm[i][p] = modified_plgndr_a(ll,mm,t) ; */
/*          } */
/*     }  */


/*   for (j=1;j<=E->sphere.lnoy;j++) { */
/*     node = E->sphere.lexs + 1 + (E->sphere.leys+j-1)*E->sphere.nox; */
/*     f=E->sphere.sx[2][node]; */
/*     for (mm=0;mm<=E->sphere.output_llmax;mm++)   { */
/*        E->sphere.tablecosf_n[j][mm] = cos( (double)(mm)*f ); */
/*        E->sphere.tablesinf_n[j][mm] = sin( (double)(mm)*f ); */
/*        } */
/*     } */
      
/*   for (i=1;i<=E->sphere.lnox;i++) { */
/*     node = E->sphere.lexs + i + (E->sphere.leys+1-1)*E->sphere.nox; */
/*     t=E->sphere.sx[1][node]; */
/*     for (ll=0;ll<=E->sphere.output_llmax;ll++) */
/*       for (mm=0;mm<=ll;mm++)  { */
/*          p = E->sphere.hindex[ll][mm]; */
/*          E->sphere.tableplm_n[i][p] = modified_plgndr_a(ll,mm,t) ; */
/*          } */
/*     }  */

 return;
 }
