#include "element_definitions.h"
#include "global_defs.h"
#include <math.h>

#include "lith_age.h"
/* ========================================== */

void velocity_boundary_conditions(E)
     struct All_variables *E;
{
  void velocity_imp_vert_bc();
  void velocity_refl_vert_bc();
  void horizontal_bc();
  void velocity_apply_periodic_bcs();
  void read_velocity_boundary_from_file(); 
  void renew_top_velocity_boundary();

  int node,d,j,noz,lv;
 
  for(lv=E->mesh.gridmax;lv>=E->mesh.gridmin;lv--)
    for (j=1;j<=E->sphere.caps_per_proc;j++)     {
      noz = E->lmesh.NOZ[lv];
      if(E->mesh.topvbc != 1) {
	horizontal_bc(E,E->sphere.cap[j].VB,noz,1,0.0,VBX,0,lv,j);	 
	horizontal_bc(E,E->sphere.cap[j].VB,noz,3,0.0,VBZ,1,lv,j);
	horizontal_bc(E,E->sphere.cap[j].VB,noz,2,0.0,VBY,0,lv,j);	 
	horizontal_bc(E,E->sphere.cap[j].VB,noz,1,E->control.VBXtopval,SBX,1,lv,j);
	horizontal_bc(E,E->sphere.cap[j].VB,noz,3,0.0,SBZ,0,lv,j);
	horizontal_bc(E,E->sphere.cap[j].VB,noz,2,E->control.VBYtopval,SBY,1,lv,j);
	}
      if(E->mesh.botvbc != 1) {
        horizontal_bc(E,E->sphere.cap[j].VB,1,1,0.0,VBX,0,lv,j); 
        horizontal_bc(E,E->sphere.cap[j].VB,1,3,0.0,VBZ,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,2,0.0,VBY,0,lv,j);	 
        horizontal_bc(E,E->sphere.cap[j].VB,1,1,E->control.VBXbotval,SBX,1,lv,j); 
        horizontal_bc(E,E->sphere.cap[j].VB,1,3,0.0,SBZ,0,lv,j);  
        horizontal_bc(E,E->sphere.cap[j].VB,1,2,E->control.VBYbotval,SBY,1,lv,j); 
        } 

      if(E->mesh.topvbc == 1) {
        horizontal_bc(E,E->sphere.cap[j].VB,noz,1,E->control.VBXtopval,VBX,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,noz,3,0.0,VBZ,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,noz,2,E->control.VBYtopval,VBY,1,lv,j); 
        horizontal_bc(E,E->sphere.cap[j].VB,noz,1,0.0,SBX,0,lv,j);	 
        horizontal_bc(E,E->sphere.cap[j].VB,noz,3,0.0,SBZ,0,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,noz,2,0.0,SBY,0,lv,j);	 

        if(E->control.vbcs_file)   {
          read_velocity_boundary_from_file(E);   /* read in the velocity boundary
condition from file if E->control.bcs_file==1   */
          }
        else
          {

/* commented out 2/26/00 CPC - remove program specification of velocity BC
           renew_top_velocity_boundary(E);
*/
          }

        }


      if(E->mesh.botvbc == 1) {
        horizontal_bc(E,E->sphere.cap[j].VB,1,1,E->control.VBXbotval,VBX,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,3,0.0,VBZ,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,2,E->control.VBYbotval,VBY,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,1,0.0,SBX,0,lv,j);	 
        horizontal_bc(E,E->sphere.cap[j].VB,1,3,0.0,SBZ,0,lv,j); 
        horizontal_bc(E,E->sphere.cap[j].VB,1,2,0.0,SBY,0,lv,j);	 
        }
      }    /* end for j and lv */

      velocity_refl_vert_bc(E);

if(E->control.verbose)
 for (j=1;j<=E->sphere.caps_per_proc;j++)
   for (node=1;node<=E->lmesh.nno;node++)
      fprintf(E->fp_out,"m=%d VB== %d %g %g %g flag %u %u %u\n",j,node,E->sphere.cap[j].VB[1][node],E->sphere.cap[j].VB[2][node],E->sphere.cap[j].VB[3][node],E->node[j][node]&VBX,E->node[j][node]&VBY,E->node[j][node]&VBZ);

  /* If any imposed internal velocity structure it goes here */

 
   return; }

/* ========================================== */

void temperature_boundary_conditions(E)
     struct All_variables *E;
{
  void temperatures_conform_bcs();
  void horizontal_bc();
  void temperature_apply_periodic_bcs();
  void temperature_imposed_vert_bcs();
  void temperature_refl_vert_bc();
  void temperature_lith_adj();
  int j,lev,noz;
  
  lev = E->mesh.levmax;


     temperature_refl_vert_bc(E);

  for (j=1;j<=E->sphere.caps_per_proc;j++)    {
    noz = E->lmesh.noz;
    if(E->mesh.toptbc == 1)    {
      horizontal_bc(E,E->sphere.cap[j].TB,noz,3,E->control.TBCtopval,TBZ,1,lev,j);
      horizontal_bc(E,E->sphere.cap[j].TB,noz,3,E->control.TBCtopval,FBZ,0,lev,j);
      }
    else   {
      horizontal_bc(E,E->sphere.cap[j].TB,noz,3,E->control.TBCtopval,TBZ,0,lev,j);
      horizontal_bc(E,E->sphere.cap[j].TB,noz,3,E->control.TBCtopval,FBZ,1,lev,j);
      }
 
    if(E->mesh.bottbc == 1)    {
      horizontal_bc(E,E->sphere.cap[j].TB,1,3,E->control.TBCbotval,TBZ,1,lev,j);
      horizontal_bc(E,E->sphere.cap[j].TB,1,3,E->control.TBCbotval,FBZ,0,lev,j);
      }
    else        {
      horizontal_bc(E,E->sphere.cap[j].TB,1,3,E->control.TBCbotval,TBZ,0,lev,j);
      horizontal_bc(E,E->sphere.cap[j].TB,1,3,E->control.TBCbotval,FBZ,1,lev,j); 
      }
 
    if((E->control.temperature_bound_adj==1) || (E->control.lith_age_time==1))  {
/* set the regions in which to use lithosphere files to determine temperature 
   note that this is called if the lithosphere age in inputted every time step
   OR it is only maintained in the boundary regions */
      lith_age_temperature_bound_adj(E,lev);
    }

    temperatures_conform_bcs(E);


    }     /* end for j */

   return; }

/* ========================================== */

void velocity_refl_vert_bc(E)
     struct All_variables *E;
{
  int m,i,j,ii,jj;
  int node1,node2;
  int level,nox,noy,noz;
  const int dims=E->mesh.nsd;

 /*  for two YOZ planes   */


  if (E->parallel.me_locl[1]==0 || E->parallel.me_locl[1]==E->parallel.nprocxl-1)  
   for (m=1;m<=E->sphere.caps_per_proc;m++)  
    for(j=1;j<=E->lmesh.noy;j++)
      for(i=1;i<=E->lmesh.noz;i++)  {
        node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
        node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;

        ii = i + E->lmesh.nzs - 1;
        if (E->parallel.me_locl[1]==0 )  {
           E->sphere.cap[m].VB[1][node1] = 0.0;
           if((ii != 1) && (ii != E->mesh.noz))
              E->sphere.cap[m].VB[3][node1] = 0.0;
               }
        if (E->parallel.me_locl[1]==E->parallel.nprocxl-1)  {
           E->sphere.cap[m].VB[1][node2] = 0.0;
           if((ii != 1) && (ii != E->mesh.noz))
              E->sphere.cap[m].VB[3][node2] = 0.0;
           }
        }      /* end loop for i and j */

/*  for two XOZ  planes  */


    if (E->parallel.me_locl[2]==0)
     for (m=1;m<=E->sphere.caps_per_proc;m++)  
      for(j=1;j<=E->lmesh.nox;j++)
        for(i=1;i<=E->lmesh.noz;i++)       {
          node1 = i + (j-1)*E->lmesh.noz;
          ii = i + E->lmesh.nzs - 1;

          E->sphere.cap[m].VB[2][node1] = 0.0;
          if((ii != 1) && (ii != E->mesh.noz))
            E->sphere.cap[m].VB[3][node1] = 0.0;
          }    /* end of loop i & j */

    if (E->parallel.me_locl[2]==E->parallel.nprocyl-1)
     for (m=1;m<=E->sphere.caps_per_proc;m++)  
      for(j=1;j<=E->lmesh.nox;j++)
        for(i=1;i<=E->lmesh.noz;i++)       {
          node2 = (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox + i + (j-1)*E->lmesh.noz;
          ii = i + E->lmesh.nzs - 1;

          E->sphere.cap[m].VB[2][node2] = 0.0;
          if((ii != 1) && (ii != E->mesh.noz))
            E->sphere.cap[m].VB[3][node2] = 0.0;
          }    /* end of loop i & j */


  /* all vbc's apply at all levels  */
  for(level=E->mesh.levmax;level>=E->mesh.levmin;level--) {

    if ( (E->control.CONJ_GRAD && level==E->mesh.levmax) ||E->control.NMULTIGRID)  {
    noz = E->lmesh.NOZ[level] ;
    noy = E->lmesh.NOY[level] ;
    nox = E->lmesh.NOX[level] ;

     for (m=1;m<=E->sphere.caps_per_proc;m++)  { 
       if (E->parallel.me_locl[1]==0 || E->parallel.me_locl[1]==E->parallel.nprocxl-1) {
         for(j=1;j<=noy;j++)
          for(i=1;i<=noz;i++) {
          node1 = i + (j-1)*noz*nox;
          node2 = node1 + (nox-1)*noz;
          ii = i + E->lmesh.NZS[level] - 1;
          if (E->parallel.me_locl[1]==0 )  {
            E->NODE[level][m][node1] = E->NODE[level][m][node1] | VBX;
            E->NODE[level][m][node1] = E->NODE[level][m][node1] & (~SBX);
            if((ii!=1) && (ii!=E->mesh.NOZ[level])) {
               E->NODE[level][m][node1] = E->NODE[level][m][node1] & (~VBY);
               E->NODE[level][m][node1] = E->NODE[level][m][node1] | SBY;
               E->NODE[level][m][node1] = E->NODE[level][m][node1] & (~ VBZ);
               E->NODE[level][m][node1] = E->NODE[level][m][node1] | SBZ;
               }
            }
          if (E->parallel.me_locl[1]==E->parallel.nprocxl-1)  {
            E->NODE[level][m][node2] = E->NODE[level][m][node2] | VBX;
            E->NODE[level][m][node2] = E->NODE[level][m][node2] & (~SBX);
            if((ii!=1) && (ii!=E->mesh.NOZ[level])) {
              E->NODE[level][m][node2] = E->NODE[level][m][node2] & (~VBY);
              E->NODE[level][m][node2] = E->NODE[level][m][node2] | SBY;
              E->NODE[level][m][node2] = E->NODE[level][m][node2] & (~ VBZ);
              E->NODE[level][m][node2] = E->NODE[level][m][node2] | SBZ;
                  }
            }
          }   /* end for loop i & j */

         }


      if (E->parallel.me_locl[2]==0)  
        for(j=1;j<=nox;j++)
          for(i=1;i<=noz;i++) {
            node1 = i + (j-1)*noz;
            ii = i + E->lmesh.NZS[level] - 1;
            jj = j + E->lmesh.NXS[level] - 1;

            E->NODE[level][m][node1] = E->NODE[level][m][node1] | VBY;
            E->NODE[level][m][node1] = E->NODE[level][m][node1] & (~SBY);
            if((ii!= 1) && (ii != E->mesh.NOZ[level]))  {
                E->NODE[level][m][node1] = E->NODE[level][m][node1] & (~VBZ);
                E->NODE[level][m][node1] = E->NODE[level][m][node1] | SBZ;
                }
            if((jj!=1) && (jj!=E->mesh.NOX[level]) && (ii!=1) && (ii!=E->mesh.NOZ[level])){
                E->NODE[level][m][node1] = E->NODE[level][m][node1] & (~VBX);
                E->NODE[level][m][node1] = E->NODE[level][m][node1] | SBX;
                }
                }    /* end for loop i & j  */

      if (E->parallel.me_locl[2]==E->parallel.nprocyl-1)  
        for(j=1;j<=nox;j++)
          for(i=1;i<=noz;i++)       {
            node2 = (noy-1)*noz*nox + i + (j-1)*noz;
            ii = i + E->lmesh.NZS[level] - 1;
            jj = j + E->lmesh.NXS[level] - 1;
            E->NODE[level][m][node2] = E->NODE[level][m][node2] | VBY;
            E->NODE[level][m][node2] = E->NODE[level][m][node2] & (~SBY);
            if((ii!= 1) && (ii != E->mesh.NOZ[level]))  {
                E->NODE[level][m][node2] = E->NODE[level][m][node2] & (~VBZ);
                E->NODE[level][m][node2] = E->NODE[level][m][node2] | SBZ;
                }
            if((jj!=1) && (jj!=E->mesh.NOX[level]) && (ii!=1) && (ii!=E->mesh.NOZ[level])){
                E->NODE[level][m][node2] = E->NODE[level][m][node2] & (~VBX);
                E->NODE[level][m][node2] = E->NODE[level][m][node2] | SBX;
                }
            }

       }       /* end for m  */
       }
       }       /*  end for loop level  */

  return;
}

void temperature_refl_vert_bc(E)
     struct All_variables *E;
{
  int i,j,m;
  int node1,node2;
  const int dims=E->mesh.nsd;

 /* Temps and bc-values  at top level only */

   if (E->parallel.me_locl[1]==0 || E->parallel.me_locl[1]==E->parallel.nprocxl-1)
    for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(j=1;j<=E->lmesh.noy;j++)
      for(i=1;i<=E->lmesh.noz;i++) {
        node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
        node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;
        if (E->parallel.me_locl[1]==0 )                   {
          E->node[m][node1] = E->node[m][node1] & (~TBX);
          E->node[m][node1] = E->node[m][node1] | FBX;
          E->sphere.cap[m].TB[1][node1] = 0.0;
              }
        if (E->parallel.me_locl[1]==E->parallel.nprocxl-1)   {
          E->node[m][node2] = E->node[m][node2] & (~TBX);
          E->node[m][node2] = E->node[m][node2] | FBX;
          E->sphere.cap[m].TB[1][node2] = 0.0;
              }
        }       /* end for loop i & j */

    if (E->parallel.me_locl[2]==0)
     for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(j=1;j<=E->lmesh.nox;j++)
        for(i=1;i<=E->lmesh.noz;i++) {
          node1 = i + (j-1)*E->lmesh.noz;
          E->node[m][node1] = E->node[m][node1] & (~TBY);
              E->node[m][node1] = E->node[m][node1] | FBY;
              E->sphere.cap[m].TB[2][node1] = 0.0;
              }

    if (E->parallel.me_locl[2]==E->parallel.nprocyl-1)
     for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(j=1;j<=E->lmesh.nox;j++)
        for(i=1;i<=E->lmesh.noz;i++) {
          node2 = i +(j-1)*E->lmesh.noz + (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox;
          E->node[m][node2] = E->node[m][node2] & (~TBY);
          E->node[m][node2] = E->node[m][node2] | FBY;
          E->sphere.cap[m].TB[3][node2] = 0.0;
          }    /* end loop for i and j */

  return;
}


/*  =========================================================  */
    

void horizontal_bc(E,BC,ROW,dirn,value,mask,onoff,level,m)
     struct All_variables *E;
     float *BC[];
     int ROW;
     int dirn;
     float value;
     unsigned int mask;
     char onoff;
     int level,m;

{
  int i,j,node,rowl;
  const int dims=E->mesh.nsd;

    /* safety feature */
  if(dirn > E->mesh.nsd) 
     return;

  if (ROW==1) 
      rowl = 1;
  else 
      rowl = E->lmesh.NOZ[level];
   
  if ( (ROW==1 && E->parallel.me_locl[3]==0) ||
       (ROW==E->lmesh.NOZ[level] && E->parallel.me_locl[3]==E->parallel.nproczl-1) ) {

    /* turn bc marker to zero */
    if (onoff == 0)          {
      for(j=1;j<=E->lmesh.NOY[level];j++)
    	for(i=1;i<=E->lmesh.NOX[level];i++)     {
    	  node = rowl+(i-1)*E->lmesh.NOZ[level]+(j-1)*E->lmesh.NOX[level]*E->lmesh.NOZ[level];
    	  E->NODE[level][m][node] = E->NODE[level][m][node] & (~ mask);
    	  }        /* end for loop i & j */
      }

    /* turn bc marker to one */    
    else        {
      for(j=1;j<=E->lmesh.NOY[level];j++)
        for(i=1;i<=E->lmesh.NOX[level];i++)       {
    	  node = rowl+(i-1)*E->lmesh.NOZ[level]+(j-1)*E->lmesh.NOX[level]*E->lmesh.NOZ[level];
    	  E->NODE[level][m][node] = E->NODE[level][m][node] | (mask);

    	  if(level==E->mesh.levmax)   /* NB */
    	    BC[dirn][node] = value;   
    	  }     /* end for loop i & j */
      }

    }             /* end for if ROW */
    
  return;
}


void velocity_apply_periodic_bcs(E)
    struct All_variables *E;
{
  int n1,n2,level;
  int i,j,ii,jj;
  const int dims=E->mesh.nsd;

  fprintf(E->fp,"Periodic boundary conditions\n");

  return;
  }

void temperature_apply_periodic_bcs(E)
    struct All_variables *E;
{
 const int dims=E->mesh.nsd;

 fprintf(E->fp,"pERIodic temperature boundary conditions\n");
   
  return;
  }



void strip_bcs_from_residual(E,Res,level)
    struct All_variables *E;
    double **Res;
    int level;
{
    int m,i;
     
  for (m=1;m<=E->sphere.caps_per_proc;m++)
    if (E->num_zero_resid[level][m])
      for(i=1;i<=E->num_zero_resid[level][m];i++)
         Res[m][E->zero_resid[level][m][i]] = 0.0;

    return;
    }


void get_bcs_id_for_residual(E,level,m)
    struct All_variables *E;
    int level,m;
  {

    int i,j;
     
    const int dims=E->mesh.nsd,dofs=E->mesh.dof;
    const int nno=E->lmesh.NNO[level];
    const int neq=E->lmesh.NEQ[level];
    const int addi_dof=additional_dof[dims];

   j = 0;
   for(i=1;i<=nno;i++) { 
      if ( (E->NODE[level][m][i] & VBX) != 0 )  {
	j++;
        E->zero_resid[level][m][j] = E->ID[level][m][i].doff[1];
	}
      if ( (E->NODE[level][m][i] & VBY) != 0 )  {
	j++;
        E->zero_resid[level][m][j] = E->ID[level][m][i].doff[2];
	}
      if ( (E->NODE[level][m][i] & VBZ) != 0 )  {
	j++;
        E->zero_resid[level][m][j] = E->ID[level][m][i].doff[3];
	}
      }

    E->num_zero_resid[level][m] = j;
 
    return;
}
    
void temperatures_conform_bcs(struct All_variables *E)
{
  void temperatures_conform_bcs2(struct All_variables *);
  if(E->control.lith_age)
    lith_age_conform_tbc(E);

  temperatures_conform_bcs2(E);
  return;
}


void temperatures_conform_bcs2(struct All_variables *E)
{
  int m,j,nno,node,nox,noz,noy,gnox,gnoy,gnoz,nodeg,ii,i,k;
  unsigned int type;

  nno=E->lmesh.nno;
  gnox=E->mesh.nox;
  gnoy=E->mesh.noy;
  gnoz=E->mesh.noz;
  nox=E->lmesh.nox;
  noy=E->lmesh.noy;
  noz=E->lmesh.noz;

  for(j=1;j<=E->sphere.caps_per_proc;j++)
    for(node=1;node<=E->lmesh.nno;node++)  {

        type = (E->node[j][node] & (TBX | TBZ | TBY));

        switch (type) {
        case 0:  /* no match, next node */
            break;
        case TBX:
            E->T[j][node] = E->sphere.cap[j].TB[1][node];
            break;
        case TBZ:
            E->T[j][node] = E->sphere.cap[j].TB[3][node];
            break;
        case TBY:
            E->T[j][node] = E->sphere.cap[j].TB[2][node];
            break;
        case (TBX | TBZ):     /* clashes ! */
            E->T[j][node] = 0.5 * (E->sphere.cap[j].TB[1][node] + E->sphere.cap[j].TB[3][node]);
            break;
        case (TBX | TBY):     /* clashes ! */
            E->T[j][node] = 0.5 * (E->sphere.cap[j].TB[1][node] + E->sphere.cap[j].TB[2][node]);
            break;
        case (TBZ | TBY):     /* clashes ! */
            E->T[j][node] = 0.5 * (E->sphere.cap[j].TB[3][node] + E->sphere.cap[j].TB[2][node]);
            break;
        case (TBZ | TBY | TBX):     /* clashes ! */
            E->T[j][node] = 0.3333333 * (E->sphere.cap[j].TB[1][node] + E->sphere.cap[j].TB[2][node] + E->sphere.cap[j].TB[3][node]);
            break;
        } /* end switch */

        
    } /* next node */

return;

 }


void velocities_conform_bcs(E,U)
    struct All_variables *E;
    double **U;
{ 
    int node,d,m;

    const unsigned int typex = VBX;
    const unsigned int typez = VBZ;
    const unsigned int typey = VBY;
    const int addi_dof = additional_dof[E->mesh.nsd];

    const int dofs = E->mesh.dof;
    const int nno = E->lmesh.nno;

    for(m=1;m<=E->sphere.caps_per_proc;m++)   {
      for(node=1;node<=nno;node++) {

        if (E->node[m][node] & typex)  
	      U[m][E->id[m][node].doff[1]] = E->sphere.cap[m].VB[1][node]; 
 	if (E->node[m][node] & typey)  
	      U[m][E->id[m][node].doff[2]] = E->sphere.cap[m].VB[2][node]; 
	if (E->node[m][node] & typez)  
	      U[m][E->id[m][node].doff[3]] = E->sphere.cap[m].VB[3][node]; 
        } 
      } 

    return;
}

 void renew_top_velocity_boundary(E)
   struct All_variables *E;
 {
   int i,k,lev;
   int nox,noz,noy,nodel;
   float fxx10,fxx20,fyy1,fyy2,fxx0,fxx,fyy;
   float vxx1,vxx2,vxx,vvo,vvc;
   float fslope,vslope;
   static float fxx1,fxx2;

   FILE *fp;
   char output_file[255];
   nox=E->lmesh.nox;
   noz=E->lmesh.noz;
   noy=E->lmesh.noy;
   lev=E->mesh.levmax;
   
   fxx10=1.0;
   fyy1=0.76;
   fxx20=1.0;   /* (fxx1,fyy1), (fxx2,fyy2) the initial coordinates of the trench position */
   fyy2=0.81;

   vxx1=-2.*2.018e0;
   
   vvo=6.*2.018e0;   
   vvc=-2.*2.018e0;     /* vvo--oceanic plate velocity; vvc--continental plate velocity      */

   if(E->advection.timesteps>1)  {
       fxx1=fxx1+E->advection.timestep*vxx1;
       fxx2=fxx2+E->advection.timestep*vxx1;
      }
    
   else  {
       fxx1=fxx10;
       fxx2=fxx20;
      }

   fprintf(stderr,"%f %f\n",fxx1,fxx2);

   if (E->parallel.me_locl[3] == E->parallel.nproczl-1 ) {
   for(k=1;k<=noy;k++)
       for(i=1;i<=nox;i++)   {
          nodel = (k-1)*nox*noz + (i-1)*noz+noz;
          fyy=E->SX[lev][1][1][nodel];
          if (fyy < fyy1 || fyy >fyy2 )   {
             E->sphere.cap[1].VB[1][nodel]=0.0;
             E->sphere.cap[1].VB[2][nodel]=-vvc;
             E->sphere.cap[1].VB[3][nodel]=0.0;
            }    /* the region outside of the domain bounded by the trench length  */
          else if (fyy>=fyy1 && fyy <=fyy2)  {
              if (E->SX[lev][1][2][nodel]>=0.00 && E->SX[lev][1][2][nodel]<= fxx1) {
                 E->sphere.cap[1].VB[1][nodel]=0.0;
                 E->sphere.cap[1].VB[2][nodel]=vvo;
                 E->sphere.cap[1].VB[3][nodel]=0.0;
                 }
               else if ( E->SX[lev][1][2][nodel]>fxx1 && E->SX[lev][1][2][nodel]<fxx2) {
                 E->sphere.cap[1].VB[1][nodel]=0.0;
                 E->sphere.cap[1].VB[2][nodel]=vxx1;
                 E->sphere.cap[1].VB[3][nodel]=0.0;
                 }
               else if ( E->SX[lev][1][2][nodel]>=fxx2) {
                 E->sphere.cap[1].VB[1][nodel]=0.0;
                 E->sphere.cap[1].VB[2][nodel]=vvc;
                 E->sphere.cap[1].VB[3][nodel]=0.0;
                 }
            }   /* end of else if (fyy>=fyy1 && fyy <=fyy2)  */

        }  /* end if for(i=1;i<nox;i++)  */
    }    /* end of E->parallel.me_locl[3]   */

 return;
}
