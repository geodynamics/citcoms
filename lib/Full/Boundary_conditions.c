#include "element_definitions.h"
#include "global_defs.h"
#include <math.h>
 
/* ========================================== */

void velocity_boundary_conditions(E)
     struct All_variables *E;
{
  void velocity_imp_vert_bc();
  void velocity_refl_vert_bc();
  void horizontal_bc();
  void velocity_apply_periodicapply_periodic_bcs();
  void read_velocity_boundary_from_file(); 
  int node,d,j,noz,lv;
 
  for(lv=E->mesh.gridmax;lv>=E->mesh.gridmin;lv--)
    for (j=1;j<=E->sphere.caps_per_proc;j++)     {
      noz = E->mesh.NOZ[lv];
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

        if(E->control.vbcs_file) 
          read_velocity_boundary_from_file(E);   

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
 
/* if(E->control.verbose) */
/*  for (j=1;j<=E->sphere.caps_per_proc;j++) */
/*    for (node=1;node<=E->lmesh.nno;node++) */
/*       fprintf(E->fp_out,"m=%d VB== %d %g %g %g flag %u %u %u\n",j,node,E->sphere.cap[j].VB[1][node],E->sphere.cap[j].VB[2][node],E->sphere.cap[j].VB[3][node],E->node[j][node]&VBX,E->node[j][node]&VBY,E->node[j][node]&VBZ); */

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
  int j,lev,noz;
  
  lev = E->mesh.levmax;
  for (j=1;j<=E->sphere.caps_per_proc;j++)    {
    noz = E->mesh.noz;
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
 
    }     /* end for j */

   return; }

/* ========================================== */

void velocity_refl_vert_bc(E)
     struct All_variables *E;
{
 
  return;
}

void temperature_refl_vert_bc(E)
     struct All_variables *E;
{
  int i,j;
  int node1,node2;
  const int dims=E->mesh.nsd;

 /* Temps and bc-values  at top level only */

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
   
  if ( ROW==1&&E->parallel.me_loc[3]==0 ||
       ROW==E->mesh.NOZ[level]&&E->parallel.me_loc[3]==E->parallel.nprocz-1 ) {

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
 int n1,n2,e1,level;
 int i,j,ii,jj;
 const int dims=E->mesh.nsd;

 fprintf(E->fp,"Periodic temperature boundary conditions\n");
   
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
    

void temperatures_conform_bcs(E)
     struct All_variables *E;
{
    int j,node;
    unsigned int type;

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
	}
	
	/* next node */ 
    }

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
