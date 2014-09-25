/* 

set of routines to deal with anisotropic viscosity

orthotropic viscosity following Muehlhaus, Moresi, Hobbs and Dufour
(PAGEOPH, 159, 2311, 2002)

tranverse isotropy following Han and Wahr (PEPI, 102, 33, 1997)


this set of subroutines for use with both CitcomCU and CitcomS

EDIT THE FILES in hc, not the citcom subdirectories

   
*/

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC

#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"

#include "material_properties.h"
#ifdef USE_GGRD
#include "ggrd_handling.h"
#endif


#include "anisotropic_viscosity.h"
void calc_cbase_at_tp(float , float , float *);
void calc_cbase_at_tp_d(double , double , double *);
#define CITCOM_DELTA(i,j) ((i==j)?(1.0):(0.0))

void myerror_s(char *,struct All_variables *); /* for compatibility with CitcomS */

/* 

output: D[6][6]

input: n[3] director
vis2: anisotropy factor
avmode: anisotropy mode
convert_to_spherical: 1/0 depending on geometry
theta, phi : coordinates for conversion
       
       
*/
void get_constitutive(double D[6][6],double theta, double phi, 
		      int convert_to_spherical,
		      float nx,float ny,float nz,
		      float vis2,
		      int avmode,
		      struct All_variables *E)
{
  double n[3];
  if(E->viscosity.allow_anisotropic_viscosity){
    if((E->monitor.solution_cycles == 0)&&
       (E->viscosity.anivisc_start_from_iso)&&(E->monitor.visc_iter_count == 0)){
      /* 
	 first iteration of "stress-dependent" loop for first timestep 
      */
      get_constitutive_isotropic(D);
    }else{      
      /* 
	 allow for a possibly anisotropic viscosity 
      */
      n[0] =  nx;n[1] =  ny;n[2] =  nz; /* Cartesian directors */
      switch(avmode){
      case  CITCOM_ANIVISC_ORTHO_MODE:
	/* 
	   orthotropic 
	*/
	get_constitutive_orthotropic_viscosity(D,vis2,n,convert_to_spherical,theta,phi); 
	break;
      case CITCOM_ANIVISC_TI_MODE:
	/* 
	   transversely isotropic 
	*/
	get_constitutive_ti_viscosity(D,vis2,0.,n,convert_to_spherical,theta,phi); 
	break;
      default:
	fprintf(stderr,"avmode %i\n",avmode);
	myerror_s("get_constitutive: error, avmode undefined",E);
	break;
      }
      //if(vis2 != 0) print_6x6_mat(stderr,D); 
    }
  }else{
    get_constitutive_isotropic(D);
  }
}


/* 

transversely isotropic viscosity following Han and Wahr (see their eq. 5)


\nu_1 = strong (isotropic) viscosity, scales e_31, e_11, e_22
\nu_2 = weak anisotropy, scales for e_31, e_32
\eta_1 = normal viscosity, (\eta_1+2\nu_1) scales e_11, e_22
\eta_2 = normal viscosity, (\eta_2+2\nu_2) = 2\eta_1 + 2\nu_1, scales e_33

we use (for consistency with anisotropic viscosity)

\delta = 1-\nu_2/\nu_1

and 

\Gamma, such that \eta_1 = \Gamma \nu_1

\nu_1 is the reference, isotropic viscosity, set to unity here, 

for \nu_1 = 1 -->

\nu_2 = 1 - \delta ; \eta_1 = \Gamma ; \eta_2 = 2\eta_1 + 2\nu_1 - 2\nu_2 = 2\Gamma -2 \delta;  isotropy \Delta = 0, \Gamma = 0

n[3] is the cartesian direction into which the weak shear points
(ie. routine will rotate the 3 axis into the n direction) and will 
normalize n, if not already normalized


*/
void get_constitutive_ti_viscosity(double D[6][6], double delta_vis, double gamma_vis,
				   double n[3], int convert_to_spherical,
				   double theta, double phi) 
{
  double nlen,delta_vis2;
  int ani;
  /* isotropic part, in units of iso_visc */
  get_constitutive_isotropic(D);
  ani = FALSE;
  if((fabs(delta_vis) > 3e-15) || (fabs(gamma_vis) > 3e-15)){
    ani = TRUE;
    /* get Cartesian anisotropy matrix by adding anisotropic
       components */
    D[0][0] += gamma_vis;
    D[1][0] = D[0][1] = gamma_vis;
    D[1][1] = D[0][0];
    D[2][2] += 2.*gamma_vis;
    D[4][4] -= delta_vis;
    D[5][5] = D[4][4];
    /* 
       the rotation routine takes care of normalization and will normalize n
    */
    //print_6x6_mat(stderr,D);
    rotate_ti6x6_to_director(D,n); /* rotate such that the generic z
				      preferred axis is aligned with
				      the director */
    //print_6x6_mat(stderr,D);fprintf(stderr,"\n");
  }
  if(ani && convert_to_spherical){
    conv_cart6x6_to_spherical(D,theta,phi,D); /* rotate, can use in place */
  }
}

								     /* 


								     compute a cartesian orthotropic anisotropic viscosity matrix (and
								     rotate it into CitcomS spherical, if requested)

								     viscosity is characterized by a eta_S (weak) viscosity in a shear
								     plane, to which the director is normal


								     output: D[0,...,5][0,...,5] constitutive matrix

								     input: delta_vis difference in viscosity from isotropic viscosity (set to unity here)
   
								     n[0,..,2]: director orientation, specify in cartesian


								     where delta_vis = (1 - eta_S/eta)


								     */
void get_constitutive_orthotropic_viscosity(double D[6][6], double delta_vis,
					    double n[3], int convert_to_spherical,
					    double theta, double phi) 
{
  double nlen,delta_vis2;
  double delta[3][3][3][3];
  int ani;
  ani=FALSE;
  /* start with isotropic */
  get_constitutive_isotropic(D);
  /* get Cartesian anisotropy matrix */
  if(fabs(delta_vis) > 3e-15){
    ani = TRUE;
    get_orth_delta(delta,n);	/* 
				   get anisotropy tensor, \Delta of
				   Muehlhaus et al. (2002)

				   this routine will normalize n, just in case

				*/
    delta_vis2 = 2.0*delta_vis;
    /* s_xx = s_tt */
    D[0][0] -= delta_vis2 * delta[0][0][0][0]; /* * e_xx */
    D[0][1] -= delta_vis2 * delta[0][0][1][1];
    D[0][2] -= delta_vis2 * delta[0][0][2][2];
    D[0][3] -= delta_vis  * (delta[0][0][0][1]+delta[0][0][1][0]);
    D[0][4] -= delta_vis  * (delta[0][0][0][2]+delta[0][0][2][0]);
    D[0][5] -= delta_vis  * (delta[0][0][1][2]+delta[0][0][2][1]);

    D[1][0] -= delta_vis2 * delta[1][1][0][0]; /* s_yy = s_pp */
    D[1][1] -= delta_vis2 * delta[1][1][1][1];
    D[1][2] -= delta_vis2 * delta[1][1][2][2];
    D[1][3] -= delta_vis  * (delta[1][1][0][1]+delta[1][1][1][0]);
    D[1][4] -= delta_vis  * (delta[1][1][0][2]+delta[1][1][2][0]);
    D[1][5] -= delta_vis  * (delta[1][1][1][2]+delta[1][1][2][1]);

    D[2][0] -= delta_vis2 * delta[2][2][0][0]; /* s_zz = s_rr */
    D[2][1] -= delta_vis2 * delta[2][2][1][1];
    D[2][2] -= delta_vis2 * delta[2][2][2][2];
    D[2][3] -= delta_vis  * (delta[2][2][0][1]+delta[2][2][1][0]);
    D[2][4] -= delta_vis  * (delta[2][2][0][2]+delta[2][2][2][0]);
    D[2][5] -= delta_vis  * (delta[2][2][1][2]+delta[2][2][2][1]);

    D[3][0] -= delta_vis2 * delta[0][1][0][0]; /* s_xy = s_tp */
    D[3][1] -= delta_vis2 * delta[0][1][1][1];
    D[3][2] -= delta_vis2 * delta[0][1][2][2];
    D[3][3] -= delta_vis  * (delta[0][1][0][1]+delta[0][1][1][0]);
    D[3][4] -= delta_vis  * (delta[0][1][0][2]+delta[0][1][2][0]);
    D[3][5] -= delta_vis  * (delta[0][1][1][2]+delta[0][1][2][1]);

    D[4][0] -= delta_vis2 * delta[0][2][0][0]; /* s_xz = s_tr */
    D[4][1] -= delta_vis2 * delta[0][2][1][1];
    D[4][2] -= delta_vis2 * delta[0][2][2][2];
    D[4][3] -= delta_vis  * (delta[0][2][0][1]+delta[0][2][1][0]);
    D[4][4] -= delta_vis  * (delta[0][2][0][2]+delta[0][2][2][0]);
    D[4][5] -= delta_vis  * (delta[0][2][1][2]+delta[0][2][2][1]);

    D[5][0] -= delta_vis2 * delta[1][2][0][0]; /* s_yz = s_pr */
    D[5][1] -= delta_vis2 * delta[1][2][1][1];
    D[5][2] -= delta_vis2 * delta[1][2][2][2];
    D[5][3] -= delta_vis  * (delta[1][2][0][1]+delta[1][2][1][0]);
    D[5][4] -= delta_vis  * (delta[1][2][0][2]+delta[1][2][2][0]);
    D[5][5] -= delta_vis  * (delta[1][2][1][2]+delta[1][2][2][1]);
  }
  if(ani && convert_to_spherical){
    //print_6x6_mat(stderr,D);
    conv_cart6x6_to_spherical(D,theta,phi,D); /* rotate, can use same mat for 6x6 */
    //print_6x6_mat(stderr,D);fprintf(stderr,"\n");
  }
}
void get_constitutive_isotropic(double D[6][6])
{
  /* isotropic part, in units of iso_visc */
  zero_6x6(D);
  D[0][0] = 2.0;		/* xx = tt*/
  D[1][1] = 2.0;		/* yy = pp */
  D[2][2] = 2.0;		/* zz = rr */
  D[3][3] = 1.0;		/* xy = tp */
  D[4][4] = 1.0;		/* xz = rt */
  D[5][5] = 1.0;		/* yz = rp */
}


void set_anisotropic_viscosity_at_element_level(struct All_variables *E, 
						int init_stage)
{
  int i,j,k,l,m,off,p,nel,elx,ely,elz,inode,elxlz,el,ani_layer;
  double vis2,n[3],u,v,s,r,xloc[3],z_top,z_bottom,log_vis;
  float base[9],rout[3];
  char tfilename[1000],gmt_string[50];
  int read_grd;
#ifdef USE_GGRD
  struct ggrd_gt *vis2_grd;
#endif
  const int vpts = vpoints[E->mesh.nsd];
  const int ends = enodes[E->mesh.nsd];
  int mgmin,mgmax;
  mgmin = E->mesh.gridmin;
  mgmax = E->mesh.gridmax;
  if(init_stage){
    if(E->parallel.me == 0)
      fprintf(stderr,"set_anisotropic_viscosity: allowing for %s viscosity\n",
	      (E->viscosity.allow_anisotropic_viscosity == 1)?("orthotropic"):("transversely isotropic"));
    if(E->viscosity.anisotropic_viscosity_init)
      myerror_s("anisotropic viscosity should not be initialized twice?!",E);
    /* first call */
    /* initialize anisotropic viscosity at element level, nodes will
       get assigned later */
    switch(E->viscosity.anisotropic_init){
    case 0:			/* isotropic */
    case 3:			/* first init for vel */
    case 4:			/* ISA */
    case 5:			/* same for mixed alignment */
      if(E->parallel.me == 0)fprintf(stderr,"set_anisotropic_viscosity_at_element_level: initializing isotropic viscosity\n");
      for(i=mgmin;i <= mgmax;i++){
	nel  = E->lmesh.NEL[i];
	  for(k=1;k <= nel;k++){
	    for(l=1;l <= vpts;l++){ /* assign to all integration points */
	      off = (k-1)*vpts + l;
	      E->EVI2[i][off] = 0.0;
	      E->EVIn1[i][off] = 1.0; E->EVIn2[i][off] = E->EVIn3[i][off] = 0.0;
	      E->avmode[i][off] = (unsigned char)
		E->viscosity.allow_anisotropic_viscosity;
	    }
	  }
      }
      /* isotropic init end */
      break;
    case 1:
      /* 
	 random fluctuations, for testing a
	 worst case scenario
	 
      */
      if(E->parallel.me == 0)fprintf(stderr,"set_anisotropic_viscosity_at_element_level: initializing random viscosity\n");
      for(i=mgmin;i <= mgmax;i++){
	nel  = E->lmesh.NEL[i];
	  for(k=1;k <= nel;k++){
	    vis2 = 0.01+drand48()*0.99; /* random fluctuation */
	    /* get random vector */
	    do{
	      u = -1 + drand48()*2;v = -1 + drand48()*2;
	      s = u*u + v*v;		
	    }while(s > 1);
	    r = 2.0 * sqrt(1.0-s );
	    n[0] = u * r;		/* x */
	    n[1] = v * r;		/* y */
	    n[2] = 2.0*s -1 ;		/* z */
	    for(l=1;l <= vpts;l++){ /* assign to all integration points */
	      off = (k-1)*vpts + l;
	      E->EVI2[i][off] = vis2;E->EVIn1[i][off] = n[0]; E->EVIn2[i][off] = n[1];E->EVIn3[i][off] = n[2];
	      E->avmode[i][off] = (unsigned char)E->viscosity.allow_anisotropic_viscosity;
	    }
	  }
      }	/* mg loop */
      /* end random init */
      break;
    case 2:			/* from file */
#ifndef USE_GGRD	
      fprintf(stderr,"set_anisotropic_viscosity_at_element_level: anisotropic_init mode 2 requires USE_GGRD compilation\n");
      parallel_process_termination();
#endif
      if(E->sphere.caps == 12)	/* global */
	ggrd_read_anivisc_from_file(E,1);
      else			/* regional */
	ggrd_read_anivisc_from_file(E,0);
      break;
    case 6:		
      /* 
	 
	 tapered within layer, if ani_vis2_factor < 0, will read from file
	 for vis2 base. else, will use constant factor everywhere 
      */
      
      if(E->viscosity.ani_vis2_factor >= 0){
	if(E->parallel.me == 0)
	  fprintf(stderr,"set_anisotropic_viscosity_at_element_level: setting orthotropic tapered, vis2 min %g, global\n",
		  E->viscosity.ani_vis2_factor);
	read_grd = 0;
      }else{
	/* 
	   init the base factor from a grid
	*/
	read_grd = 1;
	sprintf(tfilename,"%s/vis2.grd",E->viscosity.anisotropic_init_dir);
	if(E->parallel.me == 0)
	  fprintf(stderr,"set_anisotropic_viscosity_at_element_level: setting orthotropic tapered, reading base vis2 from %s\n",
		  tfilename);
	vis2_grd = (struct  ggrd_gt *)calloc(1,sizeof(struct ggrd_gt));
	if(E->sphere.caps == 12)	
	  sprintf(gmt_string,GGRD_GMT_GLOBAL_STRING); 
	else
	  sprintf(gmt_string,"");
	if(ggrd_grdtrack_init_general(FALSE,tfilename,"",gmt_string,
				      vis2_grd,(E->parallel.me == 0),FALSE,
				      FALSE))
	  myerror_s("ggrd init error for vis2 at layered anisotropic init",E);
      }
      /* 
	 select layer where the scaling applies
      */
      if(E->viscosity.anivisc_layer >= 0)
	myerror_s("set_anisotropic_viscosity_at_element_level: need to select layer",E);
      ani_layer = -E->viscosity.anivisc_layer;
      /* CitcomS, the zbase_layers are counted 0...1 top down, need to
	 convert to radius */
      z_bottom = E->sphere.ro-E->viscosity.zbase_layer[ani_layer-1];
      if(ani_layer == 1)
	z_top = E->sphere.ro;
      else
	z_top = E->sphere.ro - E->viscosity.zbase_layer[ani_layer-2];
      for(i=mgmin;i <= mgmax;i++){
	  elx = E->lmesh.ELX[i];elz = E->lmesh.ELZ[i];ely = E->lmesh.ELY[i];
	  elxlz = elx * elz;
	  for (j=1;j <= elz;j++){
	    if(E->mat[j] ==  ani_layer){
	      for(u=0.,inode=1;inode <= ends;inode++){ /* mean vertical coordinate */
		off = E->ien[j].node[inode];
		u += E->sx[3][off];
	      }
	      u /= ends;
	      if(!read_grd){
		/* 
		   do a log scale decrease of vis2 to ani_vis2_factor from bottom to top of layer 
		*/
		vis2 = exp(log(E->viscosity.ani_vis2_factor) * (u-z_bottom)/(z_top-z_bottom));
		//fprintf(stderr,"z %g (%g/%g) vis2 %g vis2_o %g frac %g\n",u,z_top,z_bottom,vis2, E->viscosity.ani_vis2_factor,(u-z_bottom)/(z_top-z_bottom));
		/* 
		   1-eta_s/eta 
		*/
		vis2 = 1 - vis2;
	      }
	      for (k=1;k <= ely;k++){
		for (l=1;l <= elx;l++)   {
		  /* eq.(1) */
		  el = j + (l-1) * elz + (k-1)*elxlz;
		  /* CitcomS */
		  xloc[0] = xloc[1] = xloc[2] = 0.0;
		  for(inode=1;inode <= ends;inode++){
		    off = E->ien[el].node[inode];
		    rtp2xyz((float)E->sx[3][off],(float)E->sx[1][off],(float)E->sx[2][off],rout);
		    xloc[0] += rout[0];xloc[1] += rout[1];xloc[2] += rout[2];
		  }
		  xloc[0]/=ends;xloc[1]/=ends;xloc[2]/=ends;
		  xyz2rtp(xloc[0],xloc[1],xloc[2],rout); 
		  /* got mean location in spherical */
		  if(read_grd){
		    /* read in a local vis2 favtors */
		    if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],
						     vis2_grd,&log_vis,
						     FALSE,FALSE)){
		      fprintf(stderr,"ggrd_read_anivisc_from_file: interpolation error at lon: %g lat: %g\n",
			      rout[2]*180/M_PI,90-rout[1]*180/M_PI);
		      parallel_process_termination();
		    }
		    vis2 = 1 - exp(log(pow(10.0,log_vis)) * (u-z_bottom)/(z_top-z_bottom));
		  }
		  /* 
		     r,t,p=(1,0,0) convert to Caretsian, reuse rout
		  */
		  calc_cbase_at_tp(rout[1],rout[2],base);
		  convert_pvec_to_cvec(1.,0.,0.,base,rout);
		  n[0]=rout[0];n[1]=rout[1];n[2]=rout[2];
		  for(p=1;p <= vpts;p++){ /* assign to all integration points */
		    off = (el-1)*vpts + p;
		    E->EVI2[i][off] = vis2;
		    E->EVIn1[i][off] = n[0]; E->EVIn2[i][off] = n[1];E->EVIn3[i][off] = n[2];
		    E->avmode[i][off] = CITCOM_ANIVISC_ORTHO_MODE;
		  }
		}
	      }
	    }else{		/* outside layer = isotropic */
	      for (k=1;k <= ely;k++){
		for (l=1;l <= elx;l++){
		  el = j + (l-1) * elz + (k-1)*elxlz;
		  for(p=1;p <= vpts;p++){ /* assign to all integration points */
		    off = (el-1)*vpts + p;
		    E->EVI2[i][off] = 0;E->EVIn1[i][off] = 1; E->EVIn2[i][off] = 0;E->EVIn3[i][off] = 0;E->avmode[i][off] = CITCOM_ANIVISC_ORTHO_MODE;
		  }
		}
	      }
	    }
	  }
      }	/* end multigrid */
	if(read_grd)
	ggrd_grdtrack_free_gstruc(vis2_grd);
	
      break;			/* end of 6 branch */
    default:
      fprintf(stderr,"set_anisotropic_viscosity_at_element_level: anisotropic_init %i undefined\n",
	      E->viscosity.anisotropic_init);
      parallel_process_termination();
      break;
    }
    E->viscosity.anisotropic_viscosity_init = TRUE;
    /* end initialization stage */
  }else{
    //if(E->parallel.me == 0)fprintf(stderr,"reassigning anisotropic viscosity, mode %i\n",E->viscosity.anisotropic_init);
    if((E->monitor.solution_cycles > 0) || (E->monitor.visc_iter_count > 0)){
      /* standard operation every time step */
      switch(E->viscosity.anisotropic_init){
	/* if flow has been computed, use director aligned with ISA */
      case 3:		     /* vel lignment */
	/* we have a velocity solution already */
	
	align_director_with_ISA_for_element(E,CITCOM_ANIVISC_ALIGN_WITH_VEL);
	break;
      case 4:		     /* vel alignment */
	if((E->monitor.solution_cycles > 0) || (E->monitor.visc_iter_count > 0))
	  align_director_with_ISA_for_element(E,CITCOM_ANIVISC_ALIGN_WITH_ISA);
	break;
      case 5:		     /* mixed alignment */
	if((E->monitor.solution_cycles > 0) || (E->monitor.visc_iter_count > 0))
	  align_director_with_ISA_for_element(E,CITCOM_ANIVISC_MIXED_ALIGN);
	break;
      default:			/* default, no further modification of
				   anisotropy */
	break;
      }
    }
  }
}

void normalize_director_at_nodes(struct All_variables *E,float **n1,float **n2, float **n3, int lev)
{
  int n,m;
    for(n=1;n<=E->lmesh.NNO[lev];n++){
      normalize_vec3(&(n1[CPPR][n]),&(n2[CPPR][n]),&(n3[CPPR][n]));
    }
}
void normalize_director_at_gint(struct All_variables *E,float **n1,float **n2, float **n3, int lev)
{
  int m,e,i,enode;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
    for(e=1;e<=E->lmesh.NEL[lev];e++)
      for(i=1;i<=vpts;i++)      {
	enode = (e-1)*vpts+i;
	normalize_vec3(&(n1[CPPR][enode]),&(n2[CPPR][enode]),&( n3[CPPR][enode]));
      }
}
/* 
   
convert cartesian fourth order tensor (input c) to spherical, CitcomS
format (output p)

c and p cannot be the same matrix

1: t 2: p 3: r

(E only passed for debugging)

*/

  void conv_cart4x4_to_spherical(double c[3][3][3][3], double theta, double phi, double p[3][3][3][3])
{
  double rot[3][3];
  get_citcom_spherical_rot(theta,phi,rot);
  rot_4x4(c,rot,p);
}

/* convert [6][6] (input c) in cartesian to citcom spherical (output
   p)

   c and p can be the same amtrix

*/
void conv_cart6x6_to_spherical(double c[6][6], 
			       double theta, double phi, double p[6][6])
{
  double c4[3][3][3][3],p4[3][3][3][3],rot[3][3];
  get_citcom_spherical_rot(theta,phi,rot);
  c4fromc6(c4,c);		
  rot_4x4(c4,rot,p4);
  c6fromc4(p,p4);
}
/* 

rotate 6x6 D matrix with preferred axis aligned with z to the
Cartesian director orientation, in place

n will be normalized, just in case

*/
void rotate_ti6x6_to_director(double D[6][6],double n[3])
{
  double a[3][3][3][3],b[3][3][3][3],rot[3][3],
    hlen2,x2,y2,xy,zm1;
  /* normalize */
  normalize_vec3d((n+0),(n+1),(n+2));
  /* calc aux variable */
  x2 = n[0]*n[0];y2 = n[1]*n[1];xy = n[0]*n[1];
  hlen2 = x2 + y2;zm1 = n[2]-1;
  if(hlen2 > 3e-15){
    /* rotation matrix to get {0,0,1} to {x,y,z} */
    rot[0][0] = (y2 + x2*n[2])/hlen2;
    rot[0][1] = (xy*zm1)/hlen2;
    rot[0][2] = n[0];
    rot[1][0] = rot[0][1];
    rot[1][1] = (x2 + y2*n[2])/hlen2;
    rot[1][2] = n[1];
    rot[2][0] = -n[0];
    rot[2][1] = -n[1];
    rot[2][2] =  n[2];

    /* rotate the D matrix */
    c4fromc6(a,D);
    rot_4x4(a,rot,b);
    c6fromc4(D,b);
  }			/* already oriented right */
    
}

void get_citcom_spherical_rot(double theta, double phi, double rot[3][3]){
  float base[9];
  calc_cbase_at_tp((float)theta,(float)phi, base); /* compute cartesian basis at
						      theta, phi location */
  rot[0][0] = base[3];rot[0][1] = base[4];rot[0][2] = base[5]; /* theta */
  rot[1][0] = base[6];rot[1][1] = base[7];rot[1][2] = base[8]; /* phi */
  rot[2][0] = base[0];rot[2][1] = base[1];rot[2][2] = base[2]; /* r */
  //fprintf(stderr,"%g %g ; %g %g %g ; %g %g %g ; %g %g %g\n\n",
  //theta,phi,rot[0][0],rot[0][1],rot[0][2],rot[1][0],rot[1][1],rot[1][2],rot[2][0],rot[2][1],rot[2][2]);
}
/* 


get fourth order anisotropy tensor for orthotropic viscosity from
Muehlhaus et al. (2002)

*/
void get_orth_delta(double d[3][3][3][3],double n[3])
{
  int i,j,k,l;
  double tmp;
  normalize_vec3d((n+0),(n+1),(n+2));
  for(i=0;i<3;i++)
    for(j=0;j<3;j++)
      for(k=0;k<3;k++)
	for(l=0;l<3;l++){	/* eq. (4) from Muehlhaus et
				   al. (2002) */
	  tmp  = n[i]*n[k]*CITCOM_DELTA(l,j);
	  tmp += n[j]*n[k]*CITCOM_DELTA(i,l);
	  tmp += n[i]*n[l]*CITCOM_DELTA(k,j);
	  tmp += n[j]*n[l]*CITCOM_DELTA(i,k);
	  tmp /= 2.0;
	  tmp -= 2*n[i]*n[j]*n[k]*n[l];
	  d[i][j][k][l] = tmp;
	}

}

/* 
   mode  = 
   CITCOM_ANIVISC_ALIGN_WITH_VEL: align with velocity
   CITCOM_ANIVISC_ALIGN_WITH_ISA: align with ISA
   CITCOM_ANIVISC_MIXED_ALIGN: mixed alignment
 

*/
void align_director_with_ISA_for_element(struct All_variables *E,
					 int mode)
{
  double rtf[4][9];
  double VV[4][9],lgrad[3][3],isa[3],evel[3];
  int e,i,off,m;
  float base[9],n[3];
  static struct CC Cc;
  static struct CCX Ccx;
  const int dims = E->mesh.nsd;
  const int ppts = ppoints[dims];
  const int vpts = vpoints[dims];
  const int ends = enodes[dims];
  const int lev = E->mesh.levmax;
  const int nel = E->lmesh.nel;
  unsigned char avmode;
  double vis2 ; 
  /* anisotropy maximum strength */
  vis2 = 1. - E->viscosity.ani_vis2_factor; /* 1-eta_w/eta_s */

  if(E->parallel.me == 0){
    switch(mode){
    case CITCOM_ANIVISC_ALIGN_WITH_VEL:
      fprintf(stderr,"align_director_with_ISA_for_element: aligning, max ani %g, align with vel\n",
	      vis2);
      break;
    case CITCOM_ANIVISC_ALIGN_WITH_ISA:
      fprintf(stderr,"align_director_with_ISA_for_element: aligning, max ani %g, align with ISA\n",
	      vis2);
      break;
    case CITCOM_ANIVISC_MIXED_ALIGN:
      fprintf(stderr,"align_director_with_ISA_for_element: aligning, max ani %g, align with uniaxial/vel\n",
	      vis2);
      break;
    default:
      fprintf(stderr,"align_director_with_ISA_for_element: mode %i undefined\n",mode);
      myerror_s("",E);
    }
  }
  for(e=1; e <= nel; e++) {
      if(((E->viscosity.anivisc_layer > 0)&&
	  (E->mat[e] <=   E->viscosity.anivisc_layer))||
	 ((E->viscosity.anivisc_layer < 0)&&
	  (E->mat[e] ==  -E->viscosity.anivisc_layer))){
	get_rtf_at_ppts(E, lev, e, rtf); /* pressure points */
	//if((e-1)%E->lmesh.elz==0)
	construct_c3x3matrix_el(E,e,&E->element_Cc,&E->element_Ccx,lev,1);
	for(i = 1; i <= ends; i++){	/* velocity at element nodes */
	  off = E->ien[e].node[i];
	  VV[1][i] = E->sphere.cap[1].V[1][off];
	  VV[2][i] = E->sphere.cap[1].V[2][off];
	  VV[3][i] = E->sphere.cap[1].V[3][off];
	}
	/* calculate velocity gradient matrix */
	get_vgm_p(VV,&(E->N),&(E->GNX[lev][e]),&E->element_Cc, 
		  &E->element_Ccx,rtf,E->mesh.nsd,ppts,ends,TRUE,lgrad,
		  evel);
	/* calculate the ISA axis and determine the type of
	   anisotropy */
	avmode = calc_isa_from_vgm(lgrad,evel,e,isa,E,mode);
	/* 
	   convert for spherical (Citcom system) to Cartesian 
	*/
	calc_cbase_at_tp(rtf[1][1],rtf[2][1],base);
	convert_pvec_to_cvec(isa[2],isa[0],isa[1],base,n);
	/* assign to director for all vpoints */
	for(i=1;i <= vpts;i++){
	  off = (e-1)*vpts + i;
	  E->avmode[lev][off] = avmode;
	  E->EVI2[lev][off] = vis2;
	  E->EVIn1[lev][off] = n[0]; 
	  E->EVIn2[lev][off] = n[1];
	  E->EVIn3[lev][off] = n[2];
	}
      }	/* in layer */
  }

}


/* 
   compute the ISA axis from velocity gradient tensor l, element
   velocity evel, and element number e
   
   mode input: 
   CITCOM_ANIVISC_ALIGN_WITH_VEL: align with velocity
   CITCOM_ANIVISC_ALIGN_WITH_ISA: align with ISA
   CITCOM_ANIVISC_MIXED_ALIGN: mixed alignment

   returns type of anisotropy  CITCOM_ANIVISC_ORTHO_MODE : orthotropic (shear/normal) 
   CITCOM_ANIVISC_TI_MODE : TI 
			      
*/
unsigned char calc_isa_from_vgm(double l[3][3], double ev[3], 
				int e,double isa[3], 
				struct All_variables *E,
				int mode)
{
  double d[3][3],r[3][3],ltrace,eval[3],lc[3][3],
    evec[3][3],strain_scale,gol,t1,t2;
  int i,j,isa_mode;
  /* copy */
  for(i=0;i<3;i++)
    for(j=0;j<3;j++)
      lc[i][j] = l[i][j];
  remove_trace_3x3(lc);
  calc_strain_from_vgm(lc,d);	/* strain-rate */
#ifndef USE_GGRD
  myerror_s("need USE_GGRD compile for ISA axes",E);
#else
  ggrd_solve_eigen3x3(d,eval,evec,E); /* compute the eigensystem */
#endif
  /* normalize by largest abs(eigenvalue) */
  strain_scale = ((t1=fabs(eval[2])) > (t2=fabs(eval[0])))?(t1):(t2);
  for(i=0;i<3;i++){
    eval[i] /= strain_scale;	/* normalized eigenvalues */
    for(j=0;j<3;j++)
      lc[i][j] /= strain_scale;
  }
  /* recompute normalized strain rate and rotation */
  calc_strain_from_vgm(lc,d);
  calc_rot_from_vgm(lc,r);	/* rotation */
  switch(mode){
  case CITCOM_ANIVISC_ALIGN_WITH_VEL:			/* use velocity */
    isa[0] = ev[0];isa[1] = ev[1];isa[2]=ev[2];
    normalize_vec3d(isa,(isa+1),(isa+2));
    return CITCOM_ANIVISC_TI_MODE;
    break;
  case CITCOM_ANIVISC_ALIGN_WITH_ISA:
    isacalc(lc,&gol,isa,E,&isa_mode);
    if((isa_mode == -1)||(isa_mode == 0)){
      /* ISA cannot be found = align with flow */
      isa[0] = ev[0];isa[1] = ev[1];isa[2]=ev[2]; 
      normalize_vec3d(isa,(isa+1),(isa+2));
      return CITCOM_ANIVISC_TI_MODE;
    }else{			/* actual ISA */
      return CITCOM_ANIVISC_ORTHO_MODE;
    }
    break;
  case CITCOM_ANIVISC_MIXED_ALIGN:
    /* mixed */
    if(is_pure_shear(lc,d,r)){
      /* align any pure shear state with the biggest absolute
	 eigenvector */
      /* largest EV (extensional) */
      //isa[0] = evec[0][0];isa[1] = evec[0][1];isa[2] = evec[0][2];
      /* smallest (compressive) EV */
      isa[0] = evec[2][0];isa[1] = evec[2][1];isa[2] = evec[2][2];     
      return CITCOM_ANIVISC_ORTHO_MODE;		
    }else{
      /* simple shear */
      isa[0] = ev[0];isa[1] = ev[1];isa[2]=ev[2]; /* align with vel for now */
      normalize_vec3d(isa,(isa+1),(isa+2));
      return CITCOM_ANIVISC_TI_MODE;			/* TI */
    }
    break;
  default:
    myerror_s("ISA mode undefined",E);
    break;
  }
  return 0;
}
/* 
   determine if deformation state is pure shear from normalized
   velocity gradient, strain, and rotation tensor

*/

int is_pure_shear(double l[3][3],double e[3][3],double r[3][3])
{

  double mrot,tmp,e2;
  /* find max rotation */
  mrot = fabs(r[0][1]);		/* xy */
  if((tmp=fabs(r[0][2]))>mrot)mrot = tmp; /* xz */
  if((tmp=fabs(r[1][2]))>mrot)mrot = tmp; /* yz */
  /* second invariant of strain-rate tensor */
  e2 = second_invariant_from_3x3(e);
  if(mrot/e2 >= 1)
    return 0;			/* simple shear */
  else
    return 1;			/* pure shear */
  
}
/* 
   rotate fourth order tensor 
   c4 and c4c cannot be the same matrix

*/
void rot_4x4(double c4[3][3][3][3], double r[3][3], 
	     double c4c[3][3][3][3])
{

  int i1,i2,i3,i4,j1,j2,j3,j4;

  zero_4x4(c4c);

  for(i1=0;i1<3;i1++)
    for(i2=0;i2<3;i2++)
      for(i3=0;i3<3;i3++)
	for(i4=0;i4<3;i4++)
	  for(j1=0;j1<3;j1++)
	    for(j2=0;j2<3;j2++)
	      for(j3=0;j3<3;j3++)
		for(j4=0;j4<3;j4++)
		  c4c[i1][i2][i3][i4] += 
		    r[i1][j1] * r[i2][j2] * 
		    r[i3][j3] * r[i4][j4] * c4[j1][j2][j3][j4];

}
void zero_6x6(double a[6][6])
{
  int i,j;
  for(i=0;i<6;i++)
    for(j=0;j<6;j++)
      a[i][j] = 0.;
  
}
void zero_4x4(double a[3][3][3][3])
{
  int i1,i2,i3,i4;
  for(i1=0;i1<3;i1++)  
    for(i2=0;i2<3;i2++)  
      for(i3=0;i3<3;i3++) 
	for(i4=0;i4<3;i4++) 
	  a[i1][i2][i3][i4] = 0.0;
  
}
void copy_4x4(double a[3][3][3][3], double b[3][3][3][3])
{

  int i1,i2,i3,i4;
  for(i1=0;i1<3;i1++)  
    for(i2=0;i2<3;i2++)  
      for(i3=0;i3<3;i3++) 
	for(i4=0;i4<3;i4++) 
	  b[i1][i2][i3][i4] = a[i1][i2][i3][i4];
}
void copy_6x6(double a[6][6], double b[6][6])
{

  int i1,i2;
  for(i1=0;i1<6;i1++)  
    for(i2=0;i2<6;i2++)  
      b[i1][i2] = a[i1][i2];
}

void print_6x6_mat(FILE *out, double c[6][6])
{
  int i,j;
  for(i=0;i<6;i++){
    for(j=0;j<6;j++)
      fprintf(out,"%14.5e ",(fabs(c[i][j])<5e-15)?(0):(c[i][j]));
    fprintf(out,"\n");
  }
}
void print_3x3_mat(FILE *out, double c[3][3])
{
  int i,j;
  for(i=0;i<3;i++){
    for(j=0;j<3;j++)
      fprintf(out,"%14.5e ",(fabs(c[i][j])<5e-15)?(0):(c[i][j]));
    fprintf(out,"\n");
  }
}
/* 
   create a fourth order tensor representation from the voigt
   notation, assuming only upper half is filled in

*/
void c4fromc6(double c4[3][3][3][3],double c[6][6])
{
  int i,j;
  
  c4[0][0][0][0] =                  c[0][0];
  c4[0][0][1][1] =                  c[0][1];
  c4[0][0][2][2] =                  c[0][2];
  c4[0][0][0][1] = c4[0][0][1][0] = c[0][3];
  c4[0][0][0][2] = c4[0][0][2][0] = c[0][4];
  c4[0][0][1][2] = c4[0][0][2][1] = c[0][5];

  c4[1][1][0][0] =                  c[0][1];
  c4[1][1][1][1] =                  c[1][1];
  c4[1][1][2][2] =                  c[1][2];
  c4[1][1][0][1] = c4[1][1][1][0] = c[1][3];
  c4[1][1][0][2] = c4[1][1][2][0] = c[1][4];
  c4[1][1][1][2] = c4[1][1][2][1] = c[1][5];
 
  c4[2][2][0][0] =                  c[0][2];
  c4[2][2][1][1] =                  c[1][2];
  c4[2][2][2][2] =                  c[2][2];
  c4[2][2][0][1] = c4[2][2][1][0] = c[2][3];
  c4[2][2][0][2] = c4[2][2][2][0] = c[2][4];
  c4[2][2][1][2] = c4[2][2][2][1] = c[2][5];

  c4[0][1][0][0] =                  c[0][3];
  c4[0][1][1][1] =                  c[1][3];
  c4[0][1][2][2] =                  c[2][3];
  c4[0][1][0][1] = c4[0][1][1][0] = c[3][3];
  c4[0][1][0][2] = c4[0][1][2][0] = c[3][4];
  c4[0][1][1][2] = c4[0][1][2][1] = c[3][5];

  c4[0][2][0][0] =                  c[0][4];
  c4[0][2][1][1] =                  c[1][4];
  c4[0][2][2][2] =                  c[2][4];
  c4[0][2][0][1] = c4[0][2][1][0] = c[3][4];
  c4[0][2][0][2] = c4[0][2][2][0] = c[4][4];
  c4[0][2][1][2] = c4[0][2][2][1] = c[4][5];

  c4[1][2][0][0] =                  c[0][5];
  c4[1][2][1][1] =                  c[1][5];
  c4[1][2][2][2] =                  c[2][5];
  c4[1][2][0][1] = c4[1][2][1][0] = c[3][5];
  c4[1][2][0][2] = c4[1][2][2][0] = c[4][5];
  c4[1][2][1][2] = c4[1][2][2][1] = c[5][5];

  /* assign the symmetric diagonal terms */
  for(i=0;i<3;i++)
    for(j=0;j<3;j++){
      c4[1][0][i][j] = c4[0][1][i][j];
      c4[2][0][i][j] = c4[0][2][i][j];
      c4[2][1][i][j] = c4[1][2][i][j];
    }

}
void c6fromc4(double c[6][6],double c4[3][3][3][3])
{
  int i,j;
  
  c[0][0] = c4[0][0][0][0];
  c[0][1] = c4[0][0][1][1];
  c[0][2] = c4[0][0][2][2];
  c[0][3] = c4[0][0][0][1];
  c[0][4] = c4[0][0][0][2];
  c[0][5] = c4[0][0][1][2];

  c[1][1] = c4[1][1][1][1];
  c[1][2] = c4[1][1][2][2];
  c[1][3] = c4[1][1][0][1];
  c[1][4] = c4[1][1][0][2];
  c[1][5] = c4[1][1][1][2];

  c[2][2] = c4[2][2][2][2];
  c[2][3] = c4[2][2][0][1];
  c[2][4] = c4[2][2][0][2];
  c[2][5] = c4[2][2][1][2];
  
  c[3][3] = c4[0][1][0][1];
  c[3][4] = c4[0][1][0][2];
  c[3][5] = c4[0][1][1][2];
  
  c[4][4] = c4[0][2][0][2];
  c[4][5] = c4[0][2][1][2];
  
  c[5][5] = c4[1][2][1][2];
  /* fill in the lower half */
  for(i=0;i<6;i++)
    for(j=i+1;j<6;j++)
      c[j][i] = c[i][j];
}

void isacalc(double l[3][3], double *gol,double isa[3],
	     struct All_variables *E,int *isa_mode)
{

  //
  // input: l: *normalized* velocity gradient tensor, in FORTRAN sorting
  // output: isa(3): infite strain axis, 
  // gol: grain orientation lag
  double ltrace,isa_diff;
  double f[3][3],eval[3],evec[3][3],u[3][3],le[3][3];

  // make sure l does not have a trace
  remove_trace_3x3(l);
  
#ifdef CITCOM_USE_EXPOKIT
  calc_exp_matrixt(l,75,le,E);	/* following Kaminski & Ribe (G-Cubed, 2001) */
  f_times_ft(le,u);ggrd_solve_eigen3x3(u,eval,evec,E); 
  isa[0] = evec[0][0]; isa[1] = evec[0][1]; isa[2] = evec[0][2];
  calc_exp_matrixt(l,80,le,E);	f_times_ft(le,u);ggrd_solve_eigen3x3(u,eval,evec,E); 
  isa_diff = 1.-fabs(evec[0][0]*isa[0]+evec[0][1]*isa[1]+evec[0][2]*isa[2]);
  if(isa_diff > 1e-4)		/* ISA does not exist */
    *isa_mode = -1;
  else
    *isa_mode = 1;
  //fprintf(stderr,"A: %11g %11g %11g - %11g %i\n",isa[0],isa[1],isa[2],isa_diff,*isa_mode);
  
#else
  // Limit deformation gradient tensor for infinite time
  // calculation of the ISE orientation using Sylvester's formula
  drex_eigen(l,f,isa_mode);
  if(*isa_mode == -1){
    // isa is flow1
    isa[0]=isa[1]=isa[2] = -1.;
  }else if(*isa_mode == 0){
    isa[0] = isa[1] = isa[2] = 0;
    *gol = -1.;
  }else{
    // 2. formation of the left-stretch tensor U = FFt
    f_times_ft(f,u);
    // 3. eigen-values and eigen-vectors of U
    ggrd_solve_eigen3x3(u,eval,evec,E); 
    // largest eigenvector
    isa[0] = evec[0][0];isa[1] = evec[0][1];isa[2] = evec[0][2];
    //fprintf(stderr,"B: %11g %11g %11g\n",evec[0][0],evec[0][1],evec[0][2]);
  }
#endif
}
/* 
   F^2 = F * F^T
*/
void f_times_ft(double f[3][3],double out[3][3])
{
  int i,j,k;
  for(i=0;i<3;i++)
    for(j=0;j<3;j++){
      out[i][j] = 0.0;
      for(k=0;k<3;k++)
	out[i][j] += f[i][k] * f[j][k];
    }
}
/* find eigenvalues of velocity gradient tensor (modified from DREX
   code of Kaminski et al. 2004)

*/
void drex_eigen(double l[3][3],double f[3][3], int *mode)
{
  double a2,a3,q,q3,r2,r,theta,xx,lambda1,lambda2,lambda3,sq2;
  const double four_pi = 4.0*M_PI,
    two_pi = 2.0*M_PI,
    one_third=1./3.;
  /* looking for the eigen-values of L (using tr(l)=0) */
  a2 = l[0][0] * l[1][1] + l[1][1] * l[2][2] + l[2][2]*l[0][0] -
    l[0][1] * l[1][0] - l[1][2]*l[2][1] - l[2][0]*l[0][2];
  a3 = l[0][0]*l[1][2]*l[2][1] + l[0][1]*l[1][0]*l[2][2] + 
    l[0][2]*l[1][1]*l[2][0] - l[0][0]*l[1][1]*l[2][2] - 
    l[0][1]*l[1][2]*l[2][0] - l[0][2]*l[1][0]*l[2][1];
  
  q = -a2/3.;
  r =  a3/2.;
  q3 = q*q*q;
  r2 = r*r;
 
  if(fabs(q) < 1e-9){
    /* simple shear, isa=veloc */
    *mode = -1;
  }else if(q3-r2 >= 0){
    sq2 = 2*sqrt(q);

    theta = acos(pow(r/q,1.5));
    lambda1 = -sq2*cos(theta/3);
    lambda2 = -sq2*cos((theta+two_pi)/3.);
    lambda3 = -sq2*cos((theta+four_pi)/3.);
    
    if (fabs(lambda1-lambda2) < 1e-13) 
      lambda1 = lambda2;
    if (fabs(lambda2-lambda3) < 1e-13) 
      lambda2 = lambda3;
    if (fabs(lambda3-lambda1) < 1e-13) 
      lambda3 = lambda1;
    
    if((lambda1 > lambda2)  && (lambda1 > lambda3)) {
      malmul_scaled_id(f,l,-lambda2,-lambda3);*mode=1;
    }else if((lambda2 > lambda3 ) && (lambda2 > lambda1)){
      malmul_scaled_id(f,l,-lambda3,-lambda1);*mode = 1;
    }else if((lambda3 > lambda1 ) && (lambda3 > lambda2)){
      malmul_scaled_id(f,l,-lambda1,-lambda2);*mode = 1;
    }else if((lambda1 == lambda2 ) && (lambda3 > lambda1)) {
      malmul_scaled_id(f,l,-lambda1,-lambda2);*mode = 1;
    }else if((lambda2 == lambda3 ) && (lambda1 > lambda2)){
      malmul_scaled_id(f,l,-lambda2,-lambda3);*mode = 1;
    }else if((lambda3 == lambda1 ) && (lambda2 > lambda3)){
      malmul_scaled_id(f,l,-lambda3,-lambda1);*mode = 1;
    }else if((lambda1 == lambda2 ) && (lambda3 < lambda1)){
      *mode =0;
    }else if((lambda2 == lambda3 ) && (lambda1 < lambda2)){
      *mode = 0;
    }else if((lambda3 == lambda1 ) && (lambda2 < lambda3)){
      *mode = 0;
    }
  }else{
    xx = pow(sqrt(r2-q3)+fabs(r),one_third);
    if(r < 0)
      lambda1 =  xx+q/xx;
    else
      lambda1 = -xx+q/xx;
    lambda2 = -lambda1/2.;
    lambda3 = -lambda1/2.;
    if (lambda1 > 1e-9) {
      malmul_scaled_id(f,l,-lambda2,-lambda3);
      *mode = 2;
    }else{
      *mode = 0;
    }

  }
}
/* f = matmul(l-lambda2*Id,l-lambda3*Id); */
void malmul_scaled_id(double f[3][3],double l[3][3],
		      double f1,double f2)
{
  double a[3][3],b[3][3];
  int i,j;
  for(i=0;i<3;i++)
    for(j=0;j<3;j++){
      a[i][j] = b[i][j] = l[i][j];
      if(i==j){
	a[i][j] += f1;
	b[i][j] += f2;
      }
    }
  matmul_3x3(a,b,f);
      

}
#ifdef CITCOM_USE_EXPOKIT
/*

calculate exp(A t) using DGPADM from EXPOKIT for a 3x3

(needs LAPACK)

*/
void calc_exp_matrixt(double a[3][3],double t,double ae[3][3],
		      struct All_variables *E)
{
  int ideg=6;// degre of Pade approximation, six should be ok
  int m=3,ldh=3;// a(ldh,m) dimensions
  double *wsp;// work space
  double af[9];
  int ipiv[4],iexph,ns;// workspace, output pointer, nr of squareas
  int iflag,lwsp;// exit code, size of workspace
  int i,j,k;
  /* 
     work space 
  */
  lwsp = 2*(4*m*m+ideg+1);// size of workspace, oversized by factor two
  wsp = (double *)calloc(lwsp,sizeof(double));
  /* assign fortran style */
  for(k=i=0;i<3;i++)
    for(j=0;j<3;j++,k++)
      af[k] = a[i][j];
  //
  // call to expokit routine
  dgpadm_(&ideg,&m,&t,af,&ldh,wsp,&lwsp,ipiv,&iexph,&ns,&iflag);
  if(iflag < 0)
    myerror_s("calc_exp_matrixt: problem in dgpadm",E);
  // assign to output
  for(i=0,k=iexph-1;i<3;i++)
    for(j=0;j<3;j++,k++)
      ae[i][j] = wsp[k];
  
  free(wsp);

}
#endif
void myerror_s(char *a,struct All_variables *E){
  myerror(E,a);
}
#endif
