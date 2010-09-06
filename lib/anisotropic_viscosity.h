#ifndef __CITCOM_READ_ANIVISC_HEADER__

void get_constitutive_orthotropic_viscosity(double [6][6], double,double [3], int, double, double) ;

void set_anisotropic_viscosity_at_element_level(struct All_variables *,int);
void normalize_director_at_nodes(struct All_variables *,float **,float **, float **, int );
void normalize_director_at_gint(struct All_variables *,float **,float **, float **, int );






void conv_cart4x4_to_spherical(double [3][3][3][3],
			       double , double, double [3][3][3][3]);
void get_delta(double [3][3][3][3],double [3]);
void rot_4x4(double [3][3][3][3],double [3][3], double [3][3][3][3]);
void print_6x6_mat(FILE *, double [6][6]);
#define __CITCOM_READ_ANIVISC_HEADER__
#endif
