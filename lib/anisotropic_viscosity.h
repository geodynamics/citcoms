#ifndef __CITCOM_READ_ANIVISC_HEADER__
void get_constitutive_ti_viscosity(double [6][6], double, double ,double [3], int ,
				   double , double) ;

void get_constitutive_orthotropic_viscosity(double [6][6], double,
					    double [3], int,
					    double , double ) ;
void set_anisotropic_viscosity_at_element_level(struct All_variables *, int ) ;

void normalize_director_at_nodes(struct All_variables *, float **, float **, float **, int);
void normalize_director_at_gint(struct All_variables *, float **, float **, float **, int);
void conv_cart4x4_to_spherical(double [3][3][3][3], double, double, double [3][3][3][3]);
void conv_cart6x6_to_spherical(double [6][6], double, double, double [6][6]);
void get_citcom_spherical_rot(double, double, double [3][3]);
void get_orth_delta(double [3][3][3][3], double [3]);
void rot_4x4(double [3][3][3][3], double [3][3], double [3][3][3][3]);
void copy_4x4(double [3][3][3][3], double [3][3][3][3]);
void copy_6x6(double [6][6], double [6][6]);
void c4fromc6(double [3][3][3][3], double [6][6]);
void c6fromc4(double [6][6], double [3][3][3][3]);
void print_6x6_mat(FILE *, double [6][6]);
void zero_6x6(double [6][6]);
void zero_4x4(double [3][3][3][3]);
void rotate_ti6x6_to_director(double [6][6],double [3]);
void normalize_vec3(float *, float *, float *);
void normalize_vec3d(double *, double *, double *);

#define __CITCOM_READ_ANIVISC_HEADER__
#endif
