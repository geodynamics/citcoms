/* Advection_diffusion.c */
void advection_diffusion_parameters(struct All_variables *E);
void advection_diffusion_allocate_memory(struct All_variables *E);
void PG_timestep_init(struct All_variables *E);
void PG_timestep(struct All_variables *E);
void std_timestep(struct All_variables *E);
void PG_timestep_solve(struct All_variables *E);
/* BC_util.c */
void strip_bcs_from_residual(struct All_variables *E, double **Res, int level);
void temperatures_conform_bcs(struct All_variables *E);
void temperatures_conform_bcs2(struct All_variables *E);
void velocities_conform_bcs(struct All_variables *E, double **U);
/* Checkpoints.c */
void output_checkpoint(struct All_variables *E);
void read_checkpoint(struct All_variables *E);
/* Citcom_init.c */
/* Composition_related.c */
void composition_input(struct All_variables *E);
void composition_setup(struct All_variables *E);
void write_composition_instructions(struct All_variables *E);
void fill_composition(struct All_variables *E);
void init_composition(struct All_variables *E);
void map_composition_to_nodes(struct All_variables *E);
void get_bulk_composition(struct All_variables *E);
/* Construct_arrays.c */
void construct_ien(struct All_variables *E);
void construct_surface(struct All_variables *E);
void construct_id(struct All_variables *E);
void get_bcs_id_for_residual(struct All_variables *E, int level, int m);
void construct_lm(struct All_variables *E);
void construct_node_maps(struct All_variables *E);
void construct_node_ks(struct All_variables *E);
void rebuild_BI_on_boundary(struct All_variables *E);
void construct_masks(struct All_variables *E);
void construct_sub_element(struct All_variables *E);
void construct_elt_ks(struct All_variables *E);
void construct_elt_gs(struct All_variables *E);
void construct_elt_cs(struct All_variables *E);
void construct_stiffness_B_matrix(struct All_variables *E);
int layers_r(struct All_variables *E, float r);
int layers(struct All_variables *E, int m, int node);
void construct_mat_group(struct All_variables *E);
/* Convection.c */
void set_convection_defaults(struct All_variables *E);
void read_convection_settings(struct All_variables *E);
void convection_derived_values(struct All_variables *E);
void convection_allocate_memory(struct All_variables *E);
void convection_initial_fields(struct All_variables *E);
void convection_boundary_conditions(struct All_variables *E);
/* Determine_net_rotation.c */
double determine_model_net_rotation(struct All_variables *E, double *omega);
double determine_netr_tp(float r, float theta, float phi, float velt, float velp, int mode, double *c9, double *omega);
void sub_netr(float r, float theta, float phi, float *velt, float *velp, double *omega);
void hc_ludcmp_3x3(double a[3][3], int *indx);
void hc_lubksb_3x3(double a[3][3], int *indx, double *b);
/* Drive_solvers.c */
void general_stokes_solver_setup(struct All_variables *E);
void general_stokes_solver(struct All_variables *E);
int need_visc_update(struct All_variables *E);
void general_stokes_solver_pseudo_surf(struct All_variables *E);
/* Element_calculations.c */
void add_force(struct All_variables *E, int e, double elt_f[24], int m);
void assemble_forces(struct All_variables *E, int penalty);
void assemble_forces_pseudo_surf(struct All_variables *E, int penalty);
void get_ba(struct Shape_function *N, struct Shape_function_dx *GNx, struct CC *cc, struct CCX *ccx, double rtf[4][9], int dims, double ba[9][9][4][7]);
void get_ba_p(struct Shape_function *N, struct Shape_function_dx *GNx, struct CC *cc, struct CCX *ccx, double rtf[4][9], int dims, double ba[9][9][4][7]);
void get_elt_k(struct All_variables *E, int el, double elt_k[24*24], int lev, int m, int iconv);
void assemble_del2_u(struct All_variables *E, double **u, double **Au, int level, int strip_bcs);
void e_assemble_del2_u(struct All_variables *E, double **u, double **Au, int level, int strip_bcs);
void n_assemble_del2_u(struct All_variables *E, double **u, double **Au, int level, int strip_bcs);
void build_diagonal_of_K(struct All_variables *E, int el, double elt_k[24*24], int level, int m);
void build_diagonal_of_Ahat(struct All_variables *E);
void assemble_c_u(struct All_variables *E, double **U, double **result, int level);
void assemble_div_rho_u(struct All_variables *E, double **U, double **result, int level);
void assemble_div_u(struct All_variables *E, double **U, double **divU, int level);
void assemble_grad_p(struct All_variables *E, double **P, double **gradP, int lev);
double assemble_dAhatp_entry(struct All_variables *E, int e, int level, int m);
void get_elt_c(struct All_variables *E, int el, higher_precision elt_c[24][1], int lev, int m);
void get_elt_g(struct All_variables *E, int el, higher_precision elt_del[24][1], int lev, int m);
void get_elt_f(struct All_variables *E, int el, double elt_f[24], int bcs, int m);
void get_elt_tr(struct All_variables *E, int bel, int side, double elt_tr[24], int m);
void get_elt_tr_pseudo_surf(struct All_variables *E, int bel, int side, double elt_tr[24], int m);
void get_aug_k(struct All_variables *E, int el, double elt_k[24*24], int level, int m);
/* Full_boundary_conditions.c */
void full_velocity_boundary_conditions(struct All_variables *E);
void full_temperature_boundary_conditions(struct All_variables *E);
/* Full_geometry_cartesian.c */
void full_set_2dc_defaults(struct All_variables *E);
void full_set_2pt5dc_defaults(struct All_variables *E);
void full_set_3dc_defaults(struct All_variables *E);
void full_set_3dsphere_defaults(struct All_variables *E);
void full_set_3dsphere_defaults2(struct All_variables *E);
/* Full_lith_age_read_files.c */
void full_lith_age_read_files(struct All_variables *E, int output);
/* Full_obsolete.c */
void parallel_process_initilization(struct All_variables *E, int argc, char **argv);
void parallel_domain_decomp2(struct All_variables *E, float *GX[4]);
void scatter_to_nlayer_id(struct All_variables *E, double **AUi, double **AUo, int lev);
void gather_to_1layer_id(struct All_variables *E, double **AUi, double **AUo, int lev);
void gather_to_1layer_node(struct All_variables *E, float **AUi, float **AUo, int lev);
void gather_to_1layer_ele(struct All_variables *E, float **AUi, float **AUo, int lev);
void gather_TG_to_me0(struct All_variables *E, float *TG);
void sum_across_depth_sph(struct All_variables *E, float *sphc, float *sphs, int dest_proc);
void sum_across_surf_sph(struct All_variables *E, float *TG, int loc_proc);
void set_communication_sphereh(struct All_variables *E);
void process_temp_field(struct All_variables *E, int ii);
void output_velo_related(struct All_variables *E, int file_number);
void output_temp(struct All_variables *E, int file_number);
void output_stress(struct All_variables *E, int file_number, float *SXX, float *SYY, float *SZZ, float *SXY, float *SXZ, float *SZY);
void print_field_spectral_regular(struct All_variables *E, float *TG, float *sphc, float *sphs, int proc_loc, char *filen);
void write_radial_horizontal_averages(struct All_variables *E);
int icheck_regular_neighbors(struct All_variables *E, int j, int ntheta, int nphi, double x, double y, double z, double theta, double phi, double rad);
int iquick_element_column_search(struct All_variables *E, int j, int iregel, int ntheta, int nphi, double x, double y, double z, double theta, double phi, double rad, int *imap, int *ich);
/* Full_parallel_related.c */
void full_parallel_processor_setup(struct All_variables *E);
void full_parallel_domain_decomp0(struct All_variables *E);
void full_parallel_domain_boundary_nodes(struct All_variables *E);
void full_parallel_communication_routs_v(struct All_variables *E);
void full_parallel_communication_routs_s(struct All_variables *E);
void full_exchange_id_d(struct All_variables *E, double **U, int lev);
void full_exchange_snode_f(struct All_variables *E, float **U1, float **U2, int lev);
/* Full_read_input_from_files.c */
void full_read_input_files_for_timesteps(struct All_variables *E, int action, int output);
/* Full_solver.c */
void full_solver_init(struct All_variables *E);
/* Full_sphere_related.c */
void spherical_to_uv2(double center[2], int len, double *theta, double *phi, double *u, double *v);
void uv_to_spherical(double center[2], int len, double *u, double *v, double *theta, double *phi);
void full_coord_of_cap(struct All_variables *E, int m, int icap);
/* Full_tracer_advection.c */
void full_tracer_input(struct All_variables *E);
void full_tracer_setup(struct All_variables *E);
void full_lost_souls(struct All_variables *E);
void full_get_shape_functions(struct All_variables *E, double shp[9], int nelem, double theta, double phi, double rad);
double full_interpolate_data(struct All_variables *E, double shp[9], double data[9]);
void full_get_velocity(struct All_variables *E, int j, int nelem, double theta, double phi, double rad, double *velocity_vector);
int full_icheck_cap(struct All_variables *E, int icap, double x, double y, double z, double rad);
int full_iget_element(struct All_variables *E, int j, int iprevious_element, double x, double y, double z, double theta, double phi, double rad);
void full_keep_within_bounds(struct All_variables *E, double *x, double *y, double *z, double *theta, double *phi, double *rad);
void analytical_test(struct All_variables *E);
void analytical_runge_kutte(struct All_variables *E, int nsteps, double dt, double *x0_s, double *x0_c, double *xf_s, double *xf_c, double *vec);
void analytical_test_function(struct All_variables *E, double theta, double phi, double rad, double *vel_s, double *vel_c);
void pdebug(struct All_variables *E, int i);
/* Full_version_dependent.c */
void full_node_locations(struct All_variables *E);
void full_construct_boundary(struct All_variables *E);
/* General_matrix_functions.c */
int solve_del2_u(struct All_variables *E, double **d0, double **F, double acc, int high_lev);
double multi_grid(struct All_variables *E, double **d1, double **F, double acc, int hl);
double conj_grad(struct All_variables *E, double **d0, double **F, double acc, int *cycles, int level);
void element_gauss_seidel(struct All_variables *E, double **d0, double **F, double **Ad, double acc, int *cycles, int level, int guess);
void gauss_seidel(struct All_variables *E, double **d0, double **F, double **Ad, double acc, int *cycles, int level, int guess);
double cofactor(double A[4][4], int i, int j, int n);
double determinant(double A[4][4], int n);
double gen_determinant(double **A, int n);
long double lg_pow(long double a, int n);
/* Ggrd_handling.c */
/* Global_operations.c */
void remove_horiz_ave(struct All_variables *E, double **X, double *H, int store_or_not);
void remove_horiz_ave2(struct All_variables *E, double **X);
void return_horiz_ave(struct All_variables *E, double **X, double *H);
void return_horiz_ave_f(struct All_variables *E, float **X, float *H);
void return_elementwise_horiz_ave(struct All_variables *E, double **X, double *H);
float return_bulk_value(struct All_variables *E, float **Z, int average);
double return_bulk_value_d(struct All_variables *E, double **Z, int average);
float find_max_horizontal(struct All_variables *E, double Tmax);
void sum_across_surface(struct All_variables *E, float *data, int total);
void sum_across_surf_sph1(struct All_variables *E, float *sphc, float *sphs);
float global_fvdot(struct All_variables *E, float **A, float **B, int lev);
double kineticE_radial(struct All_variables *E, double **A, int lev);
double global_vdot(struct All_variables *E, double **A, double **B, int lev);
double global_pdot(struct All_variables *E, double **A, double **B, int lev);
double global_v_norm2(struct All_variables *E, double **V);
double global_p_norm2(struct All_variables *E, double **P);
double global_div_norm2(struct All_variables *E, double **A);
double global_tdot_d(struct All_variables *E, double **A, double **B, int lev);
float global_tdot(struct All_variables *E, float **A, float **B, int lev);
float global_fmin(struct All_variables *E, float a);
double global_dmax(struct All_variables *E, double a);
float global_fmax(struct All_variables *E, double a);
double Tmaxd(struct All_variables *E, double **T);
float Tmax(struct All_variables *E, float **T);
double vnorm_nonnewt(struct All_variables *E, double **dU, double **U, int lev);
void sum_across_depth_sph1(struct All_variables *E, float *sphc, float *sphs);
void broadcast_vertical(struct All_variables *E, float *sphc, float *sphs, int root);
void remove_rigid_rot(struct All_variables *E);
/* Initial_temperature.c */
void tic_input(struct All_variables *E);
void convection_initial_temperature(struct All_variables *E);
/* Instructions.c */
void initial_mesh_solver_setup(struct All_variables *E);
void read_instructions(struct All_variables *E, char *filename);
void initial_setup(struct All_variables *E);
void initialize_material(struct All_variables *E);
void initial_conditions(struct All_variables *E);
void read_initial_settings(struct All_variables *E);
void check_settings_consistency(struct All_variables *E);
void global_derived_values(struct All_variables *E);
void allocate_common_vars(struct All_variables *E);
void allocate_velocity_vars(struct All_variables *E);
void global_default_values(struct All_variables *E);
void check_bc_consistency(struct All_variables *E);
void set_up_nonmg_aliases(struct All_variables *E, int j);
void report(struct All_variables *E, char *string);
void record(struct All_variables *E, char *string);
void common_initial_fields(struct All_variables *E);
void initial_pressure(struct All_variables *E);
void initial_velocity(struct All_variables *E);
void open_qfiles(struct All_variables *E);
void mkdatadir(const char *dir);
void output_init(struct All_variables *E);
void output_finalize(struct All_variables *E);
char *strip(char *input);
/* Interuption.c */
void interuption(int signal_number);
void set_signal(void);
/* Lith_age.c */
void lith_age_input(struct All_variables *E);
void lith_age_init(struct All_variables *E);
void lith_age_construct_tic(struct All_variables *E);
void lith_age_update_tbc(struct All_variables *E);
void lith_age_temperature_bound_adj(struct All_variables *E, int lv);
void lith_age_conform_tbc(struct All_variables *E);
void assimilate_lith_conform_bcs(struct All_variables *E);
/* Material_properties.c */
void mat_prop_allocate(struct All_variables *E);
void reference_state(struct All_variables *E);
/* Nodal_mesh.c */
void v_from_vector(struct All_variables *E);
void v_from_vector_pseudo_surf(struct All_variables *E);
void velo_from_element(struct All_variables *E, float VV[4][9], int m, int el, int sphere_key);
void velo_from_element_d(struct All_variables *E, double VV[4][9], int m, int el, int sphere_key);
void p_to_nodes(struct All_variables *E, double **P, float **PN, int lev);
void visc_from_gint_to_nodes(struct All_variables *E, float **VE, float **VN, int lev);
void visc_from_nodes_to_gint(struct All_variables *E, float **VN, float **VE, int lev);
void visc_from_gint_to_ele(struct All_variables *E, float **VE, float **VN, int lev);
void visc_from_ele_to_gint(struct All_variables *E, float **VN, float **VE, int lev);
/* Obsolete.c */
void get_global_shape_fn(struct All_variables *E, int el, struct Shape_function *GN, struct Shape_function_dx *GNx, struct Shape_function_dA *dOmega, int pressure, int sphere, double rtf[4][9], int lev, int m);
void get_global_1d_shape_fn_1(struct All_variables *E, int el, struct Shape_function *GM, struct Shape_function_dA *dGammax, int nodal, int m);
void get_global_side_1d_shape_fn(struct All_variables *E, int el, struct Shape_function1 *GM, struct Shape_function1_dx *GMx, struct Shape_function_side_dA *dGamma, int NS, int far, int m);
void get_elt_h(struct All_variables *E, int el, double elt_h[1], int m);
void get_ele_visc(struct All_variables *E, float *EV, int m);
void construct_interp_net(struct All_variables *E);
int locate_cap(struct All_variables *E, double x[4]);
int locate_element(struct All_variables *E, int m, double x[4], int ne);
float sphere_interpolate_point(struct All_variables *E, float **T, int m, int el, double x[4], int ne);
void sphere_interpolate(struct All_variables *E, float **T, float *TG);
void phase_change_410(struct All_variables *E, float **B, float **B_b);
void phase_change_670(struct All_variables *E, float **B, float **B_b);
void phase_change_cmb(struct All_variables *E, float **B, float **B_b);
void flogical_mesh_to_real(struct All_variables *E, float *data, int level);
void p_to_centres(struct All_variables *E, float **PN, double **P, int lev);
void v_to_intpts(struct All_variables *E, float **VN, float **VE, int lev);
void visc_to_intpts(struct All_variables *E, float **VN, float **VE, int lev);
double SIN_D(double x);
double COT_D(double x);
void *Malloc1(int bytes, char *file, int line);
float cross2d(double x11, double x12, double x21, double x22, int D);
double **dmatrix(int nrl, int nrh, int ncl, int nch);
float **fmatrix(int nrl, int nrh, int ncl, int nch);
void dfree_matrix(double **m, int nrl, int nrh, int ncl, int nch);
void ffree_matrix(float **m, int nrl, int nrh, int ncl, int nch);
double *dvector(int nl, int nh);
float *fvector(int nl, int nh);
void dfree_vector(double *v, int nl, int nh);
void ffree_vector(float *v, int nl, int nh);
int *sivector(int nl, int nh);
void sifree_vector(int *v, int nl, int nh);
void dvcopy(struct All_variables *E, double **A, double **B, int a, int b);
void vcopy(float *A, float *B, int a, int b);
double sphere_h(int l, int m, double t, double f, int ic);
double plgndr_a(int l, int m, double t);
float area_of_4node(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);
void print_elt_k(struct All_variables *E, double a[24*24]);
double sqrt_multis(int jj, int ii);
double multis(int ii);
int int_multis(int ii);
void jacobi(struct All_variables *E, double **d0, double **F, double **Ad, double acc, int *cycles, int level, int guess);
/* Output.c */
void output_common_input(struct All_variables *E);
void output(struct All_variables *E, int cycles);
FILE *output_open(char *filename, char *mode);
void output_coord(struct All_variables *E);
void output_visc(struct All_variables *E, int cycles);
void output_velo(struct All_variables *E, int cycles);
void output_surf_botm(struct All_variables *E, int cycles);
void output_geoid(struct All_variables *E, int cycles);
void output_stress(struct All_variables *E, int cycles);
void output_horiz_avg(struct All_variables *E, int cycles);
void output_mat(struct All_variables *E);
void output_pressure(struct All_variables *E, int cycles);
void output_tracer(struct All_variables *E, int cycles);
void output_comp_nd(struct All_variables *E, int cycles);
void output_comp_el(struct All_variables *E, int cycles);
void output_heating(struct All_variables *E, int cycles);
void output_time(struct All_variables *E, int cycles);
/* Output_gzdir.c */
/* Output_h5.c */
void h5output_allocate_memory(struct All_variables *E);
void h5output(struct All_variables *E, int cycles);
void h5input_params(struct All_variables *E);
/* Output_vtk.c */
void vtk_output(struct All_variables *E, int cycles);
/* Pan_problem_misc_functions.c */
int get_process_identifier(void);
void unique_copy_file(struct All_variables *E, char *name, char *comment);
void apply_side_sbc(struct All_variables *E);
void get_buoyancy(struct All_variables *E, double **buoy);
int read_double_vector(FILE *in, int num_columns, double *fields);
int read_previous_field(struct All_variables *E, float **field, char *name, char *abbr);
double myatan(double y, double x);
double return1_test(void);
void rtp2xyz(float r, float theta, float phi, float *xout);
void xyz2rtp(float x, float y, float z, float *rout);
void xyz2rtpd(float x, float y, float z, double *rout);
void calc_cbase_at_tp(float theta, float phi, float *base);
void calc_cbase_at_node(int cap, int node, float *base, struct All_variables *E);
void convert_pvec_to_cvec(float vr, float vt, float vp, float *base, float *cvec);
void *safe_malloc(size_t size);
void myerror(struct All_variables *E, char *message);
void get_r_spacing_fine(double *rr, struct All_variables *E);
void get_r_spacing_at_levels(double *rr, struct All_variables *E);
/* Parallel_util.c */
void parallel_process_termination(void);
void parallel_process_sync(struct All_variables *E);
double CPU_time0(void);
/* Parsing.c */
void setup_parser(struct All_variables *E, char *filename);
void shutdown_parser(struct All_variables *E);
void add_to_parameter_list(char *name, char *value);
int compute_parameter_hash_table(char *s);
int input_int(char *name, int *value, char *interpret, int m);
int input_string(char *name, char *value, char *Default, int m);
int input_boolean(char *name, int *value, char *interpret, int m);
int input_float(char *name, float *value, char *interpret, int m);
int input_double(char *name, double *value, char *interpret, int m);
int input_int_vector(char *name, int number, int *value, int m);
int input_char_vector(char *name, int number, char *value, int m);
int input_float_vector(char *name, int number, float *value, int m);
int input_double_vector(char *name, int number, double *value, int m);
int interpret_control_string(char *interpret, int *essential, double *Default, double *minvalue, double *maxvalue);
/* Phase_change.c */
void phase_change_allocate(struct All_variables *E);
void phase_change_input(struct All_variables *E);
void phase_change_apply_410(struct All_variables *E, double **buoy);
void phase_change_apply_670(struct All_variables *E, double **buoy);
void phase_change_apply_cmb(struct All_variables *E, double **buoy);
/* Problem_related.c */
void read_velocity_boundary_from_file(struct All_variables *E);
void read_mat_from_file(struct All_variables *E);
void read_temperature_boundary_from_file(struct All_variables *E);
void get_initial_elapsed_time(struct All_variables *E);
void set_elapsed_time(struct All_variables *E);
void set_starting_age(struct All_variables *E);
float find_age_in_MY(struct All_variables *E);
/* Process_buoyancy.c */
void post_processing(struct All_variables *E);
void heat_flux(struct All_variables *E);
void compute_horiz_avg(struct All_variables *E);
/* Regional_boundary_conditions.c */
void regional_velocity_boundary_conditions(struct All_variables *E);
void regional_temperature_boundary_conditions(struct All_variables *E);
/* Regional_geometry_cartesian.c */
void regional_set_2dc_defaults(struct All_variables *E);
void regional_set_2pt5dc_defaults(struct All_variables *E);
void regional_set_3dc_defaults(struct All_variables *E);
void regional_set_3dsphere_defaults(struct All_variables *E);
void regional_set_3dsphere_defaults2(struct All_variables *E);
/* Regional_lith_age_read_files.c */
void regional_lith_age_read_files(struct All_variables *E, int output);
/* Regional_obsolete.c */
void parallel_process_initilization(struct All_variables *E, int argc, char **argv);
void parallel_domain_decomp2(struct All_variables *E, float *GX[4]);
void scatter_to_nlayer_id(struct All_variables *E, double **AUi, double **AUo, int lev);
void gather_to_1layer_id(struct All_variables *E, double **AUi, double **AUo, int lev);
void gather_to_1layer_node(struct All_variables *E, float **AUi, float **AUo, int lev);
void gather_to_1layer_ele(struct All_variables *E, float **AUi, float **AUo, int lev);
void gather_TG_to_me0(struct All_variables *E, float *TG);
void renew_top_velocity_boundary(struct All_variables *E);
void output_stress(struct All_variables *E, int file_number, float *SXX, float *SYY, float *SZZ, float *SXY, float *SXZ, float *SZY);
void print_field_spectral_regular(struct All_variables *E, float *TG, float *sphc, float *sphs, int proc_loc, char *filen);
void output_velo_related(struct All_variables *E, int file_number);
void output_temp(struct All_variables *E, int file_number);
void output_visc_prepare(struct All_variables *E, float **VE);
void output_visc(struct All_variables *E, int cycles);
void process_temp_field(struct All_variables *E, int ii);
void process_new_velocity(struct All_variables *E, int ii);
void get_surface_velo(struct All_variables *E, float *SV, int m);
/* Regional_parallel_related.c */
void regional_parallel_processor_setup(struct All_variables *E);
void regional_parallel_domain_decomp0(struct All_variables *E);
void regional_parallel_domain_boundary_nodes(struct All_variables *E);
void regional_parallel_communication_routs_v(struct All_variables *E);
void regional_parallel_communication_routs_s(struct All_variables *E);
void regional_exchange_id_d(struct All_variables *E, double **U, int lev);
void regional_exchange_snode_f(struct All_variables *E, float **U1, float **U2, int lev);
/* Regional_read_input_from_files.c */
void regional_read_input_files_for_timesteps(struct All_variables *E, int action, int output);
/* Regional_solver.c */
void regional_solver_init(struct All_variables *E);
/* Regional_sphere_related.c */
void regional_coord_of_cap(struct All_variables *E, int m, int icap);
/* Regional_tracer_advection.c */
void regional_tracer_setup(struct All_variables *E);
int regional_iget_element(struct All_variables *E, int m, int iprevious_element, double dummy1, double dummy2, double dummy3, double theta, double phi, double rad);
int isearch_all(double *array, int nsize, double a);
int isearch_neighbors(double *array, int nsize, double a, int hint);
int regional_icheck_cap(struct All_variables *E, int icap, double theta, double phi, double rad, double junk);
void regional_get_shape_functions(struct All_variables *E, double shp[9], int nelem, double theta, double phi, double rad);
double regional_interpolate_data(struct All_variables *E, double shp[9], double data[9]);
void regional_get_velocity(struct All_variables *E, int m, int nelem, double theta, double phi, double rad, double *velocity_vector);
void regional_keep_within_bounds(struct All_variables *E, double *x, double *y, double *z, double *theta, double *phi, double *rad);
void regional_lost_souls(struct All_variables *E);
/* Regional_version_dependent.c */
void regional_node_locations(struct All_variables *E);
void regional_construct_boundary(struct All_variables *E);
/* Shape_functions.c */
void construct_shape_functions(struct All_variables *E);
double lpoly(int p, double y);
double lpolydash(int p, double y);
/* Size_does_matter.c */
void twiddle_thumbs(struct All_variables *yawn);
void construct_shape_function_derivatives(struct All_variables *E);
void get_rtf_at_vpts(struct All_variables *E, int m, int lev, int el, double rtf[4][9]);
void get_rtf_at_ppts(struct All_variables *E, int m, int lev, int el, double rtf[4][9]);
void get_side_x_cart(struct All_variables *E, double xx[4][5], int el, int side, int m);
void construct_surf_det(struct All_variables *E);
void construct_bdry_det(struct All_variables *E);
void get_global_1d_shape_fn(struct All_variables *E, int el, struct Shape_function1 *GM, struct Shape_function1_dA *dGammax, int top, int m);
void get_global_1d_shape_fn_L(struct All_variables *E, int el, struct Shape_function1 *GM, struct Shape_function1_dA *dGammax, int top, int m);
void get_global_side_1d_shape_fn(struct All_variables *E, int el, struct Shape_function1 *GM, struct Shape_function1_dx *GMx, struct Shape_function_side_dA *dGamma, int side, int m);
void construct_c3x3matrix_el(struct All_variables *E, int el, struct CC *cc, struct CCX *ccx, int lev, int m, int pressure);
void construct_side_c3x3matrix_el(struct All_variables *E, int el, struct CC *cc, struct CCX *ccx, int lev, int m, int pressure, int side);
void construct_c3x3matrix(struct All_variables *E);
void mass_matrix(struct All_variables *E);
/* Solver_conj_grad.c */
void set_cg_defaults(struct All_variables *E);
void cg_allocate_vars(struct All_variables *E);
void assemble_forces_iterative(struct All_variables *E);
/* Solver_multigrid.c */
void set_mg_defaults(struct All_variables *E);
void mg_allocate_vars(struct All_variables *E);
void inject_scalar(struct All_variables *E, int start_lev, float **AU, float **AD);
void inject_vector(struct All_variables *E, int start_lev, double **AU, double **AD);
void un_inject_vector(struct All_variables *E, int start_lev, double **AD, double **AU);
void interp_vector(struct All_variables *E, int start_lev, double **AD, double **AU);
void project_viscosity(struct All_variables *E);
void inject_scalar_e(struct All_variables *E, int start_lev, float **AU, float **AD);
void project_scalar_e(struct All_variables *E, int start_lev, float **AU, float **AD);
void project_scalar(struct All_variables *E, int start_lev, float **AU, float **AD);
void project_vector(struct All_variables *E, int start_lev, double **AU, double **AD, int ic);
void from_xyz_to_rtf(struct All_variables *E, int level, double **xyz, double **rtf);
void from_rtf_to_xyz(struct All_variables *E, int level, double **rtf, double **xyz);
void fill_in_gaps(struct All_variables *E, double **temp, int level);
/* Sphere_harmonics.c */
void set_sphere_harmonics(struct All_variables *E);
double modified_plgndr_a(int l, int m, double t);
void sphere_expansion(struct All_variables *E, float **TG, float *sphc, float *sphs);
void debug_sphere_expansion(struct All_variables *E);
/* Sphere_util.c */
void even_divide_arc12(int elx, double x1, double y1, double z1, double x2, double y2, double z2, double *theta, double *fi);
void compute_angle_surf_area(struct All_variables *E);
double area_sphere_cap(double angle[6]);
double area_of_sphere_triag(double a, double b, double c);
double area_of_5points(struct All_variables *E, int lev, int m, int el, double x[4], int ne);
void get_angle_sphere_cap(double xx[4][5], double angle[6]);
double get_angle(double x[4], double xx[4]);
/* Stokes_flow_Incomp.c */
void solve_constrained_flow_iterative(struct All_variables *E);
void solve_constrained_flow_iterative_pseudo_surf(struct All_variables *E);
/* Topo_gravity.c */
void get_STD_topo(struct All_variables *E, float **tpg, float **tpgb, float **divg, float **vort, int ii);
void get_STD_freesurf(struct All_variables *E, float **freesurf);
void allocate_STD_mem(struct All_variables *E, float **SXX, float **SYY, float **SZZ, float **SXY, float **SXZ, float **SZY, float **divv, float **vorv);
void free_STD_mem(struct All_variables *E, float **SXX, float **SYY, float **SZZ, float **SXY, float **SXZ, float **SZY, float **divv, float **vorv);
void compute_nodal_stress(struct All_variables *E, float **SXX, float **SYY, float **SZZ, float **SXY, float **SXZ, float **SZY, float **divv, float **vorv);
void stress_conform_bcs(struct All_variables *E);
void compute_geoid(struct All_variables *E);
void get_CBF_topo(struct All_variables *E, float **H, float **HB);
/* Tracer_setup.c */
void tracer_input(struct All_variables *E);
void tracer_initial_settings(struct All_variables *E);
void tracer_advection(struct All_variables *E);
void tracer_post_processing(struct All_variables *E);
void count_tracers_of_flavors(struct All_variables *E);
void initialize_tracers(struct All_variables *E);
void dump_and_get_new_tracers_to_interpolate_fields(struct All_variables *E);
void cart_to_sphere(struct All_variables *E, double x, double y, double z, double *theta, double *phi, double *rad);
void sphere_to_cart(struct All_variables *E, double theta, double phi, double rad, double *x, double *y, double *z);
void get_neighboring_caps(struct All_variables *E);
void allocate_tracer_arrays(struct All_variables *E, int j, int number_of_tracers);
void expand_tracer_arrays(struct All_variables *E, int j);
void expand_later_array(struct All_variables *E, int j);
int icheck_processor_shell(struct All_variables *E, int j, double rad);
int icheck_that_processor_shell(struct All_variables *E, int j, int nprocessor, double rad);
/* Viscosity_structures.c */
void viscosity_system_input(struct All_variables *E);
void viscosity_input(struct All_variables *E);
void get_system_viscosity(struct All_variables *E, int propogate, float **evisc, float **visc);
void initial_viscosity(struct All_variables *E);
void visc_from_mat(struct All_variables *E, float **EEta);
void visc_from_T(struct All_variables *E, float **EEta, int propogate);
void visc_from_S(struct All_variables *E, float **EEta, int propogate);
void visc_from_P(struct All_variables *E, float **EEta);
void visc_from_C(struct All_variables *E, float **EEta);
void strain_rate_2_inv(struct All_variables *E, int m, float *EEDOT, int SQRT);
