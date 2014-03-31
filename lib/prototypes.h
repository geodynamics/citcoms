/* Advection_diffusion.c */
void advection_diffusion_parameters(struct All_variables *);
void advection_diffusion_allocate_memory(struct All_variables *);
void PG_timestep_init(struct All_variables *);
void PG_timestep(struct All_variables *);
void std_timestep(struct All_variables *);
void PG_timestep_solve(struct All_variables *);
/* Advection_diffusion.sz.c */
void advection_diffusion_parameters(struct All_variables *);
void advection_diffusion_allocate_memory(struct All_variables *);
void PG_timestep_init(struct All_variables *);
void PG_timestep(struct All_variables *);
void std_timestep(struct All_variables *);
void PG_timestep_solve(struct All_variables *);
/* Anisotropic_viscosity.c */
void get_constitutive(double [6][6], double, double, int, float, float, float, float, int, struct All_variables *);
void get_constitutive_ti_viscosity(double [6][6], double, double, double [3], int, double, double);
void get_constitutive_orthotropic_viscosity(double [6][6], double, double [3], int, double, double);
void get_constitutive_isotropic(double [6][6]);
void set_anisotropic_viscosity_at_element_level(struct All_variables *, int);
void normalize_director_at_nodes(struct All_variables *, float **, float **, float **, int);
void normalize_director_at_gint(struct All_variables *, float **, float **, float **, int);
void conv_cart4x4_to_spherical(double [3][3][3][3], double, double, double [3][3][3][3]);
void conv_cart6x6_to_spherical(double [6][6], double, double, double [6][6]);
void rotate_ti6x6_to_director(double [6][6], double [3]);
void get_citcom_spherical_rot(double, double, double [3][3]);
void get_orth_delta(double [3][3][3][3], double [3]);
void align_director_with_ISA_for_element(struct All_variables *, int);
void get_vgm_p(double [4][9], struct Shape_function *, struct Shape_function_dx *, struct CC *, struct CCX *, double [4][9], int, int, int, int, double [3][3], double [3]);
unsigned char calc_isa_from_vgm(double [3][3], double [3], int, double [3], struct All_variables *, int);
int is_pure_shear(double [3][3], double [3][3], double [3][3]);
void rot_4x4(double [3][3][3][3], double [3][3], double [3][3][3][3]);
void zero_6x6(double [6][6]);
void zero_4x4(double [3][3][3][3]);
void copy_4x4(double [3][3][3][3], double [3][3][3][3]);
void copy_6x6(double [6][6], double [6][6]);
void print_6x6_mat(FILE *, double [6][6]);
void print_3x3_mat(FILE *, double [3][3]);
void c4fromc6(double [3][3][3][3], double [6][6]);
void c6fromc4(double [6][6], double [3][3][3][3]);
void isacalc(double [3][3], double *, double [3], struct All_variables *, int *);
void f_times_ft(double [3][3], double [3][3]);
void drex_eigen(double [3][3], double [3][3], int *);
void malmul_scaled_id(double [3][3], double [3][3], double, double);
void myerror_s(char *, struct All_variables *);
/* BC_util.c */
void internal_horizontal_bc(struct All_variables *, float *[], int, int, float, unsigned int, char, int, int);
void strip_bcs_from_residual(struct All_variables *, double **, int);
void temperatures_conform_bcs(struct All_variables *);
void temperatures_conform_bcs2(struct All_variables *);
void velocities_conform_bcs(struct All_variables *, double **);
void assign_internal_bc(struct All_variables *);
/* Checkpoints.c */
void output_checkpoint(struct All_variables *);
void read_checkpoint(struct All_variables *);
/* Citcom_init.c */
struct All_variables *citcom_init(MPI_Comm *);
void citcom_finalize(struct All_variables *, int);
/* Composition_related.c */
void composition_input(struct All_variables *);
void composition_setup(struct All_variables *);
void write_composition_instructions(struct All_variables *);
void fill_composition(struct All_variables *);
void init_composition(struct All_variables *);
void map_composition_to_nodes(struct All_variables *);
void get_bulk_composition(struct All_variables *);
/* Construct_arrays.c */
void construct_ien(struct All_variables *);
void construct_surface(struct All_variables *);
void construct_id(struct All_variables *);
void get_bcs_id_for_residual(struct All_variables *, int, int);
void construct_lm(struct All_variables *);
void construct_node_maps(struct All_variables *);
void construct_node_ks(struct All_variables *);
void rebuild_BI_on_boundary(struct All_variables *);
void construct_masks(struct All_variables *);
void construct_sub_element(struct All_variables *);
void construct_elt_ks(struct All_variables *);
void construct_elt_gs(struct All_variables *);
void construct_elt_cs(struct All_variables *);
void construct_stiffness_B_matrix(struct All_variables *);
int layers_r(struct All_variables *, float);
int layers(struct All_variables *, int, int);
void construct_mat_group(struct All_variables *);
/* Convection.c */
void set_convection_defaults(struct All_variables *);
void read_convection_settings(struct All_variables *);
void convection_derived_values(struct All_variables *);
void convection_allocate_memory(struct All_variables *);
void convection_initial_fields(struct All_variables *);
void convection_boundary_conditions(struct All_variables *);
/* Determine_net_rotation.c */
double determine_model_net_rotation(struct All_variables *, double *);
double determine_netr_tp(float, float, float, float, float, int, double *, double *);
void sub_netr(float, float, float, float *, float *, double *);
void hc_ludcmp_3x3(double [3][3], int, int *);
void hc_lubksb_3x3(double [3][3], int, int *, double *);
/* Drive_solvers.c */
void general_stokes_solver_setup(struct All_variables *);
void general_stokes_solver(struct All_variables *);
int need_visc_update(struct All_variables *);
int need_to_iterate(struct All_variables *);
void general_stokes_solver_pseudo_surf(struct All_variables *);
/* Element_calculations.c */
void assemble_forces(struct All_variables *, int);
void get_ba(struct Shape_function *, struct Shape_function_dx *, struct CC *, struct CCX *, double [4][9], int, double [9][9][4][7]);
void get_ba_p(struct Shape_function *, struct Shape_function_dx *, struct CC *, struct CCX *, double [4][9], int, double [9][9][4][7]);
void get_elt_k(struct All_variables *, int, double [24*24], int, int, int);
void assemble_del2_u(struct All_variables *, double **, double **, int, int);
void e_assemble_del2_u(struct All_variables *, double **, double **, int, int);
void n_assemble_del2_u(struct All_variables *, double **, double **, int, int);
void build_diagonal_of_K(struct All_variables *, int, double [24*24], int, int);
void build_diagonal_of_Ahat(struct All_variables *);
void assemble_c_u(struct All_variables *, double **, double **, int);
void assemble_div_rho_u(struct All_variables *, double **, double **, int);
void assemble_div_u(struct All_variables *, double **, double **, int);
void assemble_grad_p(struct All_variables *, double **, double **, int);
double assemble_dAhatp_entry(struct All_variables *, int, int, int);
void get_elt_c(struct All_variables *, int, higher_precision [24][1], int, int);
void get_elt_g(struct All_variables *, int, higher_precision [24][1], int, int);
void get_elt_f(struct All_variables *, int, double [24], int, int);
void get_aug_k(struct All_variables *, int, double [24*24], int, int);
/* Full_boundary_conditions.c */
void full_velocity_boundary_conditions(struct All_variables *);
void full_temperature_boundary_conditions(struct All_variables *);
/* Full_geometry_cartesian.c */
void full_set_2dc_defaults(struct All_variables *);
void full_set_2pt5dc_defaults(struct All_variables *);
void full_set_3dc_defaults(struct All_variables *);
void full_set_3dsphere_defaults(struct All_variables *);
/* Full_lith_age_read_files.c */
void full_lith_age_read_files(struct All_variables *, int);
/* Full_parallel_related.c */
void full_parallel_processor_setup(struct All_variables *);
void full_parallel_domain_decomp0(struct All_variables *);
void full_parallel_domain_boundary_nodes(struct All_variables *);
void full_parallel_communication_routs_v(struct All_variables *);
void full_parallel_communication_routs_s(struct All_variables *);
void full_exchange_id_d(struct All_variables *, double **, int);
void full_exchange_snode_f(struct All_variables *, float **, float **, int);
/* Full_read_input_from_files.c */
void full_read_input_files_for_timesteps(struct All_variables *, int, int);
/* Full_solver.c */
void full_solver_init(struct All_variables *);
/* Full_sphere_related.c */
void spherical_to_uv2(double [2], int, double *, double *, double *, double *);
void uv_to_spherical(double [2], int, double *, double *, double *, double *);
void full_coord_of_cap(struct All_variables *, int, int);
/* Full_tracer_advection.c */
void full_tracer_input(struct All_variables *);
void full_tracer_setup(struct All_variables *);
void full_lost_souls(struct All_variables *);
void full_get_shape_functions(struct All_variables *, double [9], int, double, double, double);
double full_interpolate_data(struct All_variables *, double [9], double [9]);
void full_get_velocity(struct All_variables *, int, int, double, double, double, double *);
int full_icheck_cap(struct All_variables *, int, double, double, double, double);
int full_iget_element(struct All_variables *, int, int, double, double, double, double, double, double);
void full_keep_within_bounds(struct All_variables *, double *, double *, double *, double *, double *, double *);
void analytical_test(struct All_variables *);
void analytical_runge_kutte(struct All_variables *, int, double, double *, double *, double *, double *, double *);
void analytical_test_function(struct All_variables *, double, double, double, double *, double *);
void pdebug(struct All_variables *, int);
/* Full_version_dependent.c */
void full_node_locations(struct All_variables *);
void full_construct_boundary(struct All_variables *);
/* General_matrix_functions.c */
int solve_del2_u(struct All_variables *, double **, double **, double, int);
double multi_grid(struct All_variables *, double **, double **, double, int);
double conj_grad(struct All_variables *, double **, double **, double, int *, int);
void element_gauss_seidel(struct All_variables *, double **, double **, double **, double, int *, int, int);
void gauss_seidel(struct All_variables *, double **, double **, double **, double, int *, int, int);
double determinant(double [4][4], int);
double cofactor(double [4][4], int, int, int);
long double lg_pow(long double, int);
#ifdef USE_GGRD
/* Ggrd_handling.c */
void ggrd_init_tracer_flavors(struct All_variables *);
void ggrd_full_temp_init(struct All_variables *);
void ggrd_reg_temp_init(struct All_variables *);
void ggrd_temp_init_general(struct All_variables *, int);
void ggrd_read_mat_from_file(struct All_variables *, int);
void ggrd_read_ray_from_file(struct All_variables *, int);
void ggrd_read_vtop_from_file(struct All_variables *, int);
void ggrd_vtop_helper_decide_on_internal_nodes(struct All_variables *, int, int, int, int, int, int *, int *, int *);
void ggrd_read_age_from_file(struct All_variables *, int);
void ggrd_adjust_tbl_rayleigh(struct All_variables *, double **);
void ggrd_solve_eigen3x3(double [3][3], double [3], double [3][3], struct All_variables *);
void ggrd_read_anivisc_from_file(struct All_variables *, int);
#endif
/* Global_operations.c */
void remove_horiz_ave(struct All_variables *, double **, double *, int);
void remove_horiz_ave2(struct All_variables *, double **);
void return_horiz_ave(struct All_variables *, double **, double *);
void return_horiz_ave_f(struct All_variables *, float **, float *);
void return_elementwise_horiz_ave(struct All_variables *, double **, double *);
float return_bulk_value(struct All_variables *, float **, int);
double return_bulk_value_d(struct All_variables *, double **, int);
float find_max_horizontal(struct All_variables *, double);
void sum_across_surface(struct All_variables *, float *, int);
void sum_across_surf_sph1(struct All_variables *, float *, float *);
float global_fvdot(struct All_variables *, float **, float **, int);
double kineticE_radial(struct All_variables *, double **, int);
double global_vdot(struct All_variables *, double **, double **, int);
double global_pdot(struct All_variables *, double **, double **, int);
double global_v_norm2(struct All_variables *, double **);
double global_p_norm2(struct All_variables *, double **);
double global_div_norm2(struct All_variables *, double **);
double global_tdot_d(struct All_variables *, double **, double **, int);
float global_tdot(struct All_variables *, float **, float **, int);
float global_fmin(struct All_variables *, double);
double global_dmax(struct All_variables *, double);
float global_fmax(struct All_variables *, double);
double Tmaxd(struct All_variables *, double **);
float Tmax(struct All_variables *, float **);
double vnorm_nonnewt(struct All_variables *, double **, double **, int);
void sum_across_depth_sph1(struct All_variables *, float *, float *);
void broadcast_vertical(struct All_variables *, float *, float *, int);
void remove_rigid_rot(struct All_variables *);

/* Initial_temperature.c */
void tic_input(struct All_variables *);
void convection_initial_temperature(struct All_variables *);
/* Instructions.c */
void initial_mesh_solver_setup(struct All_variables *);
void read_instructions(struct All_variables *, char *);
void initial_setup(struct All_variables *);
void initialize_material(struct All_variables *);
void initial_conditions(struct All_variables *);
void read_initial_settings(struct All_variables *);
void check_settings_consistency(struct All_variables *);
void global_derived_values(struct All_variables *);
void allocate_common_vars(struct All_variables *);
void allocate_velocity_vars(struct All_variables *);
void global_default_values(struct All_variables *);
void check_bc_consistency(struct All_variables *);
void set_up_nonmg_aliases(struct All_variables *, int);
void report(struct All_variables *, char *);
void record(struct All_variables *, char *);
void common_initial_fields(struct All_variables *);
void initial_pressure(struct All_variables *);
void initial_velocity(struct All_variables *);
void open_qfiles(struct All_variables *);
void mkdatadir(const char *);
void output_init(struct All_variables *);
void output_finalize(struct All_variables *);
char *strip(char *);
/* Interuption.c */
void interuption(int);
void set_signal(void);
/* Lith_age.c */
void lith_age_input(struct All_variables *);
void lith_age_init(struct All_variables *);
void lith_age_construct_tic(struct All_variables *);
void lith_age_update_tbc(struct All_variables *);
void lith_age_temperature_bound_adj(struct All_variables *, int);
void lith_age_conform_tbc(struct All_variables *);
void assimilate_lith_conform_bcs(struct All_variables *);
/* Material_properties.c */
void mat_prop_allocate(struct All_variables *);
void reference_state(struct All_variables *);
/* Mineral_physics_models.c */
void get_prem(double, double *, double *, double *);
void compute_seismic_model(struct All_variables *, double *, double *, double *);
/* Nodal_mesh.c */
void v_from_vector(struct All_variables *);
void assign_v_to_vector(struct All_variables *);
void v_from_vector_pseudo_surf(struct All_variables *);
void velo_from_element(struct All_variables *, float [4][9], int, int, int);
void velo_from_element_d(struct All_variables *, double [4][9], int, int, int);
void p_to_nodes(struct All_variables *, double **, float **, int);
void visc_from_gint_to_nodes(struct All_variables *, float **, float **, int);
void visc_from_nodes_to_gint(struct All_variables *, float **, float **, int);
void visc_from_gint_to_ele(struct All_variables *, float **, float **, int);
void visc_from_ele_to_gint(struct All_variables *, float **, float **, int);
/* Output.c */
void output_common_input(struct All_variables *);
void output(struct All_variables *, int);
FILE *output_open(char *, char *);
void output_coord(struct All_variables *);
void output_domain(struct All_variables *);
void output_coord_bin(struct All_variables *);
void output_visc(struct All_variables *, int);
void output_avisc(struct All_variables *, int);
void output_velo(struct All_variables *, int);
void output_surf_botm(struct All_variables *, int);
void output_geoid(struct All_variables *, int);
void output_stress(struct All_variables *, int);
void output_horiz_avg(struct All_variables *, int);
void output_volume_avg(struct All_variables *, int);
void output_seismic(struct All_variables *, int);
void output_mat(struct All_variables *);
void output_pressure(struct All_variables *, int);
void output_tracer(struct All_variables *, int);
void output_comp_nd(struct All_variables *, int);
void output_comp_el(struct All_variables *, int);
void output_heating(struct All_variables *, int);
void output_time(struct All_variables *, int);
#ifdef USE_GZDIR
/* Output_gzdir.c */
void gzdir_output(struct All_variables *, int);
gzFile *gzdir_output_open(char *, char *);
void gzdir_output_coord(struct All_variables *);
void gzdir_output_velo_temp(struct All_variables *, int);
void gzdir_output_visc(struct All_variables *, int);
void gzdir_output_avisc(struct All_variables *, int);
void gzdir_output_surf_botm(struct All_variables *, int);
void gzdir_output_geoid(struct All_variables *, int);
void gzdir_output_stress(struct All_variables *, int);
void gzdir_output_horiz_avg(struct All_variables *, int);
void gzdir_output_mat(struct All_variables *);
void gzdir_output_pressure(struct All_variables *, int);
void gzdir_output_tracer(struct All_variables *, int);
void gzdir_output_comp_nd(struct All_variables *, int);
void gzdir_output_comp_el(struct All_variables *, int);
void gzdir_output_heating(struct All_variables *, int);
void restart_tic_from_gzdir_file(struct All_variables *);
int open_file_zipped(char *, FILE **, struct All_variables *);
void gzip_file(char *);
void get_vtk_filename(char *, int, struct All_variables *, int);
int be_write_float_to_file(float *, int, FILE *);
int be_write_int_to_file(int *, int, FILE *);
void myfprintf(FILE *, char *);
int be_is_little_endian(void);
void be_flip_byte_order(void *, size_t);
void be_flipit(void *, void *, size_t);
#endif
/* Output_h5.c */
void h5output_allocate_memory(struct All_variables *);
void h5output(struct All_variables *, int);
void h5input_params(struct All_variables *);
/* Output_vtk.c */
void vtk_output(struct All_variables *, int);
/* Pan_problem_misc_functions.c */
int get_process_identifier(void);
void unique_copy_file(struct All_variables *, char *, char *);
void apply_side_sbc(struct All_variables *);
void get_buoyancy(struct All_variables *, double **);
int read_double_vector(FILE *, int, double *);
void read_visc_param_from_file(struct All_variables *, const char *, float *, FILE *);
double myatan(double, double);
double return1_test(void);
void rtp2xyzd(double, double, double, double *);
void rtp2xyz(float, float, float, float *);
void xyz2rtp(float, float, float, float *);
void xyz2rtpd(float, float, float, double *);
void calc_cbase_at_tp(float, float, float *);
void calc_cbase_at_tp_d(double, double, double *);
void calc_cbase_at_node(int, int, float *, struct All_variables *);
void convert_pvec_to_cvec(float, float, float, float *, float *);
void convert_pvec_to_cvec_d(double, double, double, double *, double *);
void *safe_malloc(size_t);
void myerror(struct All_variables *, char *);
void get_r_spacing_fine(double *, struct All_variables *);
void get_r_spacing_at_levels(double *, struct All_variables *);
void normalize_vec3(float *, float *, float *);
void normalize_vec3d(double *, double *, double *);
void matmul_3x3(double [3][3], double [3][3], double [3][3]);
void remove_trace_3x3(double [3][3]);
void get_9vec_from_3x3(double *, double [3][3]);
void get_3x3_from_9vec(double [3][3], double *);
/* Parallel_util.c */
void parallel_process_finalize(void);
void parallel_process_termination(void);
void parallel_process_sync(struct All_variables *);
double CPU_time0(void);
/* Parsing.c */
void setup_parser(struct All_variables *, char *);
void shutdown_parser(struct All_variables *);
void add_to_parameter_list(char *, char *);
int compute_parameter_hash_table(char *);
int input_int(char *, int *, char *, int);
int input_string(char *, char *, char *, int);
int input_boolean(char *, int *, char *, int);
int input_float(char *, float *, char *, int);
int input_double(char *, double *, char *, int);
int input_int_vector(char *, int, int *, int);
int input_char_vector(char *, int, char *, int);
int input_float_vector(char *, int, float *, int);
int input_double_vector(char *, int, double *, int);
int interpret_control_string(char *, int *, double *, double *, double *);
/* Phase_change.c */
void phase_change_allocate(struct All_variables *);
void phase_change_input(struct All_variables *);
void phase_change_apply_410(struct All_variables *, double **);
void phase_change_apply_670(struct All_variables *, double **);
void phase_change_apply_cmb(struct All_variables *, double **);
/* Problem_related.c */
void read_velocity_boundary_from_file(struct All_variables *);
void read_rayleigh_from_file(struct All_variables *);
void read_mat_from_file(struct All_variables *);
void read_temperature_boundary_from_file(struct All_variables *);
void get_initial_elapsed_time(struct All_variables *);
void set_elapsed_time(struct All_variables *);
void set_starting_age(struct All_variables *);
float find_age_in_MY(struct All_variables *);
/* Process_buoyancy.c */
void post_processing(struct All_variables *);
void heat_flux(struct All_variables *);
void compute_horiz_avg(struct All_variables *);
void compute_volume_avg(struct All_variables *, float *, float *);
/* Regional_boundary_conditions.c */
void regional_velocity_boundary_conditions(struct All_variables *);
void regional_temperature_boundary_conditions(struct All_variables *);
/* Regional_geometry_cartesian.c */
void regional_set_2dc_defaults(struct All_variables *);
void regional_set_2pt5dc_defaults(struct All_variables *);
void regional_set_3dc_defaults(struct All_variables *);
void regional_set_3dsphere_defaults(struct All_variables *);
/* Regional_lith_age_read_files.c */
void regional_lith_age_read_files(struct All_variables *, int);
/* Regional_parallel_related.c */
void regional_parallel_processor_setup(struct All_variables *);
void regional_parallel_domain_decomp0(struct All_variables *);
void regional_parallel_domain_boundary_nodes(struct All_variables *);
void regional_parallel_communication_routs_v(struct All_variables *);
void regional_parallel_communication_routs_s(struct All_variables *);
void regional_exchange_id_d(struct All_variables *, double **, int);
void regional_exchange_snode_f(struct All_variables *, float **, float **, int);
/* Regional_read_input_from_files.c */
void regional_read_input_files_for_timesteps(struct All_variables *, int, int);
/* Regional_solver.c */
void regional_solver_init(struct All_variables *);
/* Regional_sphere_related.c */
void regional_coord_of_cap(struct All_variables *, int, int);
/* Regional_tracer_advection.c */
void regional_tracer_setup(struct All_variables *);
int regional_iget_element(struct All_variables *, int, int, double, double, double, double, double, double);
int isearch_all(double *, int, double);
int isearch_neighbors(double *, int, double, int);
int regional_icheck_cap(struct All_variables *, int, double, double, double, double);
void regional_get_shape_functions(struct All_variables *, double [9], int, double, double, double);
double regional_interpolate_data(struct All_variables *, double [9], double [9]);
void regional_get_velocity(struct All_variables *, int, int, double, double, double, double *);
void regional_keep_within_bounds(struct All_variables *, double *, double *, double *, double *, double *, double *);
void regional_lost_souls(struct All_variables *);
/* Regional_version_dependent.c */
void regional_node_locations(struct All_variables *);
void regional_construct_boundary(struct All_variables *);
/* Shape_functions.c */
void construct_shape_functions(struct All_variables *);
double lpoly(int, double);
double lpolydash(int, double);
/* Size_does_matter.c */
void twiddle_thumbs(struct All_variables *);
void construct_shape_function_derivatives(struct All_variables *);
void get_rtf_at_vpts(struct All_variables *, int, int, int, double [4][9]);
void get_rtf_at_ppts(struct All_variables *, int, int, int, double [4][9]);
void get_side_x_cart(struct All_variables *, double [4][5], int, int, int);
void construct_surf_det(struct All_variables *);
void construct_bdry_det(struct All_variables *);
void get_global_1d_shape_fn(struct All_variables *, int, struct Shape_function1 *, struct Shape_function1_dA *, int, int);
void get_global_1d_shape_fn_L(struct All_variables *, int, struct Shape_function1 *, struct Shape_function1_dA *, int, int);
void get_global_side_1d_shape_fn(struct All_variables *, int, struct Shape_function1 *, struct Shape_function1_dx *, struct Shape_function_side_dA *, int, int);
void construct_c3x3matrix_el(struct All_variables *, int, struct CC *, struct CCX *, int, int, int);
void construct_side_c3x3matrix_el(struct All_variables *, int, struct CC *, struct CCX *, int, int, int, int);
void construct_c3x3matrix(struct All_variables *);
void mass_matrix(struct All_variables *);
/* Solver_conj_grad.c */
void set_cg_defaults(struct All_variables *);
void cg_allocate_vars(struct All_variables *);
void assemble_forces_iterative(struct All_variables *);
/* Solver_multigrid.c */
void set_mg_defaults(struct All_variables *);
void mg_allocate_vars(struct All_variables *);
void inject_scalar(struct All_variables *, int, float **, float **);
void inject_vector(struct All_variables *, int, double **, double **);
void un_inject_vector(struct All_variables *, int, double **, double **);
void interp_vector(struct All_variables *, int, double **, double **);
void project_viscosity(struct All_variables *);
void inject_scalar_e(struct All_variables *, int, float **, float **);
void project_scalar_e(struct All_variables *, int, float **, float **);
void project_scalar(struct All_variables *, int, float **, float **);
void project_vector(struct All_variables *, int, double **, double **, int);
void from_xyz_to_rtf(struct All_variables *, int, double **, double **);
void from_rtf_to_xyz(struct All_variables *, int, double **, double **);
void fill_in_gaps(struct All_variables *, double **, int);
/* Sphere_harmonics.c */
void set_sphere_harmonics(struct All_variables *);
double modified_plgndr_a(int, int, double);
void sphere_expansion(struct All_variables *, float **, float *, float *);
void debug_sphere_expansion(struct All_variables *);
/* Sphere_util.c */
void even_divide_arc12(int, double, double, double, double, double, double, double *, double *);
void compute_angle_surf_area(struct All_variables *);
double area_sphere_cap(double [6]);
double area_of_sphere_triag(double, double, double);
double area_of_5points(struct All_variables *, int, int, int, double [4], int);
void get_angle_sphere_cap(double [4][5], double [6]);
double get_angle(double [4], double [4]);
/* Stokes_flow_Incomp.c */
void solve_constrained_flow_iterative(struct All_variables *);
/* Topo_gravity.c */
void get_STD_topo(struct All_variables *, float **, float **, float **, float **, int);
void get_STD_freesurf(struct All_variables *, float **);
void allocate_STD_mem(struct All_variables *, float **, float **, float **, float **, float **, float **, float **, float **);
void free_STD_mem(struct All_variables *, float **, float **, float **, float **, float **, float **, float **, float **);
void compute_nodal_stress(struct All_variables *, float **, float **, float **, float **, float **, float **, float **, float **);
void stress_conform_bcs(struct All_variables *);
void compute_geoid(struct All_variables *);
void get_CBF_topo(struct All_variables *, float **, float **);
/* Tracer_setup.c */
void tracer_input(struct All_variables *);
void tracer_initial_settings(struct All_variables *);
void tracer_advection(struct All_variables *);
void tracer_post_processing(struct All_variables *);
void count_tracers_of_flavors(struct All_variables *);
void initialize_tracers(struct All_variables *);
void cart_to_sphere(struct All_variables *, double, double, double, double *, double *, double *);
void sphere_to_cart(struct All_variables *, double, double, double, double *, double *, double *);
void get_neighboring_caps(struct All_variables *);
void allocate_tracer_arrays(struct All_variables *, int, int);
void expand_tracer_arrays(struct All_variables *, int);
void expand_later_array(struct All_variables *, int);
int icheck_processor_shell(struct All_variables *, int, double);
int icheck_that_processor_shell(struct All_variables *, int, int, double);
/* Viscosity_structures.c */
void viscosity_system_input(struct All_variables *);
void viscosity_input(struct All_variables *);
void allocate_visc_vars(struct All_variables *);
void get_system_viscosity(struct All_variables *, int, float **, float **);
void initial_viscosity(struct All_variables *);
void visc_from_mat(struct All_variables *, float **);
void read_visc_layer_file(struct All_variables *);
void visc_from_T(struct All_variables *, float **, int);
void visc_from_S(struct All_variables *, float **, int);
void visc_from_P(struct All_variables *, float **);
void visc_from_C(struct All_variables *, float **);
void strain_rate_2_inv(struct All_variables *, int, float *, int);
double second_invariant_from_3x3(double [3][3]);
void calc_strain_from_vgm(double [3][3], double [3][3]);
void calc_strain_from_vgm9(double *, double [3][3]);
void calc_rot_from_vgm(double [3][3], double [3][3]);
