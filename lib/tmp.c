void print_all_config_parameters(struct All_variables *E)
{
  char filename[16];
  FILE *fp;
  int i,j,k;
  float tmp = 0.0f;

  if(E->parallel.me == 0) {
    sprintf(filename, "pid%09d", E->control.PID);
    fp = fopen(filename, "w");
    
    /* CitcomS*/
    fprintf(fp, "# CitcomS\n");
    fprintf(fp, "minstep=%d\n", E->advection.min_timesteps);
    fprintf(fp, "maxstep=%d\n", E->advection.max_timesteps);
    fprintf(fp, "maxtotstep=%d\n", E->advection.max_total_timesteps);
    fprintf(fp, "cpu_limits_in_seconds=%d\n", E->control.record_all_until);
    fprintf(fp, "\n\n");

    /* CitcomS.controller */
    fprintf(fp, "# CitcomS.controller\n");
    fprintf(fp, "storage_spacing=%d\n", E->control.record_every);
    fprintf(fp, "checkpointFrequency=%d\n", E->control.checkpoint_frequency);
    fprintf(fp, "\n\n");

    /* CitcomS.solver */
    fprintf(fp, "# CitcomS.solver\n");
    fprintf(fp, "datadir=%s\n", E->control.data_dir);
    fprintf(fp, "datafile=%s\n", E->control.data_prefix);
    fprintf(fp, "datadir_old=%s\n", E->control.data_dir_old);
    fprintf(fp, "datafile_old=%s\n", E->control.data_prefix_old);
    fprintf(fp, "rayleigh=%g\n", E->control.Atemp);
    fprintf(fp, "dissipation_number=%g\n", E->control.disptn_number);
    if(E->control.inv_gruneisen != 0)
      tmp = 1.0/E->control.inv_gruneisen;
    fprintf(fp, "gruneisen=%g\n", tmp);
    fprintf(fp, "surfaceT=%g\n", E->control.surface_temp);
    fprintf(fp, "Q0=%g\n", E->control.Q0);
    fprintf(fp, "stokes_flow_only=%d\n", E->control.stokes);
    fprintf(fp, "verbose=%d\n", E->control.verbose);
    fprintf(fp, "see_convergence=%d\n", E->control.print_convergence);
    fprintf(fp, "\n\n");

    /* CitcomS.solver.mesher */
    fprintf(fp, "# CitcomS.solver.mesher\n");
    fprintf(fp, "nproc_surf=%d\n", E->parallel.nprocxy);
    fprintf(fp, "nprocx=%d\n", E->parallel.nprocx);
    fprintf(fp, "nprocy=%d\n", E->parallel.nprocy);
    fprintf(fp, "nprocz=%d\n", E->parallel.nprocz);
    fprintf(fp, "coor=%d\n", E->control.coor);
    fprintf(fp, "coor_file=%s\n", E->control.coor_file);
    fprintf(fp, "coor_refine=");
    for(i=0; i<3; i++)
      fprintf(fp, "%g,", E->control.coor_refine[i]);
    fprintf(fp, "%g\n", E->control.coor_refine[3]);
    fprintf(fp, "nodex=%d\n", E->mesh.nox);
    fprintf(fp, "nodey=%d\n", E->mesh.noy);
    fprintf(fp, "nodez=%d\n", E->mesh.noz);
    fprintf(fp, "levels=%d\n", E->mesh.levels);
    fprintf(fp, "mgunitx=%d\n", E->mesh.mgunitx);
    fprintf(fp, "mgunity=%d\n", E->mesh.mgunity);
    fprintf(fp, "mgunitz=%d\n", E->mesh.mgunitz);
    fprintf(fp, "radius_outer=%g\n", E->sphere.ro);
    fprintf(fp, "radius_inner=%g\n", E->sphere.ri);
    fprintf(fp, "theta_min=%g\n", E->control.theta_min);
    fprintf(fp, "theta_max=%g\n", E->control.theta_max);
    fprintf(fp, "fi_min=%g\n", E->control.fi_min);
    fprintf(fp, "fi_max=%g\n", E->control.fi_max);
    fprintf(fp, "r_grid_layers=%d\n", E->control.rlayers);
    fprintf(fp, "rr=");
    if(E->control.rlayers > 0)
    {
      for(i=0; i<E->control.rlayers-1;i++)
	fprintf(fp, "%g,", E->control.rrlayer[i]);
      fprintf(fp, "%g\n", E->control.rrlayer[E->control.rlayers-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "nr=");
    if(E->control.rlayers > 0)
    {
      for(i=0; i<E->control.rlayers-1;i++)
        fprintf(fp, "%d,", E->control.nrlayer[i]);
      fprintf(fp, "%d\n", E->control.nrlayer[E->control.rlayers-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.tsolver\n");
    fprintf(fp, "ADV=%d\n", E->advection.ADVECTION);
    fprintf(fp, "filter_temp=%d\n", E->advection.filter_temperature);
    fprintf(fp, "monitor_max_T=%d\n", E->advection.monitor_max_T);
    fprintf(fp, "finetunedt=%f\n", E->advection.fine_tune_dt);
    fprintf(fp, "fixed_timestep=%f\n", E->advection.fixed_timestep);
    fprintf(fp, "adv_gamma=%f\n", E->advection.gamma);
    fprintf(fp, "adv_sub_iterations=%d\n", E->advection.temp_iterations);
    fprintf(fp, "inputdiffusivity=%f\n", E->control.inputdiff);
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.vsolver\n");
    fprintf(fp, "Solver=%s\n", E->control.SOLVER_TYPE); 
    fprintf(fp, "node_assemble=%d\n", E->control.NASSEMBLE);
    fprintf(fp, "precond=%d\n", E->control.precondition);
    fprintf(fp, "accuracy=%g\n", E->control.accuracy);
    fprintf(fp, "uzawa=%s\n", E->control.uzawa);
    fprintf(fp, "compress_iter_maxstep=%d\n", E->control.compress_iter_maxstep);
    fprintf(fp, "mg_cycle=%d\n", E->control.mg_cycle);
    fprintf(fp, "down_heavy=%d\n", E->control.down_heavy);
    fprintf(fp, "up_heavy=%d\n", E->control.up_heavy);
    fprintf(fp, "vlowstep=%d\n", E->control.v_steps_low);
    fprintf(fp, "vhighstep=%d\n", E->control.v_steps_high);
    fprintf(fp, "max_mg_cycles=%d\n", E->control.max_mg_cycles);
    fprintf(fp, "piterations=%d\n", E->control.p_iterations);
    fprintf(fp, "aug_lagr=%d\n", E->control.augmented_Lagr);
    fprintf(fp, "aug_number=%g\n", E->control.augmented);
    fprintf(fp, "remove_rigid_rotation=%d\n", E->control.remove_rigid_rotation);
    fprintf(fp, "remove_angular_momentum=%d\n", 
                E->control.remove_angular_momentum);
    fprintf(fp, "inner_accuracy_scale=%g\n", 
                E->control.inner_accuracy_scale);
    fprintf(fp, "check_continuity_convergence=%d\n", 
                E->control.check_continuity_convergence);
    fprintf(fp, "check_pressure_convergence=%d\n", 
                E->control.check_pressure_convergence);
    fprintf(fp, "inner_remove_rigid_rotation=%d\n", 
                E->control.inner_remove_rigid_rotation);
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.bc\n");
    fprintf(fp, "side_sbcs=%d\n", E->control.side_sbcs);
    fprintf(fp, "pseudo_free_surf=%d\n", E->control.pseudo_free_surf);
    fprintf(fp, "topvbc=%d\n", E->mesh.topvbc);
    fprintf(fp, "topvbxval=%g\n", E->control.VBXtopval);
    fprintf(fp, "topvbyval=%g\n", E->control.VBYtopval);
    fprintf(fp, "botvbc=%d\n", E->mesh.botvbc);
    fprintf(fp, "botvbxval=%g\n", E->control.VBXbotval);
    fprintf(fp, "botvbyval=%g\n", E->control.VBYbotval);
    fprintf(fp, "toptbc=%d\n", E->mesh.toptbc);
    fprintf(fp, "toptbcval=%g\n", E->control.TBCtopval);
    fprintf(fp, "bottbc=%d\n", E->mesh.bottbc);
    fprintf(fp, "bottbcval=%g\n", E->control.TBCbotval);
    fprintf(fp, "temperature_bound_adj=%d\n", E->control.temperature_bound_adj);
    fprintf(fp, "depth_bound_adj=%g\n", E->control.depth_bound_adj);
    fprintf(fp, "width_bound_adj=%g\n", E->control.width_bound_adj);
    fprintf(fp, "\n\n");
    
    fprintf(fp, "# CitcomS.solver.const\n");
    fprintf(fp, "radius=%g\n", E->data.radius_km*1000.0);
    fprintf(fp, "density=%g\n", E->data.density);
    fprintf(fp, "thermdiff=%g\n", E->data.therm_diff);
    fprintf(fp, "gravacc=%g\n", E->data.grav_acc);
    fprintf(fp, "thermexp=%g\n", E->data.therm_exp);
    fprintf(fp, "refvisc=%g\n", E->data.ref_viscosity);
    fprintf(fp, "cp=%g\n", E->data.Cp);
    fprintf(fp, "density_above=%g\n", E->data.density_above);
    fprintf(fp, "density_below=%g\n", E->data.density_below);
    fprintf(fp, "z_lith=%g\n", E->viscosity.zlith);
    fprintf(fp, "z_410=%g\n", E->viscosity.z410);
    fprintf(fp, "z_lmantle=%g\n", E->viscosity.zlm);
    fprintf(fp, "z_cmb=%g\n", E->viscosity.zcmb);
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.ic\n");
    fprintf(fp, "restart=%d\n", E->control.restart);
    fprintf(fp, "post_p=%d\n", E->control.post_p);
    fprintf(fp, "solution_cycles_init=%d\n", E->monitor.solution_cycles_init);
    fprintf(fp, "zero_elapsed_time=%d\n", E->control.zero_elapsed_time);
    fprintf(fp, "tic_method=%d\n", E->convection.tic_method);
    fprintf(fp, "num_perturbations=%d\n", E->convection.number_of_perturbations);
    if(E->convection.number_of_perturbations > 1)
    {
      fprintf(fp, "perturbl=");
      for(i=0; i<E->convection.number_of_perturbations-1; i++)
	fprintf(fp, "%d,", E->convection.perturb_ll[i]);
      fprintf(fp, "%d\n", E->convection.perturb_ll[E->convection.number_of_perturbations-1]);
      fprintf(fp, "perturbm=");
      for(i=0; i<E->convection.number_of_perturbations-1; i++)
	fprintf(fp, "%d,", E->convection.perturb_mm[i]);
      fprintf(fp, "%d\n", E->convection.perturb_mm[E->convection.number_of_perturbations-1]);
      fprintf(fp, "perturblayer=");
      for(i=0; i<E->convection.number_of_perturbations-1; i++)
	fprintf(fp, "%d,", E->convection.load_depth[i]);
      fprintf(fp, "%d\n", E->convection.load_depth[E->convection.number_of_perturbations-1]);
      fprintf(fp, "perturbmag=");
      for(i=0; i<E->convection.number_of_perturbations-1; i++)
	fprintf(fp, "%g,", E->convection.perturb_mag[i]);
      fprintf(fp, "%g\n", E->convection.perturb_mag[E->convection.number_of_perturbations-1]);
    }
    else
    {
      fprintf(fp, "perturbl=%d\n", E->convection.perturb_ll[0]);
      fprintf(fp, "perturbm=%d\n", E->convection.perturb_mm[0]);
      fprintf(fp, "perturblayer=%d\n", E->convection.load_depth[0]);
      fprintf(fp, "perturbmag=%g\n", E->convection.perturb_mag[0]);
    }
    fprintf(fp, "half_space_age=%g\n", E->convection.half_space_age);
    fprintf(fp, "mantle_temp=%g\n", E->control.mantle_temp);
    fprintf(fp, "blob_center=[%g,%g,%g]\n", 
	    E->convection.blob_center[0],
	    E->convection.blob_center[1],
	    E->convection.blob_center[2]);
    fprintf(fp, "blob_radius=%g\n", E->convection.blob_radius);
    fprintf(fp, "blob_dT=%g\n", E->convection.blob_dT);
    fprintf(fp, "blob_bc_persist=%d\n", E->convection.blob_bc_persist);
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.output\n");
    fprintf(fp, "output_format=%s\n", E->output.format);
    fprintf(fp, "output_optional=%s\n", E->output.optional);
    fprintf(fp, "output_ll_max=%d\n", E->output.llmax);
    fprintf(fp, "self_gravitation=%d\n", E->control.self_gravitation);
    fprintf(fp, "use_cbf_topo=%d\n", E->control.use_cbf_topo);
    fprintf(fp, "cb_block_size=%d\n", E->output.cb_block_size);
    fprintf(fp, "cb_buffer_size=%d\n", E->output.cb_buffer_size);
    fprintf(fp, "sieve_buf_size=%d\n", E->output.sieve_buf_size);
    fprintf(fp, "output_alignment=%d\n", E->output.alignment);
    fprintf(fp, "output_alignment_threshold=%d\n", E->output.alignment_threshold);
    fprintf(fp, "cache_mdc_nelmts=%d\n", E->output.cache_mdc_nelmts);
    fprintf(fp, "cache_rdcc_nelmts=%d\n", E->output.cache_rdcc_nelmts);
    fprintf(fp, "cache_rdcc_nbytes=%d\n", E->output.cache_rdcc_nbytes);
    fprintf(fp, "write_q_files=%d\n", E->output.write_q_files);
    fprintf(fp, "vtk_format=%s\n", E->output.vtk_format);
    fprintf(fp, "gzdir_vtkio=%d\n", E->output.gzdir.vtk_io);
    fprintf(fp, "gzdir_rnr=%d\n", E->output.gzdir.rnr);
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.param\n");
    fprintf(fp, "reference_state=%d\n", E->refstate.choice);
    fprintf(fp, "refstate_file=%s\n", E->refstate.filename);
    fprintf(fp, "mineral_physics_model=%d\n", E->control.mineral_physics_model);
    fprintf(fp, "file_vbcs=%d\n", E->control.vbcs_file);
    fprintf(fp, "vel_bound_file=%s\n", E->control.velocity_boundary_file);
    fprintf(fp, "mat_control=%d\n", E->control.mat_control);
    fprintf(fp, "mat_file=%s\n", E->control.mat_file);
    fprintf(fp, "lith_age=%d\n", E->control.lith_age);
    fprintf(fp, "lith_age_file=%s\n", E->control.lith_age_file);
    fprintf(fp, "lith_age_time=%d\n", E->control.lith_age_time);
    fprintf(fp, "lith_age_depth=%g\n", E->control.lith_age_depth);
    fprintf(fp, "start_age=%g\n", E->control.start_age);
    fprintf(fp, "reset_startage=%d\n", E->control.reset_startage);
    fprintf(fp, "file_tbcs=%d\n", E->control.tbcs_file);
    fprintf(fp, "temp_bound_file=%s\n", E->control.temperature_boundary_file);
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.phase\n");
    fprintf(fp, "Ra_410=%g\n", E->control.Ra_410);
    fprintf(fp, "clapeyron410=%g\n", E->control.clapeyron410);
    fprintf(fp, "transT410=%g\n", E->control.transT410);
    tmp = 0.0f;
    if(E->control.inv_width410 != 0.0)
        tmp = 1.0/E->control.inv_width410;
    fprintf(fp, "width410=%g\n", tmp);
    fprintf(fp, "Ra_670=%g\n", E->control.Ra_670);
    fprintf(fp, "clapeyron670=%g\n", E->control.clapeyron670);
    fprintf(fp, "transT670=%g\n", E->control.transT670);
    tmp = 0.0f;
    if(E->control.inv_width670 != 0.0)
        tmp = 1.0/E->control.inv_width670;
    fprintf(fp, "width670=%g\n", tmp);
    fprintf(fp, "Ra_cmb=%g\n", E->control.Ra_cmb);
    fprintf(fp, "clapeyroncmb=%g\n", E->control.clapeyroncmb);
    fprintf(fp, "transTcmb=%g\n", E->control.transTcmb);
    tmp = 0.0f;
    if(E->control.inv_widthcmb != 0.0)
        tmp = 1.0/E->control.inv_widthcmb;
    fprintf(fp, "widthcmb=%g\n", tmp);
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.tracer\n");
    fprintf(fp, "tracer=%d\n", E->control.tracer);
    fprintf(fp, "tracer_ic_method=%d\n", E->trace.ic_method);
    fprintf(fp, "tracers_per_element=%d\n", E->trace.itperel);
    fprintf(fp, "tracer_file=%s\n", E->trace.tracer_file);
    fprintf(fp, "tracer_flavors=%d\n", E->trace.nflavors);
    fprintf(fp, "ic_method_for_flavors=%d\n", E->trace.ic_method_for_flavors);
    fprintf(fp, "z_interface=");
    if(E->trace.nflavors > 0)
    {
      for(i=0; i<E->trace.nflavors-1;i++)
        fprintf(fp, "%g,", E->trace.z_interface[i]);
      fprintf(fp, "%g\n", E->trace.z_interface[E->trace.nflavors-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "itracer_warnings=%d\n", E->trace.itracer_warnings);
    fprintf(fp, "regular_grid_deltheta=%g\n", E->trace.deltheta[0]);
    fprintf(fp, "regular_grid_delphi=%g\n", E->trace.delphi[0]);
    fprintf(fp, "chemical_buoyancy=%d\n", E->composition.ichemical_buoyancy);
    fprintf(fp, "buoy_type=%d\n", E->composition.ibuoy_type);
    fprintf(fp, "buoyancy_ratio=");
    if(E->composition.ncomp > 0)
    {
      for(i=0; i<E->composition.ncomp-1;i++)
        fprintf(fp, "%g,", E->composition.buoyancy_ratio[i]);
      fprintf(fp, "%g\n", E->composition.buoyancy_ratio[E->composition.ncomp-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "tracer_enriched=%d\n", E->control.tracer_enriched);
    fprintf(fp, "Q0_enriched=%g\n", E->control.Q0ER);
    fprintf(fp, "\n\n");

    fprintf(fp, "# CitcomS.solver.visc\n");
    fprintf(fp, "Viscosity=%s\n", E->viscosity.STRUCTURE);
    fprintf(fp, "visc_smooth_method=%d\n", E->viscosity.smooth_cycles);
    fprintf(fp, "VISC_UPDATE=%d\n", E->viscosity.update_allowed);
    fprintf(fp, "num_mat=%d\n", E->viscosity.num_mat);
    fprintf(fp, "visc0=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", E->viscosity.N0[i]);
      fprintf(fp, "%g\n", E->viscosity.N0[E->viscosity.num_mat-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "TDEPV=%d\n", E->viscosity.TDEPV);
    fprintf(fp, "rheol=%d\n", E->viscosity.RHEOL);
    fprintf(fp, "viscE=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", E->viscosity.E[i]);
      fprintf(fp, "%g\n", E->viscosity.E[E->viscosity.num_mat-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "viscT=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", E->viscosity.T[i]);
      fprintf(fp, "%g\n", E->viscosity.T[E->viscosity.num_mat-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "viscZ=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", E->viscosity.Z[i]);
      fprintf(fp, "%g\n", E->viscosity.Z[E->viscosity.num_mat-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "SDEPV=%d\n", E->viscosity.SDEPV);
    fprintf(fp, "sdepv_expt=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", (E->viscosity.SDEPV == 0 ? 
			    1.0 : E->viscosity.sdepv_expt[i]));
      fprintf(fp, "%g\n", (E->viscosity.SDEPV == 0 ?
			   1.0 : E->viscosity.sdepv_expt[E->viscosity.num_mat-1]));
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "sdepv_misfit=%g\n", E->viscosity.sdepv_misfit);
    fprintf(fp, "PDEPV=%d\n", E->viscosity.PDEPV);
    fprintf(fp, "pdepv_a=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", (E->viscosity.PDEPV == 0 ?
			    1e20 : E->viscosity.pdepv_a[i]));
      fprintf(fp, "%g\n", (E->viscosity.PDEPV == 0 ?
			   1e20 : E->viscosity.pdepv_a[E->viscosity.num_mat-1]));
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "pdepv_b=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", (E->viscosity.PDEPV == 0 ?
			    0.0 : E->viscosity.pdepv_b[i]));
      fprintf(fp, "%g\n", (E->viscosity.PDEPV == 0 ?
			   0.0 : E->viscosity.pdepv_b[E->viscosity.num_mat-1]));
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "pdepv_y=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", (E->viscosity.PDEPV == 0 ?
			    1e20 : E->viscosity.pdepv_y[i]));
      fprintf(fp, "%g\n", (E->viscosity.PDEPV == 0 ?
			   1e20 : E->viscosity.pdepv_y[E->viscosity.num_mat-1]));
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "pdepv_eff=%d\n", E->viscosity.pdepv_eff);
    fprintf(fp, "pdepv_offset=%g\n", E->viscosity.pdepv_offset);
    fprintf(fp, "CDEPV=%d\n", E->viscosity.CDEPV);
    fprintf(fp, "cdepv_ff=");
    if(E->trace.nflavors > 0)
    {
      for(i=0; i<E->trace.nflavors-1;i++)
        fprintf(fp, "%g,", E->viscosity.cdepv_ff[i]);
      fprintf(fp, "%g\n", E->viscosity.cdepv_ff[E->trace.nflavors-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "low_visc_channel=%d\n", E->viscosity.channel);
    fprintf(fp, "low_visc_wedge=%d\n", E->viscosity.wedge);
    fprintf(fp, "lv_min_radius=%g\n", E->viscosity.lv_min_radius);
    fprintf(fp, "lv_max_radius=%g\n", E->viscosity.lv_max_radius);
    fprintf(fp, "lv_channel_thickness=%g\n", E->viscosity.lv_channel_thickness);
    fprintf(fp, "lv_reduction=%g\n", E->viscosity.lv_reduction);
    fprintf(fp, "VMIN=%d\n", E->viscosity.MIN);
    fprintf(fp, "visc_min=%g\n", E->viscosity.min_value);
    fprintf(fp, "VMAX=%d\n", E->viscosity.MAX);
    fprintf(fp, "visc_max=%g\n", E->viscosity.max_value);
    fprintf(fp, "z_layer=");
    if(E->viscosity.num_mat > 0)
    {
      for(i=0; i<E->viscosity.num_mat-1;i++)
        fprintf(fp, "%g,", E->viscosity.zbase_layer[i]);
      fprintf(fp, "%g\n", E->viscosity.zbase_layer[E->viscosity.num_mat-1]);
    }
    else
    {
      fprintf(fp, "\n");
    }
    fprintf(fp, "visc_layer_control=%d\n", E->viscosity.layer_control);
    fprintf(fp, "visc_layer_file=%s\n", E->viscosity.layer_file);

    /* close the file after we are done writing all parameters */
    fclose(fp);
  }
}
