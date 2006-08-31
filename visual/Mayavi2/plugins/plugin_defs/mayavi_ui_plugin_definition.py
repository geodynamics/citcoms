"""The MayaVi plugin definition.

"""
# Author: Prabhu Ramachandran <prabhu_r@users.sf.net>
# Copyright (c) 2005, Enthought, Inc.
# License: BSD Style.

# Enthought library imports.
from enthought.envisage import PluginDefinition, get_using_workbench

# Are we using the old UI plugin, or the shiny new Workbench plugin?
USING_WORKBENCH = get_using_workbench()

if USING_WORKBENCH:
    from enthought.envisage.workbench.action.action_plugin_definition import \
         Action, Group, Location, Menu, WorkbenchActionSet
    from enthought.envisage.workbench.workbench_plugin_definition import \
         View, Workbench
else:
    from enthought.envisage.ui.ui_plugin_definition \
         import Action, Group, Menu, UIActions, UIViews, View

# The plugin's globally unique identifier should really be
# enthought.mayavi_ui.  However this will break the module names that
# we use to define the actions.  We therefore fix the ID in the plugin
# definition.  This ID is used as the prefix for all identifiers
# defined in this module.
ID = "enthought.mayavi"

######################################################################
# Actions.

if USING_WORKBENCH:
    groups = [Group(id="M2FileGroup",
                    location=Location(path="MenuBar/FileMenu",
                                      before="TVTKFileGroup")),
              Group(id="VisualizeMenuGroup",
                    location=Location(path="MenuBar", before="ViewMenuGroup")),
              ]
    ########################################
    # Menus
    open_menu = Menu(
        id     = "OpenMenu",
        name   = "&Open",
        location   = Location(path="MenuBar/FileMenu/M2FileGroup"),
        groups = [Group(id = "M2OpenGroup"),
                  ]
    )

    visualize_menu = Menu(
        id     = "VisualizeMenu",
        name   = "Visuali&ze",
        location   = Location(path="MenuBar/VisualizeMenuGroup"),
        groups = [Group(id = "M2VizGroup"),
                  ]
    )

    modules_menu = Menu(
        id     = "ModulesMenu",
        name   = "&Modules",
        location  = Location(path="MenuBar/VisualizeMenu/M2VizGroup"),
        groups = [Group(id = "M2ModulesGroup"),
                  ]
    )

    filters_menu = Menu(
        id     = "FiltersMenu",
        name   = "&Filters",
        location   = Location(path="MenuBar/VisualizeMenu/M2VizGroup"),
        groups = [Group(id = "M2FiltersGroup"),
                  ]
    )

    ########################################
    # File menu items.
    open_vtk = Action(
        id            = ID + ".action.sources.OpenVTKFile",
        class_name    = ID + ".action.sources.OpenVTKFile",
        name          = "&VTK file",
        #image         = "images/new_scene.png",
        tooltip       = "Open a VTK data file",
        description   = "Open a VTK data file",
        locations     = [Location(path="MenuBar/FileMenu/OpenMenu/M2OpenGroup"),]
    )

    open_vtk_xml = Action(
        id            = ID + ".action.sources.OpenVTKXMLFile",
        class_name    = ID + ".action.sources.OpenVTKXMLFile",
        name          = "VTK &XML file",
        #image         = "images/new_scene.png",
        tooltip       = "Open a VTK XML data file",
        description   = "Open a VTK XML data file",
        locations     = [Location(path="MenuBar/FileMenu/OpenMenu/M2OpenGroup"),]
    )

    open_citcoms_vtk = Action(
        id            = ID + ".plugins.OpenCitcomSFILES.OpenCitcomSVTKFILE",
        class_name    = ID + ".plugins.OpenCitcomSFILES.OpenCitcomSVTKFILE",
        name          = "&Citcoms VTK file",
        #image         = "images/new_scene.png",
        tooltip       = "Open a CitcomS VTK data file",
        description   = "Open a CitcomS VTK data file",
        locations     = [Location(path="MenuBar/FileMenu/OpenMenu/M2OpenGroup"),]
    )

    open_citcoms_hdf = Action(
        id            = ID + ".plugins.OpenCitcomSFILES.OpenCitcomSHDFFILE",
        class_name    = ID + ".plugins.OpenCitcomSFILES.OpenCitcomSHDFFILE",
        name          = "Citcoms &HDF file",
        #image         = "images/new_scene.png",
        tooltip       = "Open a CitcomS HDF data file",
        description   = "Open a CitcomS HDF data file",
        locations     = [Location(path="MenuBar/FileMenu/OpenMenu/M2OpenGroup"),]
    )
    save_viz = Action(
        id            = ID + ".action.save_load.SaveVisualization",
        class_name    = ID + ".action.save_load.SaveVisualization",
        name          = "&Save Visualization",
        #image         = "images/new_scene.png",
        tooltip       = "Save current visualization",
        description   = "Save current visualization to a MayaVi2 file",
        locations     = [Location(path="MenuBar/FileMenu/M2FileGroup",
                                  after="OpenMenu"),]
    )

    load_viz = Action(
        id            = ID + ".action.save_load.LoadVisualization",
        class_name    = ID + ".action.save_load.LoadVisualization",
        name          = "&Load Visualization",
        #image         = "images/new_scene.png",
        tooltip       = "Load saved visualization",
        description   = "Load saved visualization from a MayaVi2 file",
        locations     = [Location(path="MenuBar/FileMenu/M2FileGroup",
                                  after="OpenMenu"),]
    )

    ########################################
    # Visualize/Module menu items.
    axes_module = Action(
        id            = ID + ".action.modules.AxesModule",
        class_name    = ID + ".action.modules.AxesModule",
        name          = "&Axes",
        #image         = "images/new_scene.png",
        tooltip       = "Draw axes on the outline of input data",
        description   = "Draw cubical axes on the outline for given input",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    contour_grid_plane_module = Action(
        id            = ID + ".action.modules.ContourGridPlaneModule",
        class_name    = ID + ".action.modules.ContourGridPlaneModule",
        name          = "&ContourGridPlane",
        #image         = "images/new_scene.png",
        tooltip       = "Shows a contour grid plane for the given input",
        description   = "Shows a contour grid plane for the given input",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    glyph_module = Action(
        id            = ID + ".action.modules.GlyphModule",
        class_name    = ID + ".action.modules.GlyphModule",
        name          = "Gl&yph",
        #image         = "images/new_scene.png",
        tooltip       = "Creates colored and scaled glyphs at at input points",
        description   = "Creates colored and scaled glyphs at at input points",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    grid_plane_module = Action(
        id            = ID + ".action.modules.GridPlaneModule",
        class_name    = ID + ".action.modules.GridPlaneModule",
        name          = "&GridPlane",
        #image         = "images/new_scene.png",
        tooltip       = "Shows a grid plane for the given input",
        description   = "Shows a grid plane for the given input",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    image_plane_widget_module = Action(
        id            = ID + ".action.modules.ImagePlaneWidgetModule",
        class_name    = ID + ".action.modules.ImagePlaneWidgetModule",
        name          = "I&magePlaneWidget",
        #image         = "images/new_scene.png",
        tooltip       = "Shows an image plane widget for image data",
        description   = "Shows an image plane widget for image data",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    isosurface_module = Action(
        id            = ID + ".action.modules.IsoSurfaceModule",
        class_name    = ID + ".action.modules.IsoSurfaceModule",
        name          = "&IsoSurface",
        #image         = "images/new_scene.png",
        tooltip       = "Creates an iso-surface for the given input",
        description   = "Creates an iso-surface for the given input",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    orientation_axes_module = Action(
        id            = ID + ".action.modules.OrientationAxesModule",
        class_name    = ID + ".action.modules.OrientationAxesModule",
        name          = "Orientation A&xes",
        #image         = "images/new_scene.png",
        tooltip       = "Show an axes indicating the current orientation",
        description   = "Show an axes indicating the current orientation",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    outline_module = Action(
        id            = ID + ".action.modules.OutlineModule",
        class_name    = ID + ".action.modules.OutlineModule",
        name          = "&Outline",
        #image         = "images/new_scene.png",
        tooltip       = "Draw an outline for given input",
        description   = "Draw an outline for given input",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    scalar_cut_plane_module = Action(
        id            = ID + ".action.modules.ScalarCutPlaneModule",
        class_name    = ID + ".action.modules.ScalarCutPlaneModule",
        name          = "Scalar Cut &Plane",
        #image         = "images/new_scene.png",
        tooltip       = "Slice through the data with optional contours",
        description   = "Slice through the data with optional contours",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    slice_ug_module = Action(
        id            = ID + ".action.modules.SliceUnstructuredGridModule",
        class_name    = ID + ".action.modules.SliceUnstructuredGridModule",
        name          = "S&lice Unstructured Grid",
        #image         = "images/new_scene.png",
        tooltip       = "Slice an unstructured grid to show cells",
        description   = "Slice an unstructured grid to show cells",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    streamline_module = Action(
        id            = ID + ".action.modules.StreamlineModule",
        class_name    = ID + ".action.modules.StreamlineModule",
        name          = "Stream&line",
        #image         = "images/new_scene.png",
        tooltip       = "Generate streamlines for the vectors",
        description   = "Generate streamlines for the vectors",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    surface_module = Action(
        id            = ID + ".action.modules.SurfaceModule",
        class_name    = ID + ".action.modules.SurfaceModule",
        name          = "&Surface",
        #image         = "images/new_scene.png",
        tooltip       = "Creates a surface for the given input",
        description   = "Creates a surface for the given input",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    text_module = Action(
        id            = ID + ".action.modules.TextModule",
        class_name    = ID + ".action.modules.TextModule",
        name          = "&Text",
        #image         = "images/new_scene.png",
        tooltip       = "Displays text on screen",
        description   = "Displays user specified text on screen",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    vector_cut_plane_module = Action(
        id            = ID + ".action.modules.VectorCutPlaneModule",
        class_name    = ID + ".action.modules.VectorCutPlaneModule",
        name          = "&VectorCutPlane",
        #image         = "images/new_scene.png",
        tooltip       = "Display vectors along a cut plane",
        description   = "Display vectors along a cut plane",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    vectors_module = Action(
        id            = ID + ".action.modules.VectorsModule",
        class_name    = ID + ".action.modules.VectorsModule",
        name          = "V&ectors",
        #image         = "images/new_scene.png",
        tooltip       = "Display input vectors using arrows or other glyphs",
        description   = "Display input vectors using arrows or other glyphs",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )

    ########################################
    # Visualize/Filter menu items.
    cell_to_point_data_filter = Action(
        id            = ID + ".action.filters.CellToPointDataFilter",
        class_name    = ID + ".action.filters.CellToPointDataFilter",
        name          = "&CellToPointData",
        #image         = "images/new_scene.png",
        tooltip       = "Convert cell data to point data for the active data",
        description   = "Convert cell data to point data for the active data",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    delaunay2d_filter = Action(
        id            = ID + ".action.filters.Delaunay2DFilter",
        class_name    = ID + ".action.filters.Delaunay2DFilter",
        name          = "&Delaunay2D",
        #image         = "images/new_scene.png",
        tooltip       = "Perform a 2D Delaunay triangulation for the given data",
        description   = "Perform a 2D Delaunay triangulation for the given data",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    delaunay3d_filter = Action(
        id            = ID + ".action.filters.Delaunay3DFilter",
        class_name    = ID + ".action.filters.Delaunay3DFilter",
        name          = "Delaunay&3D",
        #image         = "images/new_scene.png",
        tooltip       = "Perform a 3D Delaunay triangulation for the given data",
        description   = "Perform a 3D Delaunay triangulation for the given data",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    extract_unstructured_grid_filter = Action(
        id            = ID + ".action.filters.ExtractUnstructuredGridFilter",
        class_name    = ID + ".action.filters.ExtractUnstructuredGridFilter",
        name          = "Extract &Unstructured Grid",
        #image         = "images/new_scene.png",
        tooltip       = "Extract part of an unstructured grid",
        description   = "Extract part of an unstructured grid",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    extract_vector_norm_filter = Action(
        id            = ID + ".action.filters.ExtractVectorNormFilter",
        class_name    = ID + ".action.filters.ExtractVectorNormFilter",
        name          = "Extract &Vector Norm",
        #image         = "images/new_scene.png",
        tooltip       = "Compute the vector norm for the current vector data",
        description   = "Compute the vector norm for the current vector data",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    mask_points_filter = Action(
        id            = ID + ".action.filters.MaskPointsFilter",
        class_name    = ID + ".action.filters.MaskPointsFilter",
        name          = "&Mask Points",
        #image         = "images/new_scene.png",
        tooltip       = "Mask the input points in the data",
        description   = "Mask the input points in the data",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    poly_data_normals_filter = Action(
        id            = ID + ".action.filters.PolyDataNormalsFilter",
        class_name    = ID + ".action.filters.PolyDataNormalsFilter",
        name          = "Compute &Normals",
        #image         = "images/new_scene.png",
        tooltip       = "Compute normals and smooth the appearance",
        description   = "Compute normals and smooth the appearance",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    threshold_filter = Action(
        id            = ID + ".action.filters.ThresholdFilter",
        class_name    = ID + ".action.filters.ThresholdFilter",
        name          = "&Threshold",
        #image         = "images/new_scene.png",
        tooltip       = "Threshold input data based on scalar values",
        description   = "Threshold input data based on scalar values",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    warp_scalar_filter = Action(
        id            = ID + ".action.filters.WarpScalarFilter",
        class_name    = ID + ".action.filters.WarpScalarFilter",
        name          = "Warp S&calar",
        #image         = "images/new_scene.png",
        tooltip       = "Move points of data along normals by the scalar data",
        description   = "Move points of data along normals by the scalar data",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    warp_vector_filter = Action(
        id            = ID + ".action.filters.WarpVectorFilter",
        class_name    = ID + ".action.filters.WarpVectorFilter",
        name          = "Warp &Vector",
        #image         = "images/new_scene.png",
        tooltip       = "Move points of data along the vector data at point",
        description   = "Move points of data along the vector data at point",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    warp_vector_filter = Action(
        id            = ID + ".action.filters.WarpVectorFilter",
        class_name    = ID + ".action.filters.WarpVectorFilter",
        name          = "Warp &Vector",
        #image         = "images/new_scene.png",
        tooltip       = "Move points of data along the vector data at point",
        description   = "Move points of data along the vector data at point",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    citcoms_cap_filter = Action(
        id            = ID + ".plugins.filter.CitcomSFilterActions.CitcomSshowCaps",
        class_name    = ID + ".plugins.filter.CitcomSFilterActions.CitcomSshowCaps",
        name          = "CitcomS &ShowCaps",
        #image         = "images/new_scene.png",
        tooltip       = "Dispaly a specified range of caps",
        description   = "Dispaly a specified range of caps",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    citcoms_reduce_filter = Action(
        id            = ID + ".plugins.filter.CitcomSFilterActions.CitcomSreduce",
        class_name    = ID + ".plugins.filter.CitcomSFilterActions.CitcomSreduce",
        name          = "CitcomS &Reduce",
        #image         = "images/new_scene.png",
        tooltip       = "Dispaly a Reduce Grid for interpolation",
        description   = "Dispaly a Reduce Grid for interpolation",
        locations = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/M2FiltersGroup"),]
    )

    ########################################
    # List of all actions.
    action_set = WorkbenchActionSet(
        id = ID + '.action_set',
        name = 'Mayavi2ActionSet',
        groups = groups,
        menus = [open_menu,
                 visualize_menu,
                 modules_menu,
                 filters_menu
                 ],

        actions = [open_vtk,  # File menu
                   open_vtk_xml,
                   open_citcoms_vtk,
                   open_citcoms_hdf,
                   save_viz,
                   load_viz,
                   # Modules.
                   axes_module,
                   contour_grid_plane_module,
                   glyph_module,
                   grid_plane_module,
                   image_plane_widget_module,
                   isosurface_module,
                   orientation_axes_module,
                   outline_module,
                   scalar_cut_plane_module,
                   slice_ug_module,
                   streamline_module,
                   surface_module,
                   text_module,
                   vector_cut_plane_module,
                   vectors_module,
                   # Filters.
                   cell_to_point_data_filter,
                   delaunay2d_filter,
                   delaunay3d_filter,
                   extract_unstructured_grid_filter,
                   extract_vector_norm_filter,               
                   mask_points_filter,
                   poly_data_normals_filter,
                   threshold_filter,
                   warp_scalar_filter,
                   citcoms_cap_filter,
                   citcoms_reduce_filter,
                   ]
    )

    ######################################################################
    # Views.
    views = [View(name="MayaVi",
                  #image="images/project_view.png",
                  id=ID + ".view.engine_view.EngineView",
                  class_name=ID + ".view.engine_view.EngineView",
                  position="left",
                  )
             ]
    workbench = Workbench(views=views)

    requires = ["enthought.envisage.workbench",
                "enthought.plugins.python_shell",
                "enthought.tvtk.plugins.scene",
                 ]
    extensions = [action_set, workbench ]
else:
    ########################################
    # Code for older UI plugin.
    
    ########################################
    # Menus
    open_menu = Menu(
        id     = "OpenMenu",
        name   = "&Open",
        path   = "FileMenu/Start",
        groups = [
            Group(id = "Start"),
            Group(id = "End"),
        ]
    )

    visualize_menu = Menu(
        id     = "VisualizeMenu",
        name   = "Visuali&ze",
        path   = "ToolsGroup",
        groups = [Group(id = "Start"),
                  Group(id = "End"),
                  ]
    )

    modules_menu = Menu(
        id     = "ModulesMenu",
        name   = "&Modules",
        path   = "VisualizeMenu/Start",
        groups = [Group(id = "Start"),
                  Group(id = "End"),
                  ]
    )

    filters_menu = Menu(
        id     = "FiltersMenu",
        name   = "&Filters",
        path   = "VisualizeMenu/Start",
        groups = [Group(id = "Start"),
                  Group(id = "End"),
                  ]
    )

    ########################################
    # File menu items.
    open_vtk = Action(
        id            = ID + ".action.sources.OpenVTKFile",
        class_name    = ID + ".action.sources.OpenVTKFile",
        name          = "&VTK file",
        #image         = "images/new_scene.png",
        tooltip       = "Open a VTK data file",
        description   = "Open a VTK data file",
        menu_bar_path = "FileMenu/OpenMenu/Start"
    )

    open_vtk_xml = Action(
        id            = ID + ".action.sources.OpenVTKXMLFile",
        class_name    = ID + ".action.sources.OpenVTKXMLFile",
        name          = "VTK &XML file",
        #image         = "images/new_scene.png",
        tooltip       = "Open a VTK XML data file",
        description   = "Open a VTK XML data file",
        menu_bar_path = "FileMenu/OpenMenu/Start"
    )

    save_viz = Action(
        id            = ID + ".action.save_load.SaveVisualization",
        class_name    = ID + ".action.save_load.SaveVisualization",
        name          = "&Save Visualization",
        #image         = "images/new_scene.png",
        tooltip       = "Save current visualization",
        description   = "Save current visualization to a MayaVi2 file",
        menu_bar_path = "FileMenu/Start"
    )

    load_viz = Action(
        id            = ID + ".action.save_load.LoadVisualization",
        class_name    = ID + ".action.save_load.LoadVisualization",
        name          = "&Load Visualization",
        #image         = "images/new_scene.png",
        tooltip       = "Load saved visualization",
        description   = "Load saved visualization from a MayaVi2 file",
        menu_bar_path = "FileMenu/Start"
    )

    ########################################
    # Visualize/Module menu items.
    axes_module = Action(
        id            = ID + ".action.modules.AxesModule",
        class_name    = ID + ".action.modules.AxesModule",
        name          = "&Axes",
        #image         = "images/new_scene.png",
        tooltip       = "Draw axes on the outline of input data",
        description   = "Draw cubical axes on the outline for given input",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    contour_grid_plane_module = Action(
        id            = ID + ".action.modules.ContourGridPlaneModule",
        class_name    = ID + ".action.modules.ContourGridPlaneModule",
        name          = "&ContourGridPlane",
        #image         = "images/new_scene.png",
        tooltip       = "Shows a contour grid plane for the given input",
        description   = "Shows a contour grid plane for the given input",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    glyph_module = Action(
        id            = ID + ".action.modules.GlyphModule",
        class_name    = ID + ".action.modules.GlyphModule",
        name          = "Gl&yph",
        #image         = "images/new_scene.png",
        tooltip       = "Creates colored and scaled glyphs at at input points",
        description   = "Creates colored and scaled glyphs at at input points",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    grid_plane_module = Action(
        id            = ID + ".action.modules.GridPlaneModule",
        class_name    = ID + ".action.modules.GridPlaneModule",
        name          = "&GridPlane",
        #image         = "images/new_scene.png",
        tooltip       = "Shows a grid plane for the given input",
        description   = "Shows a grid plane for the given input",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    image_plane_widget_module = Action(
        id            = ID + ".action.modules.ImagePlaneWidgetModule",
        class_name    = ID + ".action.modules.ImagePlaneWidgetModule",
        name          = "I&magePlaneWidget",
        #image         = "images/new_scene.png",
        tooltip       = "Shows an image plane widget for image data",
        description   = "Shows an image plane widget for image data",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    isosurface_module = Action(
        id            = ID + ".action.modules.IsoSurfaceModule",
        class_name    = ID + ".action.modules.IsoSurfaceModule",
        name          = "&IsoSurface",
        #image         = "images/new_scene.png",
        tooltip       = "Creates an iso-surface for the given input",
        description   = "Creates an iso-surface for the given input",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    orientation_axes_module = Action(
        id            = ID + ".action.modules.OrientationAxesModule",
        class_name    = ID + ".action.modules.OrientationAxesModule",
        name          = "Orientation A&xes",
        #image         = "images/new_scene.png",
        tooltip       = "Show an axes indicating the current orientation",
        description   = "Show an axes indicating the current orientation",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    outline_module = Action(
        id            = ID + ".action.modules.OutlineModule",
        class_name    = ID + ".action.modules.OutlineModule",
        name          = "&Outline",
        #image         = "images/new_scene.png",
        tooltip       = "Draw an outline for given input",
        description   = "Draw an outline for given input",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    scalar_cut_plane_module = Action(
        id            = ID + ".action.modules.ScalarCutPlaneModule",
        class_name    = ID + ".action.modules.ScalarCutPlaneModule",
        name          = "Scalar Cut &Plane",
        #image         = "images/new_scene.png",
        tooltip       = "Slice through the data with optional contours",
        description   = "Slice through the data with optional contours",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    slice_ug_module = Action(
        id            = ID + ".action.modules.SliceUnstructuredGridModule",
        class_name    = ID + ".action.modules.SliceUnstructuredGridModule",
        name          = "S&lice Unstructured Grid",
        #image         = "images/new_scene.png",
        tooltip       = "Slice an unstructured grid to show cells",
        description   = "Slice an unstructured grid to show cells",
        locations = [Location(path="MenuBar/VisualizeMenu/ModulesMenu/M2ModulesGroup"),]
    )
    
    streamline_module = Action(
        id            = ID + ".action.modules.StreamlineModule",
        class_name    = ID + ".action.modules.StreamlineModule",
        name          = "Stream&line",
        #image         = "images/new_scene.png",
        tooltip       = "Generate streamlines for the vectors",
        description   = "Generate streamlines for the vectors",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    surface_module = Action(
        id            = ID + ".action.modules.SurfaceModule",
        class_name    = ID + ".action.modules.SurfaceModule",
        name          = "&Surface",
        #image         = "images/new_scene.png",
        tooltip       = "Creates a surface for the given input",
        description   = "Creates a surface for the given input",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    text_module = Action(
        id            = ID + ".action.modules.TextModule",
        class_name    = ID + ".action.modules.TextModule",
        name          = "&Text",
        #image         = "images/new_scene.png",
        tooltip       = "Displays text on screen",
        description   = "Displays user specified text on screen",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    vector_cut_plane_module = Action(
        id            = ID + ".action.modules.VectorCutPlaneModule",
        class_name    = ID + ".action.modules.VectorCutPlaneModule",
        name          = "&VectorCutPlane",
        #image         = "images/new_scene.png",
        tooltip       = "Display vectors along a cut plane",
        description   = "Display vectors along a cut plane",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    vectors_module = Action(
        id            = ID + ".action.modules.VectorsModule",
        class_name    = ID + ".action.modules.VectorsModule",
        name          = "V&ectors",
        #image         = "images/new_scene.png",
        tooltip       = "Display input vectors using arrows or other glyphs",
        description   = "Display input vectors using arrows or other glyphs",
        menu_bar_path = "VisualizeMenu/ModulesMenu/Start"
    )

    ########################################
    # Visualize/Filter menu items.
    cell_to_point_data_filter = Action(
        id            = ID + ".action.filters.CellToPointDataFilter",
        class_name    = ID + ".action.filters.CellToPointDataFilter",
        name          = "&CellToPointData",
        #image         = "images/new_scene.png",
        tooltip       = "Convert cell data to point data for the active data",
        description   = "Convert cell data to point data for the active data",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    delaunay2d_filter = Action(
        id            = ID + ".action.filters.Delaunay2DFilter",
        class_name    = ID + ".action.filters.Delaunay2DFilter",
        name          = "&Delaunay2D",
        #image         = "images/new_scene.png",
        tooltip       = "Perform a 2D Delaunay triangulation for the given data",
        description   = "Perform a 2D Delaunay triangulation for the given data",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    delaunay3d_filter = Action(
        id            = ID + ".action.filters.Delaunay3DFilter",
        class_name    = ID + ".action.filters.Delaunay3DFilter",
        name          = "Delaunay&3D",
        #image         = "images/new_scene.png",
        tooltip       = "Perform a 3D Delaunay triangulation for the given data",
        description   = "Perform a 3D Delaunay triangulation for the given data",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    extract_unstructured_grid_filter = Action(
        id            = ID + ".action.filters.ExtractUnstructuredGridFilter",
        class_name    = ID + ".action.filters.ExtractUnstructuredGridFilter",
        name          = "Extract &Unstructured Grid",
        #image         = "images/new_scene.png",
        tooltip       = "Extract part of an unstructured grid",
        description   = "Extract part of an unstructured grid",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    extract_vector_norm_filter = Action(
        id            = ID + ".action.filters.ExtractVectorNormFilter",
        class_name    = ID + ".action.filters.ExtractVectorNormFilter",
        name          = "Extract &Vector Norm",
        #image         = "images/new_scene.png",
        tooltip       = "Compute the vector norm for the current vector data",
        description   = "Compute the vector norm for the current vector data",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    mask_points_filter = Action(
        id            = ID + ".action.filters.MaskPointsFilter",
        class_name    = ID + ".action.filters.MaskPointsFilter",
        name          = "&Mask Points",
        #image         = "images/new_scene.png",
        tooltip       = "Mask the input points in the data",
        description   = "Mask the input points in the data",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    poly_data_normals_filter = Action(
        id            = ID + ".action.filters.PolyDataNormalsFilter",
        class_name    = ID + ".action.filters.PolyDataNormalsFilter",
        name          = "Compute &Normals",
        #image         = "images/new_scene.png",
        tooltip       = "Compute normals and smooth the appearance",
        description   = "Compute normals and smooth the appearance",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    threshold_filter = Action(
        id            = ID + ".action.filters.ThresholdFilter",
        class_name    = ID + ".action.filters.ThresholdFilter",
        name          = "&Threshold",
        #image         = "images/new_scene.png",
        tooltip       = "Threshold input data based on scalar values",
        description   = "Threshold input data based on scalar values",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    warp_scalar_filter = Action(
        id            = ID + ".action.filters.WarpScalarFilter",
        class_name    = ID + ".action.filters.WarpScalarFilter",
        name          = "Warp S&calar",
        #image         = "images/new_scene.png",
        tooltip       = "Move points of data along normals by the scalar data",
        description   = "Move points of data along normals by the scalar data",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    warp_vector_filter = Action(
        id            = ID + ".action.filters.WarpVectorFilter",
        class_name    = ID + ".action.filters.WarpVectorFilter",
        name          = "Warp &Vector",
        #image         = "images/new_scene.png",
        tooltip       = "Move points of data along the vector data at point",
        description   = "Move points of data along the vector data at point",
        menu_bar_path = "VisualizeMenu/FiltersMenu/Start"
    )

    ########################################
    # List of all actions.
    ui_actions = UIActions(
        menus = [open_menu,
                 visualize_menu,
                 modules_menu,
                 filters_menu
                 ],

        actions = [open_vtk,  # File menu
                   open_vtk_xml,
                   save_viz,
                   load_viz,
                   # Modules.
                   axes_module,
                   contour_grid_plane_module,
                   glyph_module,
                   grid_plane_module,
                   image_plane_widget_module,
                   isosurface_module,
                   orientation_axes_module,
                   outline_module,
                   scalar_cut_plane_module,
                   slice_ug_module,
                   streamline_module,
                   surface_module,
                   text_module,
                   vector_cut_plane_module,
                   vectors_module,
                   # Filters.
                   cell_to_point_data_filter,
                   delaunay2d_filter,
                   delaunay3d_filter,
                   extract_unstructured_grid_filter,
                   extract_vector_norm_filter,               
                   mask_points_filter,
                   poly_data_normals_filter,
                   threshold_filter,
                   warp_scalar_filter,
                   warp_vector_filter,
                   ]
    )

    ######################################################################
    # Views.
    views = [View(name="MayaVi",
                  #image="images/project_view.png",
                  id=ID + ".view.engine_view.EngineView",
                  class_name=ID + ".view.engine_view.EngineView"
                  )
             ]
    ui_views = UIViews(views=views)

    requires = ["enthought.envisage.ui",
                "enthought.envisage.ui.preference",
                "enthought.envisage.ui.python_shell",
                "enthought.tvtk.plugins.scene",
                 ]
    extensions = [ui_views,  ui_actions, ]


PluginDefinition(
    # This plugins unique identifier.
    id = ID + '_ui',

    # General info.
    name = "The MayaVi UI Plugin",
    version = "2.0",
    provider_name = "Prabhu Ramachandran",
    provider_url = "www.enthought.com",
    enabled = True,
    autostart = True,

    # Id's of plugin that this one requires.
    requires = requires + ['enthought.mayavi'],
    
    # The extension points that we provide.
    extension_points = [],

    # The contributions that this plugin makes to extension points offered by
    # either itself or other plugins.
    extensions = extensions
)
