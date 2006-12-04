"""
This module customizes the MayaVi2 UI and adds callbacks to the CitcomS
visualization plugins.
"""

# Enthought library imports.
from enthought.envisage.workbench.action.action_plugin_definition import \
     Action, Group, Location, Menu, WorkbenchActionSet

###############################################################################

citcoms_group = Group(id="CitcomsMenuGroup",
                      location=Location(path="MenuBar",
                                        after="FileMenuGroup"))

###############################################################################

citcoms_menu = Menu(
    id = "CitcomsMenu",
    name = "&CitcomS",
    location = Location(path="MenuBar/VisualizeMenu/additions",
                        after="FiltersMenu"),
)

###############################################################################

# old name: enthought.mayavi.plugins.OpenCitcomSFILES.OpenCitcomSVTKFILE
citcoms_open_vtk = Action(
    id          = "OpenCitcomsVTKFile",
    class_name  = "citcoms_display.actions.OpenVTKAction",
    name        = "&CitcomS VTK file",
    #image      = "images/new_scene.png",
    tooltip     = "Open a CitcomS VTK data file",
    description = "Open a CitcomS VTK data file",
    locations   = [Location(path="MenuBar/FileMenu/OpenMenu/additions",
                            after="OpenVTKFile"),]
)

# old name: enthought.mayavi.plugins.OpenCitcomSFILES.OpenCitcomSHDFFILE
citcoms_open_hdf = Action(
    id          = "OpenCitcomsHDF5File",
    class_name  = "citcoms_display.actions.OpenHDF5Action",
    name        = "CitcomS &HDF5 file",
    #image      = "images/new_scene.png",
    tooltip     = "Open a CitcomS HDF5 data file",
    description = "Open a CitcomS HDF5 data file",
    locations   = [Location(path="MenuBar/FileMenu/OpenMenu/additions",
                            after="OpenCitcomsVTKFile"),]
)

# old name: enthought.mayavi.plugins.CitcomSFilterActions.CitcomSshowCaps
citcoms_cap_filter = Action(
    id          = "CitcomsCapFilter",
    class_name  = "citcoms_display.mayavi_filters.ShowCapsFilter",
    name        = "&Show Citcom Caps",
    #image      = "images/new_scene.png",
    tooltip     = "Display a specified range of caps",
    description = "Display a specified range of caps",
    locations   = [Location(path="MenuBar/VisualizeMenu/CitcomsMenu/additions"),]
)

# old name: enthought.mayavi.plugins.CitcomSFilterActions.CitcomSreduce
citcoms_reduce_filter = Action(
    id          = "CitcomsReduceFilter",
    class_name  = "citcoms_display.mayavi_filters.ReduceFilter",
    name        = "&Reduce CitcomS grid",
    #image      = "images/new_scene.png",
    tooltip     = "Display a ReduceGrid for interpolation",
    description = "Display a ReduceGrid for interpolation",
    locations   = [Location(path="MenuBar/VisualizeMenu/CitcomsMenu/additions"),]
)

###############################################################################

action_set = WorkbenchActionSet(
    id = 'CitcomS.action_set',
    name = 'CitcomsActionSet',
    menus = [citcoms_menu],
    actions = [citcoms_open_vtk,
               citcoms_open_hdf,
               citcoms_cap_filter,
               citcoms_reduce_filter,]
)

###############################################################################

requires = []
extensions = [action_set]

