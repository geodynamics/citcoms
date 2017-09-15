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

citcoms_menu = Menu(
    id = "CitcomsMenu",
    name = "&CitcomS",
    location = Location(path="MenuBar/CitcomsMenuGroup"),
)

citcoms_open_menu = Menu(
    id = "CitcomsOpenMenu",
    name = "&Open",
    location = Location(path="MenuBar/CitcomsMenu/additions"),
)

citcoms_modules_menu = Menu(
    id = "CitcomsModulesMenu",
    name = "&Modules",
    location = Location(path="MenuBar/CitcomsMenu/additions",
                        after="CitcomsOpenMenu"),
)

citcoms_filters_menu = Menu(
    id = "CitcomsFiltersMenu",
    name = "&Filters",
    location = Location(path="MenuBar/CitcomsMenu/additions",
                        after="CitcomsModulesMenu"),
)

###############################################################################

# old name: enthought.mayavi.plugins.OpenCitcomSFILES.OpenCitcomSVTKFILE
citcoms_open_vtk = Action(
    id          = "OpenCitcomsVTKFile",
    class_name  = "citcoms_display.actions.OpenVTKAction",
    name        = "CitcomS &VTK file",
    #image      = "images/new_scene.png",
    tooltip     = "Open a CitcomS VTK data file",
    description = "Open a CitcomS VTK data file",
    locations   = [Location(path="MenuBar/CitcomsMenu/CitcomsOpenMenu/additions")]
)

# old name: enthought.mayavi.plugins.OpenCitcomSFILES.OpenCitcomSHDFFILE
citcoms_open_hdf = Action(
    id          = "OpenCitcomsHDF5File",
    class_name  = "citcoms_display.actions.OpenHDF5Action",
    name        = "CitcomS &HDF5 file",
    #image      = "images/new_scene.png",
    tooltip     = "Open a CitcomS HDF5 data file",
    description = "Open a CitcomS HDF5 data file",
    locations   = [Location(path="MenuBar/CitcomsMenu/CitcomsOpenMenu/additions",
                            after="OpenCitcomsVTKFile"),]
)

# old name: enthought.mayavi.plugins.CitcomSFilterActions.CitcomSreduce
citcoms_reduce_filter = Action(
    id          = "CitcomsReduceFilter",
    class_name  = "citcoms_display.actions.ReduceFilterAction",
    name        = "&Reduce Grid",
    #image      = "images/new_scene.png",
    tooltip     = "Display a ReduceGrid for interpolation",
    description = "Display a ReduceGrid for interpolation",
    locations   = [Location(path="MenuBar/CitcomsMenu/CitcomsFiltersMenu/additions"),]
)

# old name: enthought.mayavi.plugins.CitcomSFilterActions.CitcomSshowCaps
citcoms_cap_filter = Action(
    id          = "CitcomsShowCapsFilter",
    class_name  = "citcoms_display.actions.ShowCapsFilterAction",
    name        = "&Show Caps",
    #image      = "images/new_scene.png",
    tooltip     = "Display a specified range of caps",
    description = "Display a specified range of caps",
    locations   = [Location(path="MenuBar/CitcomsMenu/CitcomsFiltersMenu/additions"),]
)

###############################################################################

action_set = WorkbenchActionSet(
    id      = 'citcoms_display.action_set',
    name    = 'CitcomsActionSet',
    groups  = [citcoms_group],
    menus   = [citcoms_menu,
               citcoms_open_menu,
               citcoms_modules_menu,
               citcoms_filters_menu,],
    actions = [citcoms_open_vtk,
               citcoms_open_hdf,
               citcoms_reduce_filter,
               citcoms_cap_filter,]
)

###############################################################################

requires = []
extensions = [action_set]

