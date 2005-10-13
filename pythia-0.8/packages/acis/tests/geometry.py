#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.applications.Script import Script


class GeometryApp(Script):


    class Inventory(Script.Inventory):

        import pyre.inventory

        modeller = pyre.inventory.facility("modeller", default="cube")
        format = pyre.inventory.str(
            "format", default="sat",
            validator=pyre.inventory.choice(["sat", "tec", "vtu"]))


    def main(self):
        body = self.modeller.model()

        name = self.modeller.name
        format = self.format
        filename = name + '.' + format
        stream = file(filename, "w")
        print "saving body '%s' in '%s'" % (name, filename)

        if format == "sat":
            self._sat(body, stream)
        elif format == "tec":
            mesh, bbox = self._facet(body)
            self._tecplot(mesh, stream)
        elif format == "vtu":
            mesh, bbox = self._facet(body)
            self._paraview(mesh, stream)
        else:
            import journal
            firewall = journal.firewall(self.name)
            firewall.log("unknown format '%s'" % format)

        return


    def __init__(self):
        Script.__init__(self, "geometry")
        self.format = ""
        self.modeller = None
        return


    def _configure(self):
        self.format = self.inventory.format
        self.modeller = self.inventory.modeller
        return


    def _facet(self, body):
        import acis
        mesher = acis.surfaceMesher()
    
        properties = mesher.inventory
        properties.gridAspectRatio = 1.0
        properties.maximumEdgeLength = 0.01
    
        triangulation = mesher.facet(body)
        return triangulation


    def _sat(self, body, stream):
        import acis
        acis.save(stream, [body])
        return


    def _tecplot(self, mesh, stream):
        dim, order, vertices, simplices = mesh.statistics()

        stream.write("variables= x y z\n")
        stream.write("zone N=%d, E=%d, F=FEPOINT ET=triangle\n" % (vertices, simplices))

        for x,y,z in mesh.vertices():
            stream.write("%15.9g %15.9g %15.9g\n" % (x,y,z))

        for n0, n1, n2 in mesh.simplices():
            stream.write("%d %d %d\n" % (n0+1, n1+1, n2+1)) # tecplot is 1-based
    
        return


    def _paraview(self, mesh, stream):
        dim, order, vertices, simplices = mesh.statistics()

        stream.write('?xml version="1.0"?>\n')
        stream.write('<VTKFile type="UnstructuredGrid">\n')
        stream.write('<UnstructuredGrid>\n')
        stream.write('<Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (vertices, simplices))

        stream.write('<Points>\n')
        stream.write(
            '<DataArray type="Float32" NumberOfComponents="%d" format="ascii">\n' % order)
        for x,y,z in mesh.vertices():
            stream.write("%15.9g %15.9g %15.9g\n" % (x,y,z))
        stream.write('</DataArray>\n')
        stream.write('</Points>\n')

        stream.write('<Cells>\n')
        stream.write(
            '<DataArray type="Int32" name="connectivity" format="ascii">\n')
        for n0, n1, n2 in mesh.simplices():
            stream.write("%d %d %d\n" % (n0+1, n1+1, n2+1)) # tecplot is 1-based
        stream.write('</DataArray>\n')
        stream.write('</Cells>\n')

        stream.write('</Piece>\n')
        stream.write('</UnstructuredGrid>\n')
        stream.write('</VTKFile>\n')
        return
    


# main

if __name__ == "__main__":
    import journal
    journal.debug("pyre.geometry").activate()
    journal.debug("acis.faceting").activate()
    
    app = GeometryApp()
    app.run()


# version
__id__ = "$Id: geometry.py,v 1.1.1.1 2005/03/08 16:13:35 aivazis Exp $"

#
# End of file
