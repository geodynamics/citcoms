#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def main():


    from opal.applications.WebApplication import WebApplication


    class HtmlApp(WebApplication):


        def main(self, *args, **kwds):
            import opal.content
            page = opal.content.page()

            head = page.head()
            head.attributes['lang'] = "en-us"

            head.title("Sample page")
            head.meta(name="description", content="test page")
            head.base(url="/opaldemo")
            head.link(rel="stylesheet", type="text/css", href="visual.css")

            style = head.stylesheet(
                url="css/visual.css",
                rel="stylesheet", type="text/css", media="all")

            script = head.script(type="text/javascript")
            script.script = ['document.write("Hello world!")']

            head.meta(name="description", content="sample page")

            body = page.body(id="#page-body")
            
            self.render(page)
            return


        def __init__(self):
            WebApplication.__init__(self, name='html', asCGI=False)
            return


    app = HtmlApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: html.py,v 1.1 2005/03/20 07:29:55 aivazis Exp $"

# End of file 
