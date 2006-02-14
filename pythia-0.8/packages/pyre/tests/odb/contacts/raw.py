#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.applications.Script import Script


class RawApp(Script):


    def main(self, *args, **kwds):

        # print all the existing contacts
        print "dumping existing contacts:"
        for contact in self.retrieveContacts():
            print "    %s: %s" % (contact.id, contact)

        # create a new one
        print
        print "creating a new one"
        from Person import Person
        person = Person()
        person.id = 10002
        person.first = "Keri"
        person.middle = "Ann"
        person.last = "Aivazis"

        print "    %s: %s" % (person.id, person)
        self.storeContact(person)

        return


    def retrieveContacts(self):
        candidates = self.getCurator().retrieveShelves(address=[], extension="odb")

        for tag in candidates:
            yield self.retrieveContact(tag)
            
        return


    def retrieveContact(self, tag):
        contact = self.retrieveComponent(name=tag, factory='contact', encoding='odb')
        return contact


    def storeContact(self, person):

        text = [
            "",
            "def contact():",
            "",
            "    from Person import Person",
            "    person = Person()",
            "",
            "    person.id = %d" % person.id,
            "    person.first = %r" % person.first,
            "    person.middle = %r" % person.middle,
            "    person.last = %r" % person.last,
            "",
            "    return person",
            ""
            ]


        weaver = self.getCurator().codecs['odb'].renderer
        weaver.setCurator(self.getCurator())
        weaver.applyConfiguration()
        weaver.init()
        weaver.language = "python"

        weaver.begin()
        weaver.contents(text)
        weaver.end()

        print "\n".join(weaver.document())
            
        return


    def __init__(self):
        Script.__init__(self, 'raw')
        return


# main
if __name__ == '__main__':
    import journal
    # journal.debug("pyre.odb").activate()
    
    app = RawApp()
    app.run()


# version
__id__ = "$Id: raw.py,v 1.2 2005/03/11 07:00:01 aivazis Exp $"

# End of file 
