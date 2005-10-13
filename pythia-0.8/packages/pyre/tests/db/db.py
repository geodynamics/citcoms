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


    from pyre.db.Table import Table

    class User(Table):

        name = "users"

        import pyre.db

        username = pyre.db.varchar(name="username", length=30)
        username.meta['tip'] = "the user's name"

        password = pyre.db.varchar(name="password", length=30)
        password.meta['tip'] = "the user's password"


    from pyre.applications.Script import Script


    class DbApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory

            db = pyre.inventory.str('db', default='testdb')
            db.meta['tip'] = 'the database to connect to'

            wipe = pyre.inventory.bool('wipe', default=True)
            wipe.meta['tip'] = 'delete the table before inserting data?'

            create = pyre.inventory.bool('create', default=True)
            create.meta['tip'] = 'create the table before inserting data?'


        def main(self, *args, **kwds):
            print "database:", self.inventory.db
            print "database manager:", self.db

            if self.inventory.wipe:
                self.dropTable(User)

            if self.inventory.wipe or self.inventory.create:
                self.createTable(User)
            
            # create a user record
            user = User()
            user.username = "aivazis"
            user.password = "mga4demo"

            # store it in the database
            self.save(user)
            
            # now extract all records and print them
            self.retrieve(User)

            return


        def retrieve(self, table):
            print " -- retrieving from table %r" % table.name
            try:
                users = self.db.fetchall(table)
                print users
            except self.db.ProgrammingError, msg:
                print "    retrieve failed:", msg
            else:
                print "    success"

            index = 0
            for user in users:
                index += 1
                print "user %d: name=%s, password=%s" % (index, user.username, user.password)

            return


        def save(self, item):
            print " -- saving into table %r" % item.name
            try:
                self.db.insertRow(item)
            except self.db.ProgrammingError, msg:
                print "    insert failed:", msg
            else:
                print "    success"
                
            return


        def createTable(self, table):
            # create the user table
            print " -- creating table %r" % table.name
            try:
                self.db.createTable(table)
            except self.db.ProgrammingError:
                print "    failed; table exists?"
            else:
                print "    success"

            return

            
        def dropTable(self, table):
            # drop the user table
            print " -- dropping table %r" % table.name
            try:
                self.db.dropTable(table)
            except self.db.ProgrammingError:
                print "    failed; table doesn't exist?"
            else:
                print "    success"

            return


        def __init__(self):
            Script.__init__(self, 'db')
            self.db = None
            return


        def _init(self):
            Script._init(self)

            import pyre.db
            self.db = pyre.db.connect(self.inventory.db)

            return


    app = DbApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: db.py,v 1.3 2005/04/06 03:26:06 aivazis Exp $"

# End of file 
