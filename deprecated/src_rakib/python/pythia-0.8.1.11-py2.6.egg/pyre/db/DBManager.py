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


class DBManager(object):


    def autocommit(self, flag=True):
        return self.db.autocommit(flag)


    def commit(self):
        return self.db.commit()


    def connect(self, **kwds):
        raise NotImplementedError("class %r must override 'connect'" % self.__class__.__name__)


    def cursor(self):
        return self.db.cursor()


    def insertRow(self, row, tableName=None):
        # build the arguments
        if tableName is None:
            name = row.name
        else:
            name = tableName
            
        values = row.getWriteableValues()
        columns = row.getWriteableColumnNames()

        # build the sql query
        sql = "INSERT INTO %s (\n    %s\n    ) VALUES (\n    %s\n    )" % (
            name, ", ".join(columns), ", ".join([self.placeholder] * len(columns)))

        # execute the sql statement
        c = self.db.cursor()
        c.execute(sql, values)
        self.db.commit()

        return


    def updateRow(self, table, assignments, where):

        columns = []
        values = []

        for column, value in assignments:
            columns.append(column)
            values.append(value)

        expr = ", ".join(["%s=%s" % (column, self.placeholder) for column in columns])
        sql = "UPDATE %s\n    SET %s\n    WHERE %s" % (table.name, expr, where)

        # execute the sql statement
        c = self.db.cursor()
        c.execute(sql, values)
        self.db.commit()

        return


    def createTable(self, table):
        # build the list of table columns
        fields = []
        for name, column in table._columnRegistry.iteritems():
            text = "    %s %s" % (name, column.declaration())
            fields.append(text)

        # build the query
        sql = "CREATE TABLE %s (\n%s\n    )" % (table.name, ",\n".join(fields))

        # execute the sql statement
        c = self.db.cursor()
        c.execute(sql)

        return


    def dropTable(self, table, cascade=False):
        sql = "DROP TABLE %s" % table.name
        if cascade:
            sql += " CASCADE"

        # execute the sql statement
        c = self.db.cursor()
        c.execute(sql)

        return


    def fetchall(self, table, where=None, sort=None):
        columns = table._columnRegistry.keys()
        
        # build the sql statement
        sql = "SELECT %s FROM %s" % (", ".join(columns), table.name)
        if where:
            sql += " WHERE %s" % where
        if sort:
            sql += " ORDER BY %s" % sort
        
        # execute the sql statement
        c = self.db.cursor()
        c.execute(sql)

        # walk through the result of the query
        items = []
        for row in c.fetchall():
            # create the object
            item = table()
            item.locator = self.locator
            
            # build the dictionary with the column information
            values = {}
            for key, value in zip(columns, row):
                values[key] = value
            # attach it to the object
            item._priv_columns = values

            # add this object tothepile
            items.append(item)

        return items


    def __init__(self, name):
        self.db = self.connect(database=name)

        import pyre.parsing.locators
        self.locator = pyre.parsing.locators.simple("%s database" % name)
        return


    placeholder = "%s"


# version
__id__ = "$Id: DBManager.py,v 1.12 2005/04/24 01:16:42 pyre Exp $"

# End of file 
