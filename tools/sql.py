import sqlite3
from langchain.tools import Tool, StructuredTool
# importing BaseModel and List to add better descriptions to
# the Functions argument descriptors since LangChain uses
# __argsX, which isn't helpful for OAI
from pydantic.v1 import BaseModel
from typing import List
conn = sqlite3.connect("db.sqlite")

def list_tables():
    c = conn.cursor()
    # note: providing ChatGPT with table names results in OAI aggressively probing
    # tables until it finds the right column name
    rows = c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    # rows is a tuple and not good for OAI's understanding, convert to a string
    # return rows
    
    # let's return a string of table names instead. 
    # note: even with table names known, OAI needs more information about
    #       the database. Will need to implement a new Tool for this.
    return "\n".join(row[0] for row in rows if row[0] is not None)

## General note about Tools and OAI. By default, you cannot force OAI
## to use Tools. To do this requires more direct SystemMessage prompting

def run_sqlite_query(query):
    c = conn.cursor()
    # need to capture errors returned from the database and if
    # an error exists, return error back to ChatGPT to inform it
    # that the SQL string is bad and to make adjustments. 
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        # returning the errors is not enough to GPT to guess the right
        # column and tables. 
        return f"The following error occured: {str(err)}"

# Override LangChain's descriptors to use something more meaningful
# by creating a class (extending from BaseModel, to ensure OAI gets
# meaning function descriptions from LangChain. In this instance, 
# we want run_query_tool to be represented as a querying tool. 
class RunQueryArgsSchema(BaseModel):
    query: str
    
run_query_tool = StructuredTool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema
)

# Creating new Tool to provide OAI a way of getting table schema information 
def describe_tables(table_names):
    c = conn.cursor()
    
    # "'users,'orders','products'.... <-- structure this way
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return "\n".join(row[0] for row in rows if row[0] is not None)

# Similar to RunQeryArgsSchema for run_query_tool, we need to tell OAI
# that the describe_tables_tool returns a list of table names and not LangChain's
# default of __arg1. Create a new class with describing the argument and data type
# will help OAI understand this tools purpose and organization.
class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]

describe_tables_tool = StructuredTool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables.",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema
)