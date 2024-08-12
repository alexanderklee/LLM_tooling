from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler


import signal
import sys
from dotenv import load_dotenv

def signal_handler(sig, frame):
    print('You pressed Ctrl+C, goodbye!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

load_dotenv()

# Default messags are super messy!
handler = ChatModelStartHandler()

chat = ChatOpenAI(
    callbacks=[handler]
)

tables = list_tables()
print(tables)

prmopt = ChatPromptTemplate(
    messages=[
        # SystemMessage below is too generic and allows OAI to choose the wrong tools or
        # avoid certain tool for some reason. 
        # SystemMessage(content="You are an AI that has access to a SQLite database."),
        
        # Provide a more specific/direct SystemMessage to force OAI to use certain tools. The
        # prompt below works and enables OAI to return the correct answer.
        SystemMessage(content=(
                      "You are an AI that has access to a SQLite database.\n"
                      f"The database has tables of: {tables}\n"
                      "Do not make any assumptions about what tables exist or what columns exist."
                      "Instead, use the 'describe_tables' function."
        )),
        # second placeholder to store intermediate_data between executor runs
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad") # note: agent_scratchpad is a keyword
    ]
)

# Initialize memory for chat history to resolve the context loss between executor runs
# note: I keep forgetting, return_messages=True means returns list of messages as Message objects
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# light refactor - Tools list
tools = [run_query_tool, describe_tables_tool, write_report_tool]

# note: agent is the "thing" that knows how to use tools
# note: this replaces initial_agents()
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prmopt,
    tools=tools
)

# note: executors takes an agent (chain) repeatadly until there is no
#       request for a call to a tool
agent_executor = AgentExecutor(
    agent=agent,
#    verbose=True,
    tools=tools,
    memory=memory
)

# ------TEST SCENARIOS -------

# Easy prompt that works with minimal effort simply requesting the total 
# number of users in a database. OAI can guess its way because the database
# schema is defined in a manner OAI assumes. 

# agent_executor("How many users are in the database?")

# -------------

# Harder prompt because OAI's assumptions will be incorrect based on the prompt. 
# This will fail with OAI stating it doesn't know the database enough so additoinal
# details (or tools) will be needed.
# To make this work, we need an additioanl tool to query
# table schemas and  updates to Function descriptions to ensure OAI understands
# what the tools purpose are. This requires creating custom classes to require/alter
# LangChain's default behavior (ie. the __args1)
  
# Reason: below doesn't work to get # of users rather OAI assumes a column
# named shipping_addresses exists. Check in OAI's playground on what
# tables it would create for an ecommerce app and shipping_addresses
# is not one. How do you get OAI to inform it about the existing tables

# agent_executor("How many users provided shipping addresses?")

# -------------

# Prompt that requires multiple arguments within a single Tool that
# generates a report in HTML. The use of Tools is "strange" because it assumes
# a single argument. So given more complex Tools that require mulitple arguments, 
# StructuredTool is required. As of now, it seems Tools might be on the road of
# deprecation, which is good.

# agent_executor("Summarize the top 5 most popular products. Write the results to a reports file.")

# -------------

# Testing memory vs. scratchpad. Current implemention used scratchpad and not Memory since
# the two architectures seemed to do the same thing. The two consecutive prompts share implicity
# ideas with one another and will scratchpad preserve message history the same way as Memory. 
# TLDR, no. OAI can execute the first prompt, however the second one will only return the user
# schema table because it has no context of the first prompt. 
# Reason: AgentExecutor only stores intermediate_context in scratchpad while it is running. 
# Therefore, a second call to agent_executir will have none of the intermediate_context from the
# first executor run.
# Challenge: Will need to implement some from of Memory,similar to chat. However, the intermediate_steps
# is not persistend between executor calls; just the AssistantMessage is preserved. Will need to somehow
# leverage Memory and AgentExecutor scratchpad in order to preserve context between executor runs and
# of course, the HumanMessage as well.
# Solution: implement ConversationBufferMemory and add a second MessagesPlaceholder in the 
# ChatPromptTemplate. This will preserve the HumanPrompt, AssistantPrmopt, and the intermediate_steps
# information when subsequent executors are ran in a chat.

agent_executor(
    "How many orders are there? Write the result to an html report."
)
agent_executor(
    "Repeat the same process for users."
)