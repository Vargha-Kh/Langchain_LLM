import datetime
import operator
from collections import defaultdict
from typing import Annotated, Sequence, TypedDict, List, Literal, Tuple
from langchain.agents import create_openai_functions_agent
from langchain.chains.openai_functions import create_structured_output_runnable, create_openai_fn_runnable
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, MessageGraph
from langchain_core.messages import BaseMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import json
import operator
import pprint
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain.prompts import PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, FunctionMessage
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation, ToolExecutor, create_agent_executor
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langsmith import traceable

class Code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: List
    generation: str
    iterations: int


class CodeAssistant:
    def __init__(self, vector_db):
        # Grader prompt
        self.app = None
        self.vector_db = vector_db
        code_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", """You are a coding assistant with expertise in python. \n 
            Here is a full set of codes and documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
            question based on the above provided documentation. Ensure any code you provide can be executed \n 
            with all required imports and variables defined. Structure your answer with a description of the code solution. \n
            Then list the imports. And finally list the functioning code block. Here is the user question:"""),
             ("placeholder", "{messages}")]
        )

        expt_llm = "gpt-4-0125-preview"
        self.llm = ChatOpenAI(temperature=0, model=expt_llm)
        self.code_gen_chain = code_gen_prompt | self.llm.with_structured_output(Code)

    ### Nodes
    def generate(self, state: GraphState):
        """
        Generate a code solution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        print("---GENERATING CODE SOLUTION---")

        # State
        messages = state["messages"]
        iterations = state["iterations"]
        error = state["error"]

        # We have been routed back to generation with an error
        if error == "yes":
            messages += [("user",
                          "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:")]

        # Solution
        code_solution = self.code_gen_chain.invoke({"context": self.vector_db, "messages": messages})
        messages += [
            ("assistant", f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}")]

        # Increment
        iterations = iterations + 1
        return {"generation": code_solution, "messages": messages, "iterations": iterations}

    def code_check(self, state: GraphState):
        """
        Check code

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        print("---CHECKING CODE---")

        # State
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]

        # Get solution components
        prefix = code_solution.prefix
        imports = code_solution.imports
        code = code_solution.code

        # Check imports
        try:
            exec(imports)
        except Exception as e:
            print("---CODE IMPORT CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the import test: {e}")]
            messages += error_message
            return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}

        # Check execution
        try:
            exec(imports + "\n" + code)
        except Exception as e:
            print("---CODE BLOCK CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the code execution test: {e}")]
            messages += error_message
            return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}

        # No errors
        print("---NO CODE TEST FAILURES---")
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "no"}

    def reflect(self, state: GraphState):
        """
        Reflect on errors

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        print("---GENERATING CODE SOLUTION---")

        # State
        messages = state["messages"]
        iterations = state["iterations"]
        code_solution = state["generation"]

        # Prompt reflection
        reflection_message = [("user", """You tried to solve this problem and failed a unit test. Reflect on this failure
                                        given the provided documentation. Write a few key suggestions based on the 
                                        documentation to avoid making this mistake again.""")]

        # Add reflection
        reflections = self.code_gen_chain.invoke({"context": self.vector_db, "messages": messages})
        messages += [("assistant", f"Here are reflections on the error: {reflections}")]
        return {"generation": code_solution, "messages": messages, "iterations": iterations}

    ### Edges
    def decide_to_finish(self, state: GraphState):
        """
        Determines whether to finish.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        # Max tries
        max_iterations = 3
        # Reflect
        flag = 'reflect'
        # flag = 'do not reflect'
        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations == max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            if flag == 'reflect':
                return "reflect"
            else:
                return "generate"

    def parse_output(self, solution):
        """When we add 'include_raw=True' to structured output,
           it will return a dict w 'raw', 'parsed', 'parsing_error'. """

        return solution['parsed']

    def insert_errors(self, inputs):
        """Insert errors for tool parsing in the messages"""

        # Get errors
        error = inputs["error"]
        messages = inputs["messages"]
        messages += [
            (
                "assistant",
                f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
            )
        ]
        return {
            "messages": messages,
            "context": inputs["context"],
        }

    def create_graph(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("generate", self.generate)  # generation solution
        workflow.add_node("check_code", self.code_check)  # check code
        workflow.add_node("reflect", self.reflect)  # reflect

        # Build graph
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "check_code")
        workflow.add_conditional_edges(
            "check_code",
            self.decide_to_finish,
            {
                "end": END,
                "reflect": "reflect",
                "generate": "generate",
            },
        )
        workflow.add_edge("reflect", "generate")
        self.app = workflow.compile()

    def invoke(self, input_query):
        results = self.app.invoke({"messages": [("user", input_query)], "iterations": 3})
        print("Message: ")
        print(results['generation'].prefix)
        print("Code: ")
        print(results['generation'].code)
