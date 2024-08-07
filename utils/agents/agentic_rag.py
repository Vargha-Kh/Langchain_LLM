import json
import pprint
from langchain import hub
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import HumanMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolInvocation, ToolExecutor, ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence, TypedDict, List, Literal
import operator
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from threading import Lock

# Lock to serialize access to the SQLite connection
sqlite_lock = Lock()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]


class AgenticRAG:
    def __init__(self, retriever_tool):
        self.retriever_tool = retriever_tool
        self.app = None
        self.memory_config = None
        self.tools = [self.retriever_tool]
        self.tool_executor = ToolExecutor(self.tools)

    ## Edges
    @staticmethod
    def should_retrieve(state):
        """
        Decides whether the agent should retrieve more information or end the process.

        This function checks the last message in the state for a function call. If a function call is
        present, the process continues to retrieve information. Otherwise, it ends the process.

        Args:
            state (messages): The current state

        Returns:
            str: A decision to either "continue" the retrieval process or "end" it
        """

        print("---DECIDE TO RETRIEVE---")
        messages = state["messages"]
        last_message = messages[-1]

        # If there is no function call, then we finish
        # if "function_call" not in last_message.additional_kwargs:
        #     print("---DECISION: DO NOT RETRIEVE / DONE---")
        #     return "end"
        # # Otherwise there is a function call, so we continue
        # else:
        print("---DECISION: RETRIEVE---")
        return "continue"

    @staticmethod
    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

        # Tool
        grade_tool_oai = convert_to_openai_tool(grade)

        # LLM with tool and enforce invocation
        llm_with_tool = model.bind(
            tools=[convert_to_openai_tool(grade_tool_oai)],
            tool_choice={"type": "function", "function": {"name": "grade"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[grade])

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool | parser_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        score = chain.invoke(
            {"question": question,
             "context": docs}
        )

        grade = score[0].binary_score

        if grade == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "yes"

        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(grade)
            return "no"

    ## Nodes
    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response apended to messages
        """
        print("---CALL AGENT---")
        messages = state["messages"]
        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-0125-preview")
        functions = [format_tool_to_openai_function(t) for t in self.tools]
        model = model.bind_functions(functions)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    @staticmethod
    def rewrite(state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [HumanMessage(
            content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            additional_kwargs={}
        )]

        # Grader
        model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
        response = model.invoke(msg)
        return {"messages": [response]}

    @staticmethod
    def generate(state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
             dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")
        visa_prompt = """ You are an AI assistant specialized in Australia visa and immigration topics. Utilize the provided context to answer the questions accurately, explain specifically not general, precise and give practical answer. If the answer is unknown, clearly state that you do not know. Do not fabricate responses. Use references in your responses. If you dont have sufficient data to answer, simply ask the user to get the information.
                Previous Interactions:
                {chat_history}

                Relevant Context:
                {context}

                Question:
                {question}

                Helpful Answer:
                """

        # LLM
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

    def create_graph(self):
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", self.agent)  # agent
        retrieve = ToolNode([self.retriever_tool])
        workflow.add_node("retrieve", retrieve)  # retrieval
        workflow.add_node("rewrite", self.rewrite)  # retrieval
        workflow.add_node("generate", self.generate)  # retrieval

        # Call agent node to decide to retrieve or not
        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            self.grade_documents,
            {
                "yes": "generate",
                "no": "rewrite",
            },
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        # Compile
        conn = sqlite3.connect("checkpoints.sqlite")
        memory = SqliteSaver(conn)
        self.app = workflow.compile(checkpointer=memory)
        self.memory_config = {"configurable": {'thread_id': '1'}}
        self.app.get_state(self.memory_config)

    def invoke(self, input_query):
        output_response = ""
        inputs = {
            "messages": [
                HumanMessage(
                    content=input_query,
                    additional_kwargs={"function_call": True}
                )
            ]
        }
        for output in self.app.stream(inputs, self.memory_config):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                # pprint.pprint(value, indent=2, width=80, depth=None)
                output_response += str(value['messages'][0].content)
            # pprint.pprint("\n---\n")
        return output_response
