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


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# Retrieval Grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


# Hallucination Grader
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


# Answer Grader
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


class AgenticRAG:
    def __init__(self, retriever_tool):
        self.app = None
        self.tools = [retriever_tool]
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
        if "function_call" not in last_message.additional_kwargs:
            print("---DECISION: DO NOT RETRIEVE / DONE---")
            return "end"
        # Otherwise there is a function call, so we continue
        else:
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

    def retrieve(self, state):
        """
        Uses tool to execute retrieval.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with retrieved docs
        """
        print("---EXECUTE RETRIEVAL---")
        messages = state["messages"]
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            ),
        )
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)
        function_message = FunctionMessage(content=str(response), name=action.tool)

        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}

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
        question = messages[0].content
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

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
        workflow.add_node("retrieve", self.retrieve)  # retrieval
        workflow.add_node("rewrite", self.rewrite)  # retrieval
        workflow.add_node("generate", self.generate)  # retrieval

        # Call agent node to decide to retrieve or not
        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            self.should_retrieve,
            {
                # Call tool node
                "continue": "retrieve",
                "end": END,
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
        self.app = workflow.compile()

    def invoke(self, input_query):
        inputs = {
            "messages": [
                HumanMessage(
                    content=input_query
                )
            ]
        }
        for output in self.app.stream(inputs):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                pprint.pprint(value, indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")


class AdaptiveRAG:
    def __init__(self, retriever):
        self.retriever = retriever
        self.web_search_tool = TavilySearchResults(k=3)
        self.app = None
        # LLM with function call
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

        # Router
        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        self.question_router = route_prompt | structured_llm_router

        # Grader
        structured_llm_grader = llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        self.retrieval_grader = grade_prompt | structured_llm_grader

        # RAG Chain
        prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = prompt | llm | StrOutputParser()

        # Hallucination Grader
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)

        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        self.hallucination_grader = hallucination_prompt | structured_llm_grader

        # Answer Grader
        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
             Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        self.answer_grader = answer_prompt | structured_llm_grader

        # Question Re-Writer
        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
             for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
            ]
        )

        self.question_rewriter = re_write_prompt | llm | StrOutputParser()

    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    ### Edges ###

    def route_question(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        if source.datasource == 'web_search':
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == 'vectorstore':
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    @staticmethod
    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    def create_graph(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("web_search", self.web_search)  # web search
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate
        workflow.add_node("transform_query", self.transform_query)  # transform_query

        # Build graph
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        # Compile
        self.app = workflow.compile()

    def invoke(self, input_query):
        inputs = {"question": input_query}
        for output in self.app.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            print("\n---\n")
            # Final generation
            print(value["generation"])


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""
    response: str


class PlannerAgent:
    def __init__(self):
        # inti tools
        self.tools = [TavilySearchResults(max_results=3)]
        # Get the prompt to use - you can modify this!
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        # Choose the LLM that will drive the agent
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")

        self.app = None
        self.agent_executor = None
        self.replanner = None
        self.agent_runnable = None
        self.planner = None

    def agent_init(self):
        # Construct the OpenAI Functions agent
        self.agent_runnable = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = create_agent_executor(self.agent_runnable, self.tools)

    def planning(self):
        planner_prompt = ChatPromptTemplate.from_template(
            """For the given objective, come up with a simple step by step plan. \
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        {objective}"""
        )
        self.planner = create_structured_output_runnable(
            Plan, ChatOpenAI(model="gpt-4-turbo-preview", temperature=0), planner_prompt
        )

        replanner_prompt = ChatPromptTemplate.from_template(
            """For the given objective, come up with a simple step by step plan. \
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        Your objective was this:
        {input}

        Your original plan was this:
        {plan}

        You have currently done the follow steps:
        {past_steps}

        Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
        )

        self.replanner = create_openai_fn_runnable(
            [Plan, Response],
            ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
            replanner_prompt,
        )

    async def execute_step(self, state: PlanExecute):
        task = state["plan"][0]
        agent_response = await self.agent_executor.ainvoke({"input": task, "chat_history": []})
        return {
            "past_steps": (task, agent_response["agent_outcome"].return_values["output"])
        }

    async def plan_step(self, state: PlanExecute):
        plan = await self.planner.ainvoke({"objective": state["input"]})
        return {"plan": plan.steps}

    async def replan_step(self, state: PlanExecute):
        output = await self.replanner.ainvoke(state)
        if isinstance(output, Response):
            return {"response": output.response}
        else:
            return {"plan": output.steps}

    def create_graph(self):
        # Initialize openai agent's functions 
        self.agent_init()

        def should_end(state: PlanExecute):
            if state["response"]:
                return True
            else:
                return False

        workflow = StateGraph(PlanExecute)

        # Add the plan node
        workflow.add_node("planner", self.plan_step)

        # Add the execution step
        workflow.add_node("agent", self.execute_step)

        # Add a replan node
        workflow.add_node("replan", self.replan_step)

        workflow.set_entry_point("planner")

        # From plan we go to agent
        workflow.add_edge("planner", "agent")

        # From agent, we replan
        workflow.add_edge("agent", "replan")

        workflow.add_conditional_edges(
            "replan",
            # Next, we pass in the function that will determine which node is called next.
            should_end,
            {
                # If `tools`, then we call the tool node.
                True: END,
                False: "agent",
            },
        )
        # This compiles it into a LangChain Runnable,
        self.app = workflow.compile()

    def invoke(self, input_query):
        config = {"recursion_limit": 50}
        inputs = {"input": input_query}
        # event = self.app.stream(
        #     [HumanMessage(content=inputs)], config=config
        # )
        for event in self.app.astream(inputs, config=config):
            for k, v in event.items():
                if k != "__end__":
                    print(v)


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    @traceable
    def respond(self, state: List[BaseMessage]):
        response = []
        for attempt in range(3):
            try:
                response = self.runnable.invoke({"messages": state})
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response


# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )


class Reflexion:
    def __init__(self):
        self.revisor = None
        self.first_responder = None
        self.app = None
        search = TavilySearchAPIWrapper()
        tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
        # It takes in an agent action and calls that tool and returns the result
        self.tool_executor = ToolExecutor([tavily_tool])
        # Parse the tool messages for the execution / invocation
        self.parser = JsonOutputToolsParser(return_id=True)

    def create_chain(self):
        # actor prompt
        actor_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are expert researcher.
        Current time: {time}

        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. Recommend search queries to research information and improve your answer.""",
                ),
                MessagesPlaceholder(variable_name="messages"),
                ("system", "Answer the user's question above using the required format."),
            ]
        ).partial(
            time=lambda: datetime.datetime.now().isoformat(),
        )
        # Revisor Prompt
        revise_instructions = """Revise your previous answer using the new information.
                 - You should use the previous critique to add important information to your answer.
                     - You MUST include numerical citations in your revised answer to ensure it can be verified.
                     - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
                         - [1] https://example.com
                         - [2] https://example.com
                 - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
             """

        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")

        initial_answer_chain = actor_prompt_template.partial(
            first_instruction="Provide a detailed ~250 word answer."
        ) | self.llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        validator = PydanticToolsParser(tools=[AnswerQuestion])

        self.first_responder = ResponderWithRetries(
            runnable=initial_answer_chain, validator=validator
        )

        revision_chain = actor_prompt_template.partial(
            first_instruction=revise_instructions
        ) | self.llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
        revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

        self.revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)

    def execute_tools(self, state: List[BaseMessage]) -> List[BaseMessage]:
        tool_invocation: AIMessage = state[-1]
        parsed_tool_calls = self.parser.invoke(tool_invocation)
        ids = []
        tool_invocations = []
        for parsed_call in parsed_tool_calls:
            for query in parsed_call["args"]["search_queries"]:
                tool_invocations.append(
                    ToolInvocation(
                        # We only have this one for now. Would want to map it
                        # if we change
                        tool="tavily_search_results_json",
                        tool_input=query,
                    )
                )
                ids.append(parsed_call["id"])

        outputs = self.tool_executor.batch(tool_invocations)
        outputs_map = defaultdict(dict)
        for id_, output, invocation in zip(ids, outputs, tool_invocations):
            outputs_map[id_][invocation.tool_input] = output

        return [
            ToolMessage(content=json.dumps(query_outputs), tool_call_id=id_)
            for id_, query_outputs in outputs_map.items()
        ]

    def create_graph(self):
        # chain init
        self.create_chain()

        MAX_ITERATIONS = 5
        builder = MessageGraph()
        builder.add_node("draft", self.first_responder.respond)
        builder.add_node("execute_tools", self.execute_tools)
        builder.add_node("revise", self.revisor.respond)
        # draft -> execute_tools
        builder.add_edge("draft", "execute_tools")
        # execute_tools -> revise
        builder.add_edge("execute_tools", "revise")

        # Define looping logic:

        def _get_num_iterations(state: List[BaseMessage]):
            i = 0
            for m in state[::-1]:
                if not isinstance(m, (ToolMessage, AIMessage)):
                    break
                i += 1
            return i

        def event_loop(state: List[BaseMessage]) -> str:
            # in our case, we'll just stop after N plans
            num_iterations = _get_num_iterations(state)
            if num_iterations > MAX_ITERATIONS:
                return END
            return "execute_tools"

        # revise -> execute_tools OR end
        builder.add_conditional_edges("revise", event_loop)
        builder.set_entry_point("draft")
        # Compile
        self.app = builder.compile()

    def invoke(self, input_query):
        output, node = "", ""
        events = self.app.stream(
            [HumanMessage(content=input_query)]
        )
        for i, step in enumerate(events):
            node, output = next(iter(step.items()))
            print(f"## {i + 1}. {node}")
            print(str(output)[:1000] + " ...")
            print("---")
        if "__end__" == node:
            print(self.parser.invoke(step[END][-1])[0]["args"]["answer"])
        else:
            print(output.tool_calls[0]['args']['answer'])


# Data model
class code(BaseModel):
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
        self.code_gen_chain = code_gen_prompt | self.llm.with_structured_output(code)

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
        # flag = 'reflect'
        flag = 'do not reflect'
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
        results = self.app.invoke({"messages": [("user", input_query)], "iterations": 0})
        print(results)
