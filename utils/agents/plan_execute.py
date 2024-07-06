from typing import Annotated, TypedDict, List, Tuple
from langchain.agents import create_openai_functions_agent
from langchain.chains.openai_functions import create_structured_output_runnable, create_openai_fn_runnable
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
import operator
from langchain import hub
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation, ToolExecutor, create_agent_executor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from threading import Lock

# Lock to serialize access to the SQLite connection
sqlite_lock = Lock()


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
        conn = sqlite3.connect("checkpoints.sqlite")
        memory = SqliteSaver(conn)
        self.app = workflow.compile(checkpointer=memory)
        self.memory_config = {"configurable": {'thread_id': '1'}}
        self.app.get_state(self.memory_config)

    def invoke(self, input_query):
        config = {"recursion_limit": 50, "configurable": {'thread_id': '1'}}
        inputs = {"input": input_query}
        # event = self.app.stream(
        #     [HumanMessage(content=inputs)], config=config
        # )
        for event in self.app.astream(inputs, config=config):
            for k, v in event.items():
                if k != "__end__":
                    print(v)
