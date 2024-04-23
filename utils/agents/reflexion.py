import datetime
import operator
from collections import defaultdict
from typing import TypedDict, List
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, MessageGraph
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import json
from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langsmith import traceable


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
