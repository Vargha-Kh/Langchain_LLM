import asyncio
import re
import playwright
from IPython import display
from playwright.async_api import async_playwright, Page
from langchain_core.messages import BaseMessage, SystemMessage
import platform
from typing import TypedDict, List, Optional
import base64
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langchain_core.runnables import chain as chain_decorator
import nest_asyncio
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# This is just required for running async playwright in a Jupyter notebook
nest_asyncio.apply()


class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


# This represents the state of the agent
# as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool


class WebVoyager:
    def __init__(self):
        self.app = None
        self.prompt = hub.pull("wfh/web-voyager")
        self.llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=4096)
        self.mark_page_script = None


    # Define Tools
    # The agent has 6 simple tools:

    # Click (at labeled box)
    # Type
    # Scroll
    # Wait
    # Go back
    # Go to search engine (Google)

    async def click(self, state: AgentState):
        # - Click [Numerical_Label]
        page = state["page"]
        click_args = state["prediction"]["args"]
        if click_args is None or len(click_args) != 1:
            return f"Failed to click bounding box labeled as number {click_args}"
        bbox_id = click_args[0]
        bbox_id = int(bbox_id)
        try:
            bbox = state["bboxes"][bbox_id]
        except:
            return f"Error: no bbox for : {bbox_id}"
        x, y = bbox["x"], bbox["y"]
        res = await page.mouse.click(x, y)
        # TODO: In the paper, they automatically parse any downloaded PDFs
        # We could add something similar here as well and generally
        # improve response format.
        return f"Clicked {bbox_id}"

    async def type_text(self, state: AgentState):
        page = state["page"]
        type_args = state["prediction"]["args"]
        if type_args is None or len(type_args) != 2:
            return (
                f"Failed to type in element from bounding box labeled as number {type_args}"
            )
        bbox_id = type_args[0]
        bbox_id = int(bbox_id)
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        text_content = type_args[1]
        await page.mouse.click(x, y)
        select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
        await page.keyboard.press(select_all)
        await page.keyboard.press("Backspace")
        await page.keyboard.type(text_content)
        await page.keyboard.press("Enter")
        return f"Typed {text_content} and submitted"

    async def scroll(self, state: AgentState):
        page = state["page"]
        scroll_args = state["prediction"]["args"]
        if scroll_args is None or len(scroll_args) != 2:
            return "Failed to scroll due to incorrect arguments."

        target, direction = scroll_args

        if target.upper() == "WINDOW":
            # Not sure the best value for this:
            scroll_amount = 500
            scroll_direction = (
                -scroll_amount if direction.lower() == "up" else scroll_amount
            )
            await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        else:
            # Scrolling within a specific element
            scroll_amount = 200
            target_id = int(target)
            bbox = state["bboxes"][target_id]
            x, y = bbox["x"], bbox["y"]
            scroll_direction = (
                -scroll_amount if direction.lower() == "up" else scroll_amount
            )
            await page.mouse.move(x, y)
            await page.mouse.wheel(0, scroll_direction)

        return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

    async def wait(self, state: AgentState):
        sleep_time = 5
        await asyncio.sleep(sleep_time)
        return f"Waited for {sleep_time}s."

    async def go_back(self, state: AgentState):
        page = state["page"]
        await page.go_back()
        return f"Navigated back a page to {page.url}."

    async def to_google(self, state: AgentState):
        page = state["page"]
        await page.goto("https://www.google.com/")
        return "Navigated to google.com."

    async def annotate(self, state):
        marked_page = await self.mark_page.with_retry().ainvoke(state["page"])
        return {**state, **marked_page}

    def format_descriptions(self, state):
        labels = []
        for i, bbox in enumerate(state["bboxes"]):
            text = bbox.get("ariaLabel") or ""
            if not text.strip():
                text = bbox["text"]
            el_type = bbox.get("type")
            labels.append(f'{i} (<{el_type}/>): "{text}"')
        bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
        return {**state, "bbox_descriptions": bbox_descriptions}

    def parse(self, text: str) -> dict:
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
        action_block = text.strip().split("\n")[-1]

        action_str = action_block[len(action_prefix):]
        split_output = action_str.split(" ", 1)
        if len(split_output) == 1:
            action, action_input = split_output[0], None
        else:
            action, action_input = split_output
        action = action.strip()
        if action_input is not None:
            action_input = [
                inp.strip().strip("[]") for inp in action_input.strip().split(";")
            ]
        return {"action": action, "args": action_input}

    @chain_decorator
    async def mark_page(self, page):
        # Some javascript we will run on each step
        # to take a screenshot of the page, select the
        # elements to annotate, and add bounding boxes
        with open("mark_page.js") as f:
            self.mark_page_script = f.read()
        await page.evaluate(self.mark_page_script)
        for _ in range(10):
            try:
                bboxes = await page.evaluate("markPage()")
                break
            except:
                # May be loading...
                asyncio.sleep(3)
        screenshot = await page.screenshot()
        # Ensure the bboxes don't follow us around
        await page.evaluate("unmarkPage()")
        return {
            "img": base64.b64encode(screenshot).decode(),
            "bboxes": bboxes,
        }

    def update_scratchpad(self, state: AgentState):
        """After a tool is invoked, we want to update
        the scratchpad so the agent is aware of its previous steps"""
        old = state.get("scratchpad")
        if old:
            txt = old[0].content
            last_line = txt.rsplit("\n", 1)[-1]
            step = int(re.match(r"\d+", last_line).group()) + 1
        else:
            txt = "Previous action observations:\n"
            step = 1
        txt += f"\n{step}. {state['observation']}"

        return {**state, "scratchpad": [SystemMessage(content=txt)]}

    def create_graph(self):
        agent = self.annotate | RunnablePassthrough.assign(
            prediction=self.format_descriptions | self.prompt | self.llm | StrOutputParser() | self.parse
        )

        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("agent", agent)
        graph_builder.set_entry_point("agent")

        graph_builder.add_node("update_scratchpad", self.update_scratchpad)
        graph_builder.add_edge("update_scratchpad", "agent")

        tools = {
            "Click": self.click,
            "Type": self.type_text,
            "Scroll": self.scroll,
            "Wait": self.wait,
            "GoBack": self.go_back,
            "Google": self.to_google,
        }

        for node_name, tool in tools.items():
            graph_builder.add_node(
                node_name,
                # The lambda ensures the function's string output is mapped to the "observation"
                # key in the AgentState
                RunnableLambda(tool) | (lambda observation: {"observation": observation}),
            )
            # Always return to the agent (by means of the update-scratchpad node)
            graph_builder.add_edge(node_name, "update_scratchpad")

        def select_tool(state: AgentState):
            # Any time the agent completes, this function
            # is called to route the output to a tool or
            # to the end user.
            action = state["prediction"]["action"]
            if action == "ANSWER":
                return END
            if action == "retry":
                return "agent"
            return action

        graph_builder.add_conditional_edges("agent", select_tool)

        self.app = graph_builder.compile()

    async def call_agent(self, question: str, page, max_steps: int = 150):
        event_stream = self.app.astream(
            {
                "page": page,
                "input": question,
                "scratchpad": [],
            },
            {
                "recursion_limit": max_steps,
            },
        )
        final_answer = None
        steps = []
        async for event in event_stream:
            # We'll display an event stream here
            if "agent" not in event:
                continue
            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")
            display.clear_output(wait=False)
            steps.append(f"{len(steps) + 1}. {action}: {action_input}")
            print("\n".join(steps))
            display.display(display.Image(base64.b64decode(event["agent"]["img"])))
            if "ANSWER" in action:
                final_answer = action_input[0]
                break
        return final_answer

    async def invoke(self, question: str):
        browser = await async_playwright().start()
        # We will set headless=False so we can watch the agent navigate the web.
        browser = await browser.chromium.launch(headless=False, args=None)
        page = await browser.new_page()
        _ = await page.goto("https://www.google.com")

        res = await self.call_agent("Could you explain the WebVoyager paper (on arxiv)?", page)
        print(f"Final response: {res}")
