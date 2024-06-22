from typing import Dict, Optional, TypedDict, List
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


# Data model
class Code(BaseModel):
    """Plan to follow in future"""

    code: str = Field(
        description="Detailed optmized error-free Python code on the provided requirements"
    )


class Test(BaseModel):
    """Plan to follow in future"""

    Input: List[List] = Field(
        description="Input for Test cases to evaluate the provided code"
    )
    Output: List[List] = Field(
        description="Expected Output for Test cases to evaluate the provided code"
    )


class RefineCode(BaseModel):
    code: str = Field(
        description="Optimized and Refined Python code to resolve the error"
    )


class ExecutableCode(BaseModel):
    """Plan to follow in future"""

    code: str = Field(
        description="Detailed optmized error-free Python code with test cases assertion"
    )


class AgentCoder(TypedDict):
    requirement: str
    code: str
    tests: Dict[str, any]
    errors: Optional[str]


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
    errors: str
    tests: Dict[str, any]
    input: str
    output: str
    requirement: str
    code: str
    messages: List
    generation: str
    iterations: int


class CodeAssistant:
    def __init__(self):
        # Grader prompt
        self.app = None
        self.llm = ChatOpenAI(model="gpt-4o")
        # Prompt to enforce tool use

        # Coder Agent
        code_gen_prompt = ChatPromptTemplate.from_template(
            '''**Role**: You are a expert software python programmer. You need to develop python code
        **Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break
        down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is
        efficient, readable, and well-commented.

        **Instructions**:
        1. **Understand and Clarify**: Make sure you understand the task.
        2. **Algorithm/Method Selection**: Decide on the most efficient way.
        3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode.
        4. **Code Generation**: Translate your pseudocode into executable Python code

        *REQURIEMENT*
        {requirement}'''
        )
        self.coder = create_structured_output_runnable(
            Code, self.llm, code_gen_prompt
        )

        # Tester Agent
        test_gen_prompt = ChatPromptTemplate.from_template(
            '''**Role**: As a tester, your task is to create Basic and Simple test cases based on provided Requirement and Python Code. 
        These test cases should encompass Basic, Edge scenarios to ensure the code's robustness, reliability, and scalability.
        **1. Basic Test Cases**:
        - **Objective**: Basic and Small scale test cases to validate basic functioning 
        **2. Edge Test Cases**:
        - **Objective**: To evaluate the function's behavior under extreme or unusual conditions.
        **Instructions**:
        - Implement a comprehensive set of test cases based on requirements.
        - Pay special attention to edge cases as they often reveal hidden bugs.
        - Only Generate Basics and Edge cases which are small
        - Avoid generating Large scale and Medium scale test case. Focus only small, basic test-cases
        *REQURIEMENT*
        {requirement}
        **Code**
        {code}
        '''
        )
        self.tester_agent = create_structured_output_runnable(
            Test, self.llm, test_gen_prompt
        )

        # Execute Agent
        python_execution_gen = ChatPromptTemplate.from_template(
            """You have to add testing layer in the *Python Code* that can help to execute the code. You need to pass only Provided Input as argument and validate if the Given Expected Output is matched.
        *Instruction*:
        - Make sure to return the error if the assertion fails
        - Generate the code that can be execute
        Python Code to excecute:
        *Python Code*:{code}
        Input and Output For Code:
        *Input*:{input}
        *Expected Output*:{output}"""
        )
        self.execution = create_structured_output_runnable(
            ExecutableCode, self.llm, python_execution_gen
        )

        # Refine
        python_refine_gen = ChatPromptTemplate.from_template(
            """You are expert in Python Debugging. You have to analysis Given Code and Error and generate code that handles the error
            *Instructions*:
            - Make sure to generate error free code
            - Generated code is able to handle the error

            *Code*: {code}
            *Error*: {error}
            """
        )
        self.refine_code = create_structured_output_runnable(
            RefineCode, self.llm, python_refine_gen
        )

    def programmer(self, state):
        print(f'Entering in Programmer')
        requirement = state['requirement']
        code_ = self.coder.invoke({'requirement': requirement})
        return {'code': code_.code}

    def debugger(self, state):
        print(f'Entering in Debugger')
        errors = state['errors']
        code = state['code']
        refine_code_ = self.refine_code.invoke({'code': code, 'error': errors})
        return {'code': refine_code_.code, 'errors': None}

    def executer(self, state):
        print(f'Entering in Executer')
        tests = state['tests']
        input_ = tests['input']
        output_ = tests['output']
        code = state['code']
        executable_code = self.execution.invoke({"code": code, "input": input_, 'output': output_})
        # print(f"Executable Code - {executable_code.code}")
        error = None
        try:
            exec(executable_code.code)
            print("Code Execution Successful")
        except Exception as e:
            print('Found Error While Running')
            error = f"Execution Error : {e}"
        return {'code': executable_code.code, 'errors': error}

    def tester(self, state):
        print(f'Entering in Tester')
        requirement = state['requirement']
        code = state['code']
        tests = self.tester_agent.invoke({'requirement': requirement, 'code': code})
        # tester.invoke({'requirement':'Generate fibbinaco series','code':code_.code})
        return {'tests': {'input': tests.Input, 'output': tests.Output}}

    def decide_to_end(self, state):
        print(f'Entering in Decide to End')
        if state['errors']:
            return 'debugger'
        else:
            return 'end'

    def create_graph(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("programmer", self.programmer)
        workflow.add_node("debugger", self.debugger)
        workflow.add_node("executer", self.executer)
        workflow.add_node("tester", self.tester)
        # workflow.add_node('decide_to_end',decide_to_end)

        # Build graph
        workflow.set_entry_point("programmer")
        workflow.add_edge("programmer", "tester")
        workflow.add_edge("debugger", "executer")
        workflow.add_edge("tester", "executer")
        # workflow.add_edge("executer", "decide_to_end")

        workflow.add_conditional_edges(
            "executer",
            self.decide_to_end,
            {
                "end": END,
                "debugger": "debugger",
            },
        )

        # Compile
        self.app = workflow.compile()

    def invoke(self, input_query):
        requirement = """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.You can return the answer in any order."""
        config = {"recursion_limit": 50}
        inputs = {"requirement": input_query}
        running_dict = {}
        for event in self.app.stream(inputs, config=config):
            for k, v in event.items():
                running_dict[k] = v
                if k != "__end__":
                    print(v)
                    print('----------' * 20)
