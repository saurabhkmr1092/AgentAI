import os
import json
import shutil
import subprocess
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import StateGraph, START,END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage


class State(TypedDict):
    messages:Annotated[list,add_messages]
    plan: dict

@tool
def run_command(cmd: str):
    """
    Execute a shell command on the user's computer and return the output.

    Example commands:
    mkdir chat_gpt
    New-Item -ItemType Directory -Path chat_gpt
    pip install fastapi
    """
    try: 
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        output = result.stdout if result.stdout else result.stderr
        return f"Command executed.\nOutput:\n{output}"
    except Exception as e:
        return f"Error executing command: {str(e)}"    
    

@tool
def create_file(path: str):
    """Create an empty file."""
    try:
        open(path, "w").close()
        return f"{path} created."
    except Exception as e:
        return f"Error creating file: {str(e)}"


@tool
def write_file(path: str, content: str):
    """
    Write multi-line content to a file.

    This tool is used to write the initial code into a file that has
    already been created using create_file.
    
    It will NOT create a file or overwrite a missing file.

    Example:
    write_file(
        path="chat_gpt/bst.py",
        content="print('hello')"
    )
    """

    try:

        if not os.path.exists(path):
            return f"Error: File {path} does not exist. Create it first using create_file."

        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

        return f"File successfully written to {path}"

    except Exception as e:
        return f"Error writing file: {str(e)}"    



@tool
def read_file(path: str):
    """
    Read the contents of a file.
    Useful when the AI needs to inspect existing code before modifying it.
    """

    try:

        if not os.path.exists(path):
            return f"Error: File {path} does not exist."
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def append_file(path: str, content: str):
    """
    Append content to an existing file.
    Useful for adding new functions or extending code.
    """

    try:
        
        if not os.path.exists(path):
            return f"Error: File {path} does not exist. Create it first using create_file."
        
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + content.strip())

        return f"Content appended to {path}"

    except Exception as e:
        return f"Error appending to file: {str(e)}"


@tool
def run_python(path: str):
    """
    Run a Python file and return the output or error.
    """

    try:

        if not os.path.exists(path):
            return f"Error: File {path} does not exist."
        
        result = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr

        return f"Execution result:\n{output}"

    except Exception as e:
        return f"Error running python file: {str(e)}"


@tool
def list_directory(path: str = "chat_gpt"):
    """
    List all files and folders in a directory.
    Default directory is chat_gpt.
    """

    try:

        if not os.path.exists(path):
            return f"Error: Directory {path} does not exist."

        files = os.listdir(path)
        if not files:
            return "Directory is empty."
        return "\n".join(files)

    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def delete_path(path: str):
    """
    Delete a file or directory from the current project workspace.
    """
    

    try:
        # absolute path of project root
        root = os.getcwd()

        target = os.path.abspath(path)

        # safety check: prevent deleting outside project
        if not target.startswith(root):
            return "Error: You can only delete files inside the project directory."

        if not os.path.exists(target):
            return f"{path} does not exist."

        if os.path.isfile(target):
            os.remove(target)
            return f"File {path} deleted successfully."

        if os.path.isdir(target):
            shutil.rmtree(target)
            return f"Directory {path} deleted successfully."

    except Exception as e:
        return f"Error deleting path: {str(e)}"


llm = init_chat_model(
    model_provider="openai", model="gpt-4o-mini"
)


llm_with_tool = llm.bind_tools(
    tools=[
        run_command,
        create_file, 
        write_file,
        read_file,
        append_file,
        list_directory,
        run_python,
        delete_path,
        ])


def planner(state: State):

    planner_prompt = SystemMessage(content="""
You are a planning agent.

Break the user's request into clear high-level steps.

Rules:
- Return the result in JSON format
- Only return JSON
- Do not execute anything
- Use 3 to 6 steps maximum.
- Avoid very small implementation details.
- Combine related programming tasks into a single step.
- Steps should correspond to meaningful tool actions.
- Never include explanations or conversation in the output.
- Never ask questions inside the plan.
- If information is missing, create a step that asks the user for clarification.
- Prefer steps like:
  - create file
  - implement code
  - modify file
  - run program
  - inspect directory
  - delete file
  - delete directory                                                                   
                                   

Example output format:

{
  "steps": [
    "create factorial.py",
    "implement factorial program in factorial.py",
    "run factorial.py"
  ]
}
""")
    response = llm.invoke([planner_prompt] + state["messages"])

    try:
        plan = json.loads(response.content)
    except:
        plan = {"steps": [response.content]}

    print("\n--- PLAN GENERATED ---")
    print(plan)
    print("----------------------\n")

    return {
        "messages": [response],
        "plan": plan
        }


def executor(state: State):
    last_msg = state["messages"][-1]
    if getattr(last_msg, "type", None) == "tool":
        return {}

    plan = state.get("plan", {})
    steps = plan.get("steps", [])

    if not steps:
        print("\n--- PLAN COMPLETE ---\n")
        return {
            "messages": [{"role": "ai", "content": "done"}]
        }

    next_step = steps.pop(0)

    print("\n--- EXECUTING STEP ---")
    print(next_step)
    print("----------------------\n")

     
    instruction = f"""
    You are executing a step from a multi-step plan.

    Current step:
    {next_step}

IMPORTANT RULES:
- Only perform this step.
- Do NOT complete future steps.
- If the step is to create a file → use create_file to create an empty file.
- If the step is to implement code in an empty file → use write_file.
- If the file already contains code and you need to add more → use append_file.
- if the step is to run the file -> use run_python.
- If modifying an existing file → read it first using read_file.
- Do not implement the entire program unless the step explicitly says so.
"""

    return {
        "messages": [
            {"role": "user", 
            "content": instruction}
        ],
        "plan": {"steps": steps}
    }


def chatbot(state:State):


    system_prompt = SystemMessage(content=               
                """
                You are a voice-controlled coding assistant.
                You help the user by executing commands on their computer using the available tool.

                Rules:
                - Use the appropriate tool for each task:
                - Always work inside a folder called "chat_gpt".
                - If the folder does not exist, create it first using run_command.
                - Never use echo for writing code.
                - Always use create_file before writing to a new file.
                - Use write_file when writing the initial code to a file.
                - Use read_file before modifying existing files.
                - Use append_file when extending code in a file.
                - Use run_python to test programs.
                - Use list_directory to inspect project files.
                - Use delete_path when the user asks to delete or remove a file or folder.
                Examples:

                Create folder:
                run_command(cmd="mkdir chat_gpt")


                Create file:
                create_file(path="chat_gpt/main.py")

                Write code:
                write_file(
                    path="chat_gpt/main.py",
                    content="print('Hello')"
                )

                Install package:
                run_command(cmd="pip install fastapi")
 
                Run python:
                run_python(path="chat_gpt/main.py")
                
                Tool descriptions:

                create_file
                - create a empty file.

                write_file
                - Use this when writing code or multi-line content.
                - Always use this tool when generating Python programs.

                read_file
                - read existing files.

                append_file
                - modify or extend code to an existing file.

                run_python
                - execute python files and see errors

                list_directory
                - explore the project structure

                delete_path
                - delete an existing file or directory.

                Always prefer using the tool instead of explaining.
            """)
    
    message = llm_with_tool.invoke([system_prompt] + state["messages"]) 
        
    assert len(message.tool_calls)<=1
    return {"messages":[message]}
tool_node = ToolNode(
    tools=[
        run_command,
        create_file, 
        write_file,
        read_file,
        run_python,
        list_directory,
        append_file,
        delete_path,
        ])


graph_builder = StateGraph(State)

graph_builder.add_node("planner", planner)
graph_builder.add_node("executor", executor)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",tool_node)


graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "executor")
graph_builder.add_edge("executor", "chatbot")

graph_builder.add_conditional_edges(
      "chatbot",
      tools_condition,
          {
        "tools": "tools",
        "__end__": END
    }
)

graph_builder.add_edge("tools", "chatbot")


def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

