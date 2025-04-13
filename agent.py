import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()


def get_weather(city: str):
    print("🔨 Tool called:get_weather",city)
    url = f"http://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code==200:
        return f"The weather in {city} is {response.text}"
    return "something went wrong"


def run_command(command):
    result = os.system(command=command)
    return result
print(run_command("dir"))


available_tools = {
    "get_weather": {
        "fn":get_weather,
        "description":"Takes a city name as an input and returns the current weather of the city"
    },

    "run_command":{
        "fn":run_command,
        "description":"Takes a command as input to execute on system and returns the output"
    }
}

system_prompt = """

You are an helpful AI assistant who is specialized in resolving user query.
You work on start,plan,action and observed mode.
For the given user query and available tools,plan the step by step execution,based on the planning,select the relevant tool from the available tool and based on the tool selection you perform an action to call the tool.
Wait for the observation and based on the observation from the tool call resolve the user query.

Rules:
- Follow the output JSON format.
- Always perform one step at a time and wait for next input. 
- Carefully analyse the user query.

Output JSON Format:
{{
"step":"string",
"content":"string",
"function":"The name of function if the step is action",
"input":"The input parameter for the funciton",

}}

Available tools:
-get_weather : Takes a city name as an input and returns the current weather of the city.
-run_command : Takes a command as input to execute on system and returns the output.

Example:
User Query: What is the weather of New York?
Output:{{"step":"plan","content":"The user is interested in weather data of New York"}}
Output:{{"step":"plan","content":"From the available tools I should call the get_weather"}}
Output:{{"step":"action","function":"get_weather","input":"New York"}}
Output:{{"step":"observe","output":"14 degree celsius"}}
Output:{{"step":"output","content":"The weather of New York seems to be 14 degree celsius"}}

"""

messages = [
    {"role":"system","content":system_prompt}
]

while True:

  user_query = input(">")
  messages.append({"role":"user","content":user_query})

  while True:

    response = client.chat.completions.create(
    model = "gpt-4o",
    response_format={"type":"json_object"},
    messages= messages
    )  
    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role":"assistant","content":json.dumps(parsed_response)})

    if parsed_response.get("step") =="plan":
        print(f"🧠 {parsed_response.get("content")}")
        continue
    
    if parsed_response.get("step")=="action":
        tool_name = parsed_response.get("function")
        tool_input = parsed_response.get("input")

        if available_tools.get(tool_name,False) != False:
            output = available_tools[tool_name].get("fn")(tool_input)
            messages.append({"role":"assistant","content":json.dumps({"step": "observe", "output": output})})
            continue


    if parsed_response.get("step")=="output":
        print(f"🤖{parsed_response.get("content")}")
        break    

    