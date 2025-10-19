from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI(
   model="gpt-4o-mini",
   api_key=os.getenv("OPENAI_API_KEY")
)

@tool
def add(a: int, b: int) -> int:
  """
  Add a and b.
  Args:
    a (int): first integer to be added
    b (int): second integer to be added
  Returns:
    int: sum of a and b
  """
  return a + b


@tool
def subtract(a: int, b:int) -> int:
  """
  Subtracts b from a.
  """
  return a - b

@tool
def multiply(a: int, b:int) -> int:
  """
  Multiply a and b.
  """
  return a * b

tool_map = {
  "add": add,
  "subtract": subtract,
  "multiply": multiply
}

print("\n -- Testing Tools --")
test_input = {"a": 10, "b": 3}
print(f"add(10,3) = {tool_map['add'].invoke(test_input)}")
print(f"subtract(10,3) = {tool_map['subtract'].invoke(test_input)}")
print(f"multiply(10,3) = {tool_map['multiply'].invoke(test_input)}")

tools = [add,subtract,multiply]
llm_with_tools = llm.bind_tools(tools)

print("\n✓ Alls tools bound to LLM successfully")

#------
#  Section 3: Making the LLM Actually Use Your Tools.
#------

print("\n--- Complete Tool Calling Interaction ---")

query = "what is 3 + 2"


chat_history = [HumanMessage(content=query)]
response1 = llm_with_tools.invoke(chat_history)
print(f"User: {query}")

# Step 2: LLM decides what to do
chat_history.append(response1)

print(f"\nLLM Response Type: {type(response1).__name__}")

tool_calls = response1.tool_calls
if tool_calls:
  tc = tool_calls[0]
  tc_name = tc["name"]
  tc_args = tc["args"]
  tc_id = tc["id"]

  print(f"\n🔧 Tool Call Details:")
  print(f"  Tool Name: {tc_name}")
  print(f"  Tool Args: {tc_args}")
  print(f"  Call ID: {tc_id}")
  tool_result = tool_map[tc_name].invoke(tc_args)
  print(f"Tool Result {tool_result}")

  tool_msg = ToolMessage(
    content=str(tool_result),
    tool_call_id=tc_id
  )

  chat_history.append(tool_msg)
  final_response = llm_with_tools.invoke(chat_history)
  print(f"\n Final ans: {final_response.content}")
else:
  print("No tool class - LLM Answered directly")


class ToolCallingAgent:
  def __init__(self, llm, tools, tool_map):
    """
      Initialize the agent with an LLM and tools.

      Args:
        llm: The language model
        tools: List of tool functions
        tool_map: Dictionary mapping toolname to functions
    """
    self.llm_with_tools = llm.bind_tools(tools)
    self.tool_map = tool_map
  
  def run(self, query: str) -> str:
    """
    Run the agent on a query.
    
    Args:
        query (str): User's question
        
    Returns:
        str: Final answer from the LLM
    """
    # Step 1: Create initial chat history with user message
    chat_history = [HumanMessage(content=query)]

    # Step 2: Get LLM response
    response = self.llm_with_tools.invoke(chat_history)

    # Step 3: Check if LLM wants to use a tool
    if not response.tool_calls:
        # No tool needed - return direct answer
        return response.content

    # Step 4: Parse the tool call
    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_call_id = tool_call["id"]

    # Step 5: Execute the tool
    tool_result = self.tool_map[tool_name].invoke(tool_args)

    # Step 6: Create tool message and update history
    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call_id
    )
    chat_history.extend([response, tool_message])

    # Step 7: Get final answer from LLM
    final_response = self.llm_with_tools.invoke(chat_history)
    return final_response.content



# Create the agent
my_agent = ToolCallingAgent(llm, tools, tool_map)
test_queries = [
  "What is 5 plus 3?",
  "Calculate 10 minus 7",
  "What is 4 times 6?",
  "What is the capital of France?"  # Should answer directly  without tools
]

for query in test_queries:
  result = my_agent.run(query)
  print(f"✅ answer: {result}")


#--
# Exercise 1: Create the calculate_tip Tool
#--
@tool
def calculate_tip(total_bill: int, tip_percent: int) -> int:
  """
  Returns the tip amount of a given bill

  Args:
    total_bill (int): Total amount of bill
    tip_percent (int): percent amount on the bill to calculate the tip
    id: tool caller id from llm
  Returns:
    int: the caluclated tip
  """
  tip = total_bill * tip_percent * .01
  return tip



# Test the tool directly
test_tip_input = {"total_bill": 120, "tip_percent": 15}
result = calculate_tip.invoke(test_tip_input)
print(f"Test: 15% tip on $120 = ${result}")



query = "How much should I tip on $60 bill at 20%?"
calc_tip_llm = llm.bind_tools([calculate_tip])

chat_history = [HumanMessage(content=query)]

response = calc_tip_llm.invoke(chat_history)
print(f"\n tip answer = {response.content}")


# i wanna see the full structure of response  pretty print. or even better if i could stop a debugger here
print(f"{response}")


if response.tool_calls:
  tc = response.tool_calls[0]
  t_n = tc["name"]
  t_args = tc["args"]
  t_id = tc["id"]

  print(f"Tool to call: {t_n}")
  print(f"Arguments: {t_args}")

  # call the tool
  tool_res = calculate_tip.invoke(t_args)
  
  tool_msg = ToolMessage(
    content=str(tool_res),
    tool_call_id=t_id
  )
  chat_history.extend([response, tool_msg])
  final_answer = calc_tip_llm.invoke(chat_history)
  print(f"\nfinal answer : {final_answer.content}")
else:
  print("no tool call made")

