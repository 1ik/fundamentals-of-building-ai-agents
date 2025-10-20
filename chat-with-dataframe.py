import os
import warnings
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import \
    create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')

load_dotenv()

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(
  "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZNoKMJ9rssJn-QbJ49kOzA/student-mat.csv"
)

print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
print("\n dataset info:")
print(df.info())

llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0, # deterministric output for data naalysis
  api_key=os.getenv("OPENAI_API_KEY")
)

# Create the pandas DataFrame agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,  # Don't print every step
    return_intermediate_steps=True,  # Save steps for inspection
    allow_dangerous_code=True,  # Required for code execution
    max_iterations=10,  # Limit iterations to prevent infinite loops
    agent_type="openai-tools"  # Use the newer agent type
)
 
print("✓ Agent created successfully!")


# --- Section 2: Interacting with Data ---
print("\n" + "="*60)
print("Section 2: Simple Data Queries")
print("="*60)

print("\n--- Task 1: How many rows? ---")
response = agent.invoke("How many rows of data are in this file?")
print(f"Answer: {response['output']}")

# Inspect the code LLM generated
print("\nGenerated code:")

def dump_fn_call_llm_generated(res: Dict[str, Any]):
    if response.get('intermediate_steps'):
        for step in response['intermediate_steps']:
            action = step[0]
            if hasattr(action, 'tool_input'):
                code = action.tool_input
                if isinstance(code, dict) and 'query' in code:
                    print(code['query'])
                elif isinstance(code, str):
                    print(code.replace('; ', '\n'))
    else:
        print("No intermediate steps found")


dump_fn_call_llm_generated(response)

# Verify manually
print(f"\nManual verification: {len(df)} rows")


print("\n--- Task 2: Filter students over 18 ---")
response = agent.invoke("How many students are over 18 years old?")
print(f"Answer: {response['output']}")
dump_fn_call_llm_generated(response)



# Task 3: Statistical query
print("\n--- Task 3: Average final grade ---")
response = agent.invoke("What is the average final grade (G3) of all students?")
print(f"Answer: {response['output']}")
dump_fn_call_llm_generated(response)


print("\n--- Task 4: Alcohol consumption vs grades ---")
response = agent.invoke("Generate scatter plots to examine the correlation between 'Dalc')(daily alcohol) and 'G3', and between 'Walc' (weekend alcohol) and 'G3'.")
print(f"Answer: {response['output']}")
print("\nGenerated code:")
dump_fn_call_llm_generated(response)


print("\n--- Exercise 1: Parental Education Impact ---")
response = agent.invoke("Plot scatter plots to show the relationship between 'Medu' (mother's education level) and 'G3' (final grade), and between 'Fedu' (father's education level) and 'G3'.")
print(f"Answer: {response['output']}")
print("\nGenerated code:")
dump_fn_call_llm_generated(response)


print("\n--- Exercise 2: Internet Access Impact ---")
response = agent.invoke("Use bar plots to compare the average final grades ('G3') of students with internet access at home versus those without ('internet' column).")
print(f"Answer: {response['output']}")
print("\nGenerated code:")
dump_fn_call_llm_generated(response)


# Exercise 3: Absences vs performance
print("\n--- Exercise 3: Absences vs Grades ---")
response = agent.invoke("Plot a scatter plot showing the correlation between the number of absences ('absences') and final grades ('G3') of students.")
print(f"Answer: {response['output']}")
print("\nGenerated code:")
dump_fn_call_llm_generated(response)

