# LLM Course Notes

My implementations from the Coursera LLM course. Writing actual code instead of just running notebooks.

## Projects

### 1. DataWizard (`datawizard.py`)
LLM-powered data analysis tool. Done.

### 2. Interactive Tool-Calling Agent (`interactive-tool-calling.py`)
Built an agent that lets LLMs call Python functions dynamically.

**What I learned:**
- The `@tool` decorator makes functions callable by LLMs
- Chat history keeps context (LLMs are stateless)
- Flow: User asks → LLM picks tool → I run it → LLM answers
- Built arithmetic tools (add, subtract, multiply) and a tip calculator

**Key insight:** The LLM doesn't run tools, it just tells you what to run. You execute them and send results back.

**Messages:**
- `HumanMessage` = user input
- `AIMessage` = LLM response
- `ToolMessage` = tool result

Created two agents:
1. `ToolCallingAgent` - general purpose
2. `TipAgent` - calculates tips from natural language

### 3. Chat with DataFrame (`chat-with-dataframe.py`)
Built a data analysis agent that writes pandas/matplotlib code from natural language.

**What I learned:**
- `create_pandas_dataframe_agent` generates code dynamically (doesn't use predefined tools)
- Flow: Ask question → LLM writes Python code → Code executes → Get result + visualization
- Can inspect generated code via `intermediate_steps`

**Key difference from tool-calling:**
- Tool-calling: LLM picks from predefined functions
- DataFrame agent: LLM writes new code for each question

**Exercises completed:**
- Parental education vs grades
- Internet access impact on performance
- Absences correlation with grades

Dataset: Student alcohol consumption (395 students, 33 columns)

## Setup

```bash
# Install dependencies
pip install langchain==0.3.25 langchain-openai==0.3.19 python-dotenv
pip install langchain-experimental matplotlib seaborn tabulate

# Add your API key to .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Run
python3 interactive-tool-calling.py
python3 chat-with-dataframe.py
```

## Notes

Using OpenAI's gpt-4o-mini. Skipped Watsonx (confusing UI).

Writing code myself instead of copy-paste from notebooks = way better learning.
