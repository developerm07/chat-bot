import streamlit as st
import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# -------------------------------
# Setup model and load tools
# -------------------------------

# Define available tools
get_weather_api = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "str",
                    "description": "The city and state, e.g. San Francisco, New York",
                },
                "unit": {
                    "type": "str",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature to return",
                },
            },
            "required": ["location"],
        },
    },
}

get_price_api = {
    "type": "function",
    "function": {
        "name": "get_price",
        "description": "Get the current price of a product",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "str",
                    "description": "The name of the product, e.g., 'laptop', 'smartphone'"
                },
                "currency": {
                    "type": "str",
                    "enum": ["USD", "EUR", "GBP"],
                    "description": "The currency in which the price should be returned"
                },
            },
            "required": ["object_name"],
        },
    },
}

get_shop_address_api = {
    "type": "function",
    "function": {
        "name": "get_shop_address",
        "description": "Retrieve the address of a specific shop",
        "parameters": {
            "type": "object",
            "properties": {
                "shop_name": {
                    "type": "str",
                    "description": "The name of the shop, e.g., 'Coffee Corner', 'Tech World'"
                },
                "city": {
                    "type": "str",
                    "description": "The city where the shop is located"
                },
            },
            "required": ["shop_name"],
        },
    },
}

global_search_api = {
    "type": "function",
    "function": {
        "name": "global_search",
        "description": "When users query has something related to a question. Fetch general information from Google Search and summarise the results. Use this function when the user asks for general information or the other functions cannot resolve the query see if the query can be search on google.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "str",
                    "description": "The search query to look up on Google."
                }
            },
            "required": ["query"]
        },
    },
}

tools = [get_weather_api, global_search_api, get_price_api, get_shop_address_api]

# Prompts for the model (internal strings, do not change)
TASK_PROMPT = """
You are a helpful assistant.
""".strip()

TOOL_PROMPT = """
# Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_text}
</tools>
""".strip()

FORMAT_PROMPT = """
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""".strip()

def convert_tools(tools):
    return "\n".join([json.dumps(tool) for tool in tools])

def format_prompt(tools):
    tool_text = convert_tools(tools)
    return (
        TASK_PROMPT
        + "\n\n"
        + TOOL_PROMPT.format(tool_text=tool_text)
        + "\n\n"
        + FORMAT_PROMPT
        + "\n"
    )

# -------------------------------
# Load Model and Tokenizer
# -------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    model_dir = Path("arch_function")
    if not model_dir.exists():
        repo_id = 'katanemo/Arch-Function-1.5B'
        model_dir.mkdir(parents=True, exist_ok=True)
        hf_token = "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"
        snapshot_download(repo_id=repo_id, local_dir=str(model_dir), token=hf_token)
    model_dir = "./arch_function"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -------------------------------
# Dummy function implementations
# -------------------------------
def get_weather(arguments):
    return {"name": "get_weather", "results": {"temperature": "62°", "unit": "fahrenheit"}}

def get_price(arguments):
    object_name = arguments.get("object_name", "unknown object")
    currency = arguments.get("currency", "USD")
    return {"name": "get_price", "results": {"object": object_name, "price": "100", "currency": currency}}

def get_shop_address(arguments):
    shop_name = arguments.get("shop_name", "unknown shop")
    city = arguments.get("city", "unknown city")
    return {"name": "get_shop_address", "results": {"shop_name": shop_name, "address": f"123 {shop_name} Ave, {city}"}}

def global_search(arguments):
    summary = "Google is called to search for " + str(arguments)
    return {"name": "global_search", "results": summary}

function_mapping = {
    "get_weather": get_weather,
    "get_price": get_price,
    "get_shop_address": get_shop_address,
    "global_search": global_search,
}

def add_execution_results(messages, execution_results):
    content = "\n".join([f"<tool_response>\n{json.dumps(result)}</tool_response>" for result in execution_results])
    messages.append({"role": "user", "content": content})
    return messages

# -------------------------------
# Main Streamlit App
# -------------------------------
st.title("Function Calling")
st.markdown("This app wraps an LLM with tool calls. Enter your query below.")

# Get user input via Streamlit widget
user_input = st.text_input("User Input:", "")



if user_input:
    system_prompt = format_prompt(tools)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    
    # Generate initial response (expected to output a tool call)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    # Extract tool call
    match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    if match:
        tool_call_json = match.group(1).strip()
        try:
            tool_call = json.loads(tool_call_json)
        except Exception as e:
            st.error("Failed to parse tool call: " + str(e))
            tool_call = None
    else:
        tool_call = None
    
    # Execute tool call if valid
    execution_results = []
    if tool_call and "name" in tool_call:
        func_name = tool_call["name"]
        arguments = tool_call.get("arguments", {})
        if func_name in function_mapping:
            result = function_mapping[func_name](arguments)
            execution_results.append(result)
        else:
            execution_results.append({"name": func_name, "results": "Function not found"})
    else:
        execution_results.append({"error": "No valid tool call detected"})
    
    # Append execution results to messages
    messages = add_execution_results(messages, execution_results)
    
    # Generate final response using updated conversation
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    final_response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    # Display the responses in Streamlit
    # st.markdown("### Initial Model Output (Tool Call)")
    # # Render the initial response in italic
    # st.markdown(f"*{response}*")
    
    # st.markdown("### Execution Results")
    # st.write(execution_results)
    
    st.markdown("### Final Response")
    # Render the final response normally
    st.write(final_response)
