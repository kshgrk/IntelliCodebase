from flask import Flask, request, jsonify
import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
import utils
import os
import yaml
from dotenv import load_dotenv

# --- Load Environment Variables and Configuration ---
load_dotenv()
app = Flask(__name__)

# --- Vertex AI Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("REGION")
MODEL_NAME = "gemini-2.0-flash-exp"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- Function Declaration for Linux Command Execution ---
# Load command whitelist from commands.yaml
with open("commands.yaml", "r") as f:
    COMMAND_WHITELIST = yaml.safe_load(f)

# Create a function declaration for each command in the whitelist
function_declarations = []
for intent, details in COMMAND_WHITELIST["commands"].items():
    # Determine the description based on whether it's a function or command
    if "function" in details:
        description = details.get("description", f"Performs the action: {intent}")
    else:
        description = details.get(
            "description", f"Executes the Linux command: {details['command']}"
        )

    function_declarations.append(
        FunctionDeclaration(
            name=intent,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    param_data["name"]: {
                        "type": "string",
                        "description": param_data["name"],
                    }
                    for param_data in details.get("parameters", [])
                },
                "required": [
                    param_data["name"] for param_data in details.get("parameters", [])
                ],
            },
        )
    )

# Create the Tool object
linux_command_tool = Tool(function_declarations=function_declarations)

# --- Model Initialization ---
model = GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=GenerationConfig(
        temperature=0.9, top_p=0.95, top_k=40, candidate_count=1, max_output_tokens=2048
    ),
    tools=[linux_command_tool],
)

# --- Chat Session ---
chat = model.start_chat()

# --- API Endpoint ---
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.get_json()
        user_input = data.get("message", "")

        # Initialize text_parts here
        text_parts = []

        # Basic parsing
        analyze_command = "analyze the codebase"
        if analyze_command in user_input.lower():
            # Handling for "analyze the codebase"
            parts = user_input.lower().split(analyze_command)
            if len(parts) > 1:
                path_part = parts[1].strip()

                if " for " in path_part:
                    path, issue = path_part.split(" for ", 1)
                    issue = issue.strip()
                else:
                    path = path_part
                    issue = None

                base_path = path.strip()
                function_args = {"base_path": base_path, "issue": issue}
                command_output = utils.analyze_codebase(function_args, model=chat)

                # Send function response back to model
                function_response = Part.from_function_response(
                    name="analyze_codebase",
                    response={"content": command_output}
                )
                followup_response = chat.send_message(function_response)

                # Add function output to response
                if followup_response.candidates:
                    followup_text = followup_response.candidates[0].content.parts[0].text
                    text_parts.append(f"Command output: {command_output}\n{followup_text}")

        elif "analyze the file" in user_input.lower():
            # Handling for "analyze the file"
            parts = user_input.lower().split("analyze the file")
            if len(parts) > 1:
                path_part = parts[1].strip()

                if " for " in path_part:
                    path, issue = path_part.split(" for ", 1)
                    issue = issue.strip()
                else:
                    path = path_part
                    issue = None

                filename = os.path.basename(path)
                base_path = os.path.dirname(path)

                function_args = {"base_path": base_path, "filename": filename, "issue": issue}
                command_output = utils.analyze_codebase(function_args, model=chat)

                # Send function response back to model
                function_response = Part.from_function_response(
                    name="analyze_codebase",
                    response={"content": command_output}
                )
                followup_response = chat.send_message(function_response)

                # Add function output to response
                if followup_response.candidates:
                    followup_text = followup_response.candidates[0].content.parts[0].text
                    text_parts.append(f"Command output: {command_output}\n{followup_text}")

        else:
            # Send Message to Gemini (non-analyze command)
            response = chat.send_message(user_input)
            print(f"Initial response: {response.candidates[0] if response.candidates else 'No candidates'}")

            if not response.candidates:
                return jsonify({"response": "No response from the model."})

            candidate = response.candidates[0]
            
            # Handle different response types
            if candidate.content and candidate.content.parts:
                # Get all text parts and combine them
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    elif hasattr(part, 'function_call'):
                        # Handle function call
                        function_name = part.function_call.name
                        function_args = {k: v for k, v in part.function_call.args.items()}
                            
                        # Call functions from utils.py, passing the model
                        if function_name == "analyze_codebase":
                            command_output = utils.analyze_codebase(function_args, model=chat)
                        elif function_name == "read_file":
                            command_output = utils.read_file(function_args)
                        else:
                            command_output = utils.execute_command(function_name, function_args)
                            
                        # Send function response back to model
                        function_response = Part.from_function_response(
                            name=function_name,
                            response={"content": command_output}
                        )
                        followup_response = chat.send_message(function_response)

                        # Add function output to response
                        if followup_response.candidates:
                            followup_text = followup_response.candidates[0].content.parts[0].text
                            text_parts.append(f"Command output: {command_output}\n{followup_text}")

        # Combine all responses
        if text_parts:
            return jsonify({"response": "\n".join(text_parts)})

        return jsonify({"response": "No clear response from the model."})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Create the chatbot workspace directory if it doesn't exist
    os.makedirs(os.path.expanduser("~/chatbot_workspace"), exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
