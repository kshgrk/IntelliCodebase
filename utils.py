import subprocess
import re
import os
import shlex
from codeanalysis import CodebaseAnalyzer, Issue

def validate_parameter(parameter_name, value, validation_rule):
    """Validates a parameter value against a given rule."""
    if not re.match(validation_rule, value):
        raise ValueError(f"Invalid value for parameter '{parameter_name}': {value}")

def create_file(parameters):
    """Creates a file with the given filename, overwriting if it exists."""
    filename = parameters["filename"]
    content = parameters.get("content", "")
    filepath = os.path.join(os.path.expanduser("~/chatbot_workspace"), filename)
    try:
        with open(filepath, "w") as f:
            f.write(unescape_string(content)) # Unescape the content
        return f"File '{filename}' created successfully."
    except Exception as e:
        raise RuntimeError(f"Error creating file: {e}")

def analyze_codebase(parameters, model):  # Add model as a parameter
    """Analyzes a codebase for issues using CodebaseAnalyzer.
    Optionally analyzes a single file if filename is provided.
    """
    base_path = parameters["base_path"]
    filename = parameters.get("filename")

    analyzer = CodebaseAnalyzer(base_path)

    if filename:
        # Analyze a specific file
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            chunks = analyzer.get_file_chunks(file_path)
            for chunk in chunks:
                issues = analyzer.analyze_chunk(chunk, model=model)  # Pass model to analyze_chunk
                if issues:
                    analyzer.issues_db.setdefault(file_path, []).extend([
                        {"issue": issue, "cached_content_name": analyzer.cache_mapping.get(file_path)}
                        for issue in issues
                    ])
        else:
            return f"File not found: {file_path}"
    else:
        # Analyze all files in the base_path
        print('no filename')
        analyzer.process_codebase(model=model)  # Pass model to process_codebase

    response_parts = []
    for file_path, issues_data in analyzer.issues_db.items():
        response_parts.append(f"Issues in {file_path}:")
        for issue_data in issues_data:
            issue = issue_data["issue"]
            response_parts.append(f"  - {issue.description}")
            if issue.fix_suggestion:
                response_parts.append(f"    Fix: {issue.fix_suggestion}")
            response_parts.append(f"    Priority: {issue.priority}")

    return "\n".join(response_parts) if response_parts else "No issues found."

def read_file(parameters):
    """Reads the content of a file."""
    filename = parameters["filename"]
    filepath = os.path.join(os.path.expanduser("~/chatbot_workspace"), filename)
    try:
        with open(filepath, "r") as f:
            content = f.read()
        return f"Content of {filename}:\n```\n{content}\n```"
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

def append_to_file(parameters):
    """Appends the given content to a file."""
    filename = parameters["filename"]
    content = parameters["content"]
    filepath = os.path.join(os.path.expanduser("~/chatbot_workspace"), filename)
    try:
        with open(filepath, "a") as f:
            f.write(unescape_string(content)) # Unescape the content
        return f"Content appended to '{filename}' successfully."
    except Exception as e:
        raise RuntimeError(f"Error appending to file: {e}")

def overwrite_file(parameters):
    """Overwrites the given content to a file."""
    filename = parameters["filename"]
    content = parameters["content"]
    filepath = os.path.join(os.path.expanduser("~/chatbot_workspace"), filename)
    try:
        with open(filepath, "w") as f:
            f.write(unescape_string(content)) # Unescape the content
        return f"Content overwritten to '{filename}' successfully."
    except Exception as e:
        raise RuntimeError(f"Error overwriting to file: {e}")

def unescape_string(content):
    """
    Unescapes special characters in the content.
    """
    return content.encode().decode('unicode_escape')

def execute_command(intent, parameters):
    """
    Executes a command based on the intent and parameters.
    """
    from app import COMMAND_WHITELIST

    if intent not in COMMAND_WHITELIST["commands"]:
        raise ValueError(f"Invalid intent: {intent}")

    command_data = COMMAND_WHITELIST["commands"][intent]

    # Check if a Python function is defined for this intent
    if "function" in command_data:
        function_name = command_data["function"]
        if function_name == "create_file":
            return create_file(parameters)
        elif function_name == "append_to_file":
            return append_to_file(parameters)
        elif function_name == "overwrite_file":
            return overwrite_file(parameters)
        else:
            raise ValueError(f"Function '{function_name}' not implemented.")

    # Otherwise, it's a Linux command
    command_template = command_data["command"]

    command_args = []
    for param_data in command_data.get("parameters", []):
        param_name = param_data["name"]
        if param_name not in parameters:
            raise ValueError(f"Missing parameter: {param_name}")
        param_value = parameters[param_name]
        validate_parameter(param_name, param_value, param_data["validation"])

        command_args.append(param_value)
    
    # Use shlex.split() to correctly split command
    full_command = shlex.split(command_template) + command_args

    try:
        process = subprocess.run(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=os.path.expanduser("~/chatbot_workspace")
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command execution failed: {e.stderr}")
