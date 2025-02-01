import subprocess
from llm_integration.deepseek import generate_with_deepseek

def generate_with_deepseek(prompt: str,
                           model_name: str = "deepseek-r1:8b",
                           additional_args: list = None) -> str:
    """
    Calls the ollama CLI to run the specified model with the given prompt.
    Returns the model's text output.
    
    :param prompt: The full text prompt to feed into the model.
    :param model_name: The name of the model to run via ollama.
    :param additional_args: Extra arguments to pass to the ollama command.
    :return: Generated text response from the model.
    """
    if additional_args is None:
        additional_args = []
        
    # Example command: ollama run -m deepseek-r1:8b --prompt "Your prompt"
    cmd = ["ollama", "run", model_name, prompt]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error = process.communicate()
    
    if process.returncode != 0:
        print("Error running ollama:", error)
        return ""
    
    return output
