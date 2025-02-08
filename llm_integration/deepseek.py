import subprocess
import requests
import json
from typing import Generator, Union

def generate_with_deepseek(prompt: str,
                           model_name: str = "deepseek-r1:8b",
                           additional_args: list = None,
                           stream: bool = False) -> Union[str, Generator[str, None, None]]:
    """
    Calls the ollama CLI to run the specified model with the given prompt.
    Returns either a string or a generator of strings depending on stream parameter.
    
    :param prompt: The full text prompt to feed into the model.
    :param model_name: The name of the model to run via ollama.
    :param additional_args: Extra arguments to pass to the ollama command.
    :param stream: Whether to return the response as a stream of tokens.
    :return: Generated text response from the model or a generator of response tokens.
    """
    if additional_args is None:
        additional_args = []
        
    # Example command: ollama run -m deepseek-r1:8b --prompt "Your prompt"
    cmd = ["ollama", "run", model_name, prompt]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        text=True
    )
    
    output, error = process.communicate()
    
    if process.returncode != 0:
        print("Error running ollama:", error)
        return ""
    
    if stream:
        url = "http://localhost:11434/api/generate"
        
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }

        try:
            response = requests.post(url, json=data, stream=True)
            
            if response.status_code != 200:
                yield f"Error: Received status code {response.status_code}"
                return

            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line.decode('utf-8'))
                        if json_response.get("response"):
                            yield json_response["response"]
                        if json_response.get("done", False):
                            break
                    except json.JSONDecodeError as e:
                        yield f"\nError decoding JSON: {e}"
                        break
        except requests.exceptions.ConnectionError:
            yield "Error: Could not connect to Ollama. Is it running?"
        except Exception as e:
            yield f"Error: {str(e)}"
    else:
        return output
