import requests
import json

def test_llama_model(url='http://127.0.0.1:8080/', 
                     prompt="Who are you?", 
                     max_tokens=100):
    """
    Simple function to test Llama model response
    
    :param url: Model server URL
    :param prompt: Prompt to send to the model
    :param max_tokens: Maximum tokens in response
    :return: Model's response text
    """
    try:
        # Prepare the request payload
        payload = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': 0.7  # Adjust temperature as needed
        }
        
        # Send POST request to the model
        response = requests.post(f'{url}completion', json=payload)
        
        # Check response status
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()
            
            # Extract and return the generated content
            generated_text = response_data.get('content', 'No response generated')
            
            print("Model Response:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)
            
            return generated_text
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response text: {response.text}")
            return None
    
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Run the test
if __name__ == "__main__":
    test_llama_model()