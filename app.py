from smolagents import CodeAgent, DuckDuckGoSearchTool, load_tool, tool
from smolagents.models import OpenAIModel
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

import os 

from dotenv import load_dotenv
load_dotenv() #loads the environment variables from the .env file




def predict_event_type_with_api(subject, key_details):
    """Predicts the event type using an API call to a language model
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Warning-api key not found")
            return "announcement" #default if no api key
        #tells the code exactly where to send the request
        api_url = "https://api.openai.com/v1/completions"
        #proves only users with api key can access the request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        #headers will ensuee that openai knows its you making the req
        #knows youre sending json data in req body
        #req will be processed correctly
        prompt = f"""
        Classify the following social media post into one of these categories:
        - achievement (for accomplishments, awards, milestones)
        - announcement (for news, updates, information)
        - event (for meetings, webinars, gatherings)
        
        Subject: {subject}
        Details: {key_details}
        
        Category:
        """

        #creating a request body for the model to use
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "max_tokens": 10
        }
        
        response = requests.post(api_url, headers=headers, json=data)
        result = response.json()
        predicted_type = result["choices"][0]["text"].strip().lower()
        
        # Validate the response
        valid_types = ["achievement", "announcement", "event"]
        if predicted_type in valid_types:
            return predicted_type
        else:
            return "announcement"  # Default
            
    except Exception as e:
        print(f"Error predicting event type: {e}")
        return "announcement"  # Default on error

def predict_event_type_with_keywords(subject, key_details):
    """Fallback method to predict event type using keywords."""
    # Keywords that might indicate different event types
    achievement_keywords = ["achieved", "won", "milestone", "success", "award", "accomplishment"]
    event_keywords = ["happening", "join", "upcoming", "schedule", "attend", "meeting", "webinar"]
    
    combined_text = (subject + " " + key_details).lower()
    
    # Simple keyword-based classification
    for keyword in achievement_keywords:
        if keyword in combined_text:
            return "achievement"
    
    for keyword in event_keywords:
        if keyword in combined_text:
            return "event"
    
    # Default to announcement
    return "announcement"
    
def template_based_caption(event_type, subject, key_details, tone):
    """Generate a caption using pre-defined templates."""
    # Template library
    templates = {
        "achievement": [
            "🏆 Celebrating a major milestone! Our team has achieved {subject}. {key_details}. #TeamSuccess",
            "✨ Proud to announce that we've reached {subject}! {key_details}. #Achievement #TeamWork"
        ],
        "announcement": [
            "📢 Important update: {subject}. {key_details}. Stay tuned for more information.",
            "🔔 Attention team! {subject} is now official. {key_details}. #CompanyNews"
        ],
        "event": [
            "📆 Join us for {subject}! {key_details}. We look forward to seeing everyone there!",
            "🎉 It's happening! {subject} is coming up. {key_details}. Don't miss out!"
        ]
    }
    
    # Handle unknown event types
    if event_type.lower() not in templates:
        print(f"Unknown event type: {event_type}. Using announcement template.")
        event_type = "announcement"
    
    import random
    template = random.choice(templates[event_type.lower()])
    caption = template.format(subject=subject, key_details=key_details)
    
    # Adjust tone if needed
    if tone.lower() == "formal":
        caption = caption.replace("!", ".").replace("🏆", "").replace("✨", "").replace("📢", "").replace("🔔", "").replace("📆", "").replace("🎉", "")
    elif tone.lower() == "celebratory":
        caption = caption + " 🎊 🎉 Let's celebrate this amazing moment together!"
    elif tone.lower() == "motivational":
        caption = caption + " 💪 Together we can achieve even more! #Motivation"
    
    return caption

def generate_caption_with_llm(event_type, subject, key_details, tone):
    """Generate a complete caption using a language model API call."""
    try:
        # Get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        api_url = "https://api.openai.com/v1/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Generate a social media caption for an internal company post with these details:
        - Type: {event_type} (achievement, announcement, or event)
        - Subject: {subject}
        - Key details: {key_details}
        - Tone: {tone} (professional, formal, celebratory, or motivational)
        
        The caption should be engaging, include relevant hashtags, and be appropriate for internal company communication.
        Keep it concise (2-3 sentences maximum) and include appropriate emojis if the tone allows.
        """
        
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        response = requests.post(api_url, headers=headers, json=data)
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["text"].strip()
        else:
            raise ValueError(f"Unexpected API response: {result}")
    except Exception as e:
        print(f"LLM caption generation failed: {e}")
        raise e  # Re-raise to be caught by the main function

# Below is an example of a tool that generates a caption. 
@tool
def generate_caption(event_type:str, subject:str, key_details:str, tone:str="professional")-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """Generates a social media caption for internal team posts.
    Args:
        event_type: Type of post ('achievement','announcement','event', or 'auto' for AI prediction)
        subject: The main topic or subject of the post
        key_details: Important information to include in the caption
        tone: Desired emotional tone (default: professional, options: formal, celebratory, motivational)
    """
    if event_type.lower() == "auto":
        try:
            event_type = predict_event_type_with_api(subject, key_details)
            print(f"AI predicted event type: {event_type}")
        except Exception as e:
            print(f"API prediction failed: {e}. Using Keyword fallback.")
            event_type = predict_event_type_with_keywords(subject, key_details)

    try:
        return generate_caption_with_llm(event_type, subject, key_details, tone)
    except Exception as e:
        print(f"Falling back to template-based caption generation: {e}")
        return template_based_caption(event_type, subject, key_details, tone)          
  

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = OpenAIModel(
max_tokens=2096,
temperature=0.5,
model_id='gpt-3.5-turbo', # it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
# Model is defined above

agent = CodeAgent(
    model=model,
    tools=[final_answer, generate_caption, get_current_time_in_timezone, image_generation_tool], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()