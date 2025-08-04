

# My AI Caption Generator Project

## What I Built

I created an AI-powered tool that automatically generates social media captions for internal team communications. This project helped me learn about building AI agents and integrating with language models.

## Requirements

To build this project, I needed:
- A Hugging Face account for deploying the app
- An OpenAI API key for generating captions
- Python 3.8 or higher
- Basic understanding of Python and AI concepts

## How I Built It

### Step 1: Setting Up the Environment
I started by creating a virtual environment and installing the necessary packages:
- smolagents for the agent framework
- gradio for the user interface
- python-dotenv for securely handling API keys

### Step 2: Creating the Caption Generator Tool
I built a tool that can:
- Take inputs like event type, subject, and key details
- Analyze the content to determine the appropriate tone
- Generate a suitable caption

### Step 3: Adding AI-Powered Event Type Detection
I implemented two approaches for determining event types:
- Keyword-based detection as a fallback method
- API-based detection using OpenAI's language model for more accurate results

### Step 4: Implementing Hybrid Caption Generation
I created a system that:
- First tries to generate captions using the OpenAI API
- Falls back to template-based generation if the API call fails
- Customizes the tone based on user preferences

### Step 5: Deploying to Hugging Face Spaces
I deployed my application to Hugging Face Spaces, making it accessible online while keeping my API key secure using environment secrets.

## Results

The final product is a working AI agent that:
- Saves time for HR and communications teams
- Creates consistent, engaging captions
- Adapts to different types of company announcements
- Provides a simple interface for users to generate captions quickly

This project demonstrates my ability to work with AI APIs, build practical tools, and deploy applications to cloud platforms.