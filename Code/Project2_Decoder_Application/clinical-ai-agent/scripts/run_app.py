#!/usr/bin/env python3
"""
Launch the Medical AI Chatbot Web Interface
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API token
if not os.getenv("HUGGINGFACE_API_TOKEN"):
    print("="*60)
    print("ERROR: HUGGINGFACE_API_TOKEN not found!")
    print("="*60)
    print("\nPlease follow these steps:")
    print("1. Get your token from: https://huggingface.co/settings/tokens")
    print("2. Create a .env file in the project root")
    print("3. Add: HUGGINGFACE_API_TOKEN=your_token_here")
    print("\nOr set it in your environment:")
    print("export HUGGINGFACE_API_TOKEN=your_token_here")
    print("="*60)
    sys.exit(1)

# Import and run the app
from app_gradio import create_gradio_interface

if __name__ == "__main__":
    print("="*60)
    print("Starting Medical AI Chatbot")
    print("="*60)
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
