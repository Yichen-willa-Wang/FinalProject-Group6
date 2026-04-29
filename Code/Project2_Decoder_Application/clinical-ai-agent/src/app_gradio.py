"""
Gradio Web Interface for Medical AI Chatbot
User-friendly interface for patient Q&A
"""

import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from vector_db_builder import VectorDBBuilder
from rag_agent import HuggingFaceLLM, MedicalRAGAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MedicalChatBot:
    """Medical Chatbot with RAG"""
    
    def __init__(self, 
                 vector_db_path="./data/vector_db",
                 embedding_model="BAAI/bge-base-zh-v1.5",
                 llm_model="Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize the chatbot
        
        Args:
            vector_db_path: Path to vector database
            embedding_model: Embedding model name
            llm_model: LLM model name
        """
        print("Initializing Medical AI Chatbot...")
        
        # Load vector database
        print(f"Loading vector database from: {vector_db_path}")
        builder = VectorDBBuilder(model_name=embedding_model, db_type="chroma")
        self.vectordb = builder.load_vectordb(vector_db_path)
        
        # Initialize LLM
        print(f"Connecting to HuggingFace API...")
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment!")
        
        self.llm = HuggingFaceLLM(
            model_id=llm_model,
            api_token=api_token
        )
        
        # Initialize RAG agent
        print("Initializing RAG agent...")
        self.agent = MedicalRAGAgent(
            vectordb=self.vectordb,
            llm=self.llm,
            top_k=5,
            similarity_threshold=0.5
        )
        
        print("✓ System initialized successfully!")
    
    def chat(self, message, history):
        """
        Handle chat messages
        
        Args:
            message: User message
            history: Chat history (Gradio format)
        
        Returns:
            Updated history
        """
        if not message.strip():
            return history
        
        # Get answer from RAG agent
        try:
            result = self.agent.query(message)
            answer = result['answer']
            sources = result['sources']
            
            # Format sources
            if sources:
                source_text = "\n\n📚 **Reference Sources:**\n"
                for i, source in enumerate(sources[:3], 1):
                    source_text += f"{i}. {source['source']} (Page {source['page']}, {source['content_type']})\n"
                
                full_answer = answer + source_text
            else:
                full_answer = answer
            
            # Add disclaimer
            full_answer += "\n\n⚠️ *Reminder: This information is for reference only. Please consult a professional doctor for medical advice.*"
            
        except Exception as e:
            full_answer = f"Sorry, an error occurred: {str(e)}\nPlease try again or rephrase your question."
        
        # Update history
        history.append((message, full_answer))
        return history
    
    def clear_chat(self):
        """Clear chat history"""
        return []

def create_gradio_interface():
    """Create Gradio interface"""
    
    # Initialize chatbot
    try:
        bot = MedicalChatBot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("\nPlease ensure:")
        print("1. Vector database exists at ~/data/vector_db/")
        print("2. HUGGINGFACE_API_TOKEN is set in .env file")
        raise
    
    # Create interface
    with gr.Blocks(
        title="Medical AI Assistant",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🏥 Clinical Medical AI Assistant
        
        ### Features
        - Intelligent Q&A based on professional medical knowledge base
        - Support for Chinese medical literature retrieval
        - Provides reference sources for verification
        
        ### Usage Tips
        - Please describe your question in detail
        - This system is for reference only - always consult a professional doctor
        """)
        
        chatbot = gr.Chatbot(
            height=500,
            label="Chat Window",
            show_label=True,
            avatar_images=(None, "🤖")
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Example: What are the early symptoms of diabetes?",
                scale=4,
                lines=2
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat")
        
        # Event handlers
        def respond(message, chat_history):
            return bot.chat(message, chat_history), ""
        
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        gr.Markdown("""
        ---
        ⚠️ **Disclaimer**: The information provided by this system is for reference only and cannot replace professional medical advice. 
        If you have health concerns, please seek medical attention promptly.
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False  # Set to True for public URL
    )
