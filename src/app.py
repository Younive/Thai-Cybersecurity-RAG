import gradio as gr
from langchain_google_genai import GoogleGenerativeAI
from vectorstore.manage_vectorstore import VectorStoreManager
from prompt_template import build_gemini_rag_prompt
from dotenv import load_dotenv
import os

load_dotenv()

vectorstore = VectorStoreManager().get_exist_cromadb()

model = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY"))

def query_rag(question: str, num_docs: int = 3, language: str = 'Auto-detect'):
    """
    Process RAG query and return formatted response.
    
    Args:
        question: User's question
        num_docs: Number of documents to retrieve (k)
        language: Language preference
        
    Returns:
        answer: Generated answer
        sources: Formatted source information
        debug: Debug information
    """
    if not question.strip():
        return "Please enter a question.", "", ""
    
    try:
        lang_map = {
            "Auto-detect": "auto",
            "English": "en",
            "Thai (ไทย)": "th"
        }
        lang_code = lang_map.get(language, "auto")
        
        results = vectorstore.similarity_search(question, k=num_docs)
        
        if not results:
            return (
                "❌ No relevant documents found. Try rephrasing your question.",
                "No sources retrieved",
                f"Query: {question}\nRetrieved: 0 documents"
            )
        
        prompt = build_gemini_rag_prompt(question, results, language=lang_code)
        
        response = model.invoke(prompt)
        
        sources = "\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'unknown')}"
            for doc in results
        ])
        
        return (
            response,
            sources,
            f"Query: {question}\nRetrieved: {len(results)} documents"
        )
    except Exception as e:
        return (
            f"❌ An error occurred: {str(e)}",
            "No sources retrieved",
            f"Query: {question}\nRetrieved: 0 documents"
        )

with gr.Blocks(title='Cyber RAG', theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Cyber RAG")
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="Enter your question")
            num_docs_slider = gr.Slider(minimum=1, maximum=5, step=1, label="Number of documents to retrieve", value=3)
            submit_btn = gr.Button("🔍 Ask Question", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=2):
                answer_output = gr.Markdown(label="💬 Answer")
                
            with gr.Column(scale=1):
                sources_output = gr.Markdown(label="📚 Sources")
    
        with gr.Accordion("🔧 Debug Information", open=False):
            debug_output = gr.Markdown(label="Debug Info")

        # Event handler
        submit_btn.click(
            fn=query_rag,
            inputs=[question_input, num_docs_slider],
            outputs=[answer_output, sources_output, debug_output]
        )
    
        gr.Markdown(
            """
            ---
            ### 📝 Notes:
            - All answers include citations to source documents
            - Supports both English and Thai queries
            - Uses Google Gemini 2.0 Flash for generation
            - Vector store powered by ChromaDB with Google embeddings
            """
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
