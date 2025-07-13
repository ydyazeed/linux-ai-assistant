from typing import List, Optional, Dict
import logging
from ctransformers import AutoModelForCausalLM
from rag_pipeline import RAGPipeline, RetrievedContext
from dataclasses import dataclass
from typing import List
import time
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    user_query: str
    assistant_response: str

class LLMPipeline:
    def __init__(
        self,
        model_path: str = "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        max_new_tokens: int = 96,  # Further reduced
        temperature: float = 0.1,
        context_window: int = 1024,  # Further reduced
        max_history_chars: int = 256,  # Further reduced
        gpu_layers: int = 0,
    ):
        """Initialize the LLM Pipeline with Mistral 7B model."""
        logger.info(f"Initializing LLM Pipeline with model: {model_path}")
        try:
            start_time = time.time()
            
            # Determine optimal thread count based on CPU cores
            cpu_count = torch.multiprocessing.cpu_count()
            optimal_threads = min(max(4, cpu_count - 2), 32)  # Leave some cores for system
            
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                context_length=context_window,
                gpu_layers=gpu_layers,
                batch_size=32,      # Further increased for better throughput
                threads=optimal_threads,  # Dynamic thread allocation
                stream=False,       # Disable streaming for faster response
                reset=False,        # Don't reset model between calls
                top_k=10,          # Reduce sampling space
                top_p=0.3,         # More focused sampling
                repetition_penalty=1.1,  # Slight penalty for repetition
            )
            self.max_history_chars = max_history_chars
            self.conversation_history: List[ConversationTurn] = []
            logger.info(f"LLM model loaded successfully in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}", exc_info=True)
            raise

    def _format_context(self, contexts: List[RetrievedContext]) -> str:
        """Format retrieved contexts into a single string."""
        formatted_contexts = []
        for ctx in contexts[:1]:  # Limit to top 1 context for faster processing
            formatted_contexts.append(
                f"Source: {ctx.source}\n"
                f"Content: {ctx.text[:300]}\n"  # Limit context length
            )
        return "\n".join(formatted_contexts)

    def _format_conversation_history(self) -> str:
        """Format the conversation history into a string."""
        if not self.conversation_history:
            return ""
        
        # Only keep the last conversation turn
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            return f"\nPrevious: {last_turn.user_query[:50]}\n"  # Limit history length
        return ""

    def _update_conversation_history(self, query: str, response: str):
        """Update conversation history while maintaining the character limit."""
        # Keep only the last turn
        new_turn = ConversationTurn(user_query=query, assistant_response=response)
        self.conversation_history = [new_turn]

    def _create_prompt(self, query: str, contexts: List[RetrievedContext]) -> str:
        """Create a prompt for the LLM using the query and retrieved contexts."""
        context_text = self._format_context(contexts)
        history_text = self._format_conversation_history()
        
        # Minimal prompt
        return f"<s>[INST] Linux expert. Context:\n{context_text}\n{history_text}Q: {query}[/INST]"

    def generate_response(
        self,
        query: str,
        contexts: List[RetrievedContext],
    ) -> str:
        """Generate a response using the LLM."""
        try:
            start_time = time.time()
            prompt = self._create_prompt(query, contexts)
            response = str(self.llm(prompt)).strip()
            self._update_conversation_history(query, response)
            logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating the response. Please try again."

class CombinedPipeline:
    def __init__(
        self,
        rag_data_dir: str = "data",
        llm_model_path: str = "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    ):
        """Initialize both RAG and LLM pipelines."""
        logger.info("Initializing Combined Pipeline")
        start_time = time.time()
        self.rag = RAGPipeline(data_dir=rag_data_dir)
        self.llm = LLMPipeline(model_path=llm_model_path)
        logger.info(f"Combined Pipeline initialized in {time.time() - start_time:.2f} seconds")

    def process_query(
        self,
        query: str,
        top_k: int = 1,  # Reduced to 1 for faster processing
        quality_threshold: float = 1.5,
    ) -> str:
        """Process a query through both RAG and LLM pipelines."""
        start_time = time.time()
        
        # Get relevant contexts from RAG pipeline
        rag_start = time.time()
        contexts = self.rag.retrieve(query, top_k, quality_threshold)
        logger.info(f"RAG retrieval completed in {time.time() - rag_start:.2f} seconds")
        
        # Generate response using LLM
        llm_start = time.time()
        response = self.llm.generate_response(query, contexts)
        logger.info(f"LLM generation completed in {time.time() - llm_start:.2f} seconds")
        
        logger.info(f"Total query processing time: {time.time() - start_time:.2f} seconds")
        return response

def main():
    # Example usage
    pipeline = CombinedPipeline()
    
    test_queries = [
        "How do I check disk space usage?",
        "What is the command to create a new directory?",
        "How to fix permission denied errors?",
        "Show me how to use the ls command",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = pipeline.process_query(query)
        print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main() 