import socket
import json
import logging
import threading
from llm_pipeline import CombinedPipeline
import signal
import sys
import time
import queue
import multiprocessing
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotServer:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.pipeline = None
        self.running = True
        self.clients = []
        self.query_queue = queue.Queue()
        self.response_queues = {}
        self.worker_threads = []
        
        # Determine optimal number of worker threads
        self.num_workers = min(max(2, multiprocessing.cpu_count() - 2), 4)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def initialize_pipeline(self):
        """Initialize the combined pipeline with model loading"""
        logger.info("Loading the model and initializing pipelines...")
        start_time = time.time()
        
        # Set thread count for PyTorch
        torch.set_num_threads(min(8, multiprocessing.cpu_count()))
        
        self.pipeline = CombinedPipeline()
        logger.info(f"Model loaded and pipelines initialized successfully in {time.time() - start_time:.2f} seconds!")

    def process_query_worker(self):
        """Worker thread to process queries from the queue"""
        while self.running:
            try:
                # Get query from queue with timeout
                query_data = self.query_queue.get(timeout=1.0)
                if query_data is None:
                    continue
                
                client_id, query = query_data
                
                # Process the query
                start_time = time.time()
                response = self.pipeline.process_query(query)
                process_time = time.time() - start_time
                
                # Put response in client's response queue
                if client_id in self.response_queues:
                    self.response_queues[client_id].put({
                        'response': response,
                        'process_time': process_time
                    })
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker thread: {str(e)}")
                if client_id in self.response_queues:
                    self.response_queues[client_id].put({
                        'response': "Error processing query",
                        'process_time': 0
                    })

    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        logger.info(f"New client connected from {address}")
        buffer_size = 65536  # Further increased buffer size
        client_id = id(client_socket)
        self.response_queues[client_id] = queue.Queue()
        
        try:
            while self.running:
                try:
                    # Receive data with timeout
                    client_socket.settimeout(1.0)  # 1 second timeout
                    data = client_socket.recv(buffer_size).decode('utf-8')
                    
                    if not data:
                        break
                    
                    # Process query
                    start_time = time.time()
                    query = json.loads(data)['query']
                    logger.info(f"Received query: {query}")
                    
                    # Add query to processing queue
                    self.query_queue.put((client_id, query))
                    
                    # Wait for response
                    try:
                        response_data = self.response_queues[client_id].get(timeout=30.0)  # 30 second timeout
                        client_socket.send(json.dumps(response_data).encode('utf-8'))
                        
                        total_time = time.time() - start_time
                        logger.info(f"Query processed in {total_time:.2f} seconds")
                    except queue.Empty:
                        error_response = {
                            'response': "Query processing timeout",
                            'process_time': 30.0
                        }
                        client_socket.send(json.dumps(error_response).encode('utf-8'))
                    
                except socket.timeout:
                    continue
                except json.JSONDecodeError:
                    continue
                
        except Exception as e:
            logger.error(f"Error handling client {address}: {str(e)}")
        finally:
            logger.info(f"Client {address} disconnected")
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            if client_id in self.response_queues:
                del self.response_queues[client_id]
            client_socket.close()

    def start(self):
        """Start the chatbot server"""
        try:
            # Initialize the pipeline first
            self.initialize_pipeline()
            
            # Start worker threads
            for _ in range(self.num_workers):
                worker = threading.Thread(target=self.process_query_worker)
                worker.daemon = True
                worker.start()
                self.worker_threads.append(worker)
            
            # Create and configure the server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # Increased receive buffer
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)  # Increased send buffer
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)  # Increased pending connections
            
            logger.info(f"Server started on {self.host}:{self.port} with {self.num_workers} workers")
            
            while self.running:
                try:
                    # Accept connections with timeout
                    self.server_socket.settimeout(1.0)
                    client_socket, address = self.server_socket.accept()
                    
                    # Configure client socket
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    self.clients.append(client_socket)
                    
                    # Start a new thread for each client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                    
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            self.handle_shutdown()

    def handle_shutdown(self, signum=None, frame=None):
        """Handle graceful shutdown"""
        logger.info("Shutting down server...")
        self.running = False
        
        # Close all client connections
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        
        # Clear queues
        while not self.query_queue.empty():
            try:
                self.query_queue.get_nowait()
            except:
                pass
        
        for response_queue in self.response_queues.values():
            while not response_queue.empty():
                try:
                    response_queue.get_nowait()
                except:
                    pass
        
        if self.server_socket:
            self.server_socket.close()
        
        sys.exit(0)

def main():
    server = ChatbotServer()
    server.start()

if __name__ == "__main__":
    main() 