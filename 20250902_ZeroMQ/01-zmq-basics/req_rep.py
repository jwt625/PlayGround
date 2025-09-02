#!/usr/bin/env python3
"""
ZeroMQ Request-Reply Pattern Example

This demonstrates the basic REQ-REP pattern where:
- Client sends requests and waits for replies
- Server receives requests and sends replies
- Communication is synchronous and follows strict alternating pattern
"""

import zmq
import time
import sys
from typing import Optional


def server(port: int = 5555) -> None:
    """
    Simple REQ-REP server that responds to client requests.
    
    Args:
        port: Port to bind the server socket to
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    print(f"Server listening on port {port}")
    
    try:
        while True:
            # Wait for next request from client
            message = socket.recv_string()
            print(f"Received request: {message}")
            
            # Simulate some work
            time.sleep(1)
            
            # Send reply back to client
            reply = f"Echo: {message}"
            socket.send_string(reply)
            print(f"Sent reply: {reply}")
            
    except KeyboardInterrupt:
        print("\nServer shutting down...")
    finally:
        socket.close()
        context.term()


def client(server_host: str = "localhost", server_port: int = 5555, 
           message: str = "Hello") -> Optional[str]:
    """
    Simple REQ-REP client that sends requests to server.
    
    Args:
        server_host: Server hostname or IP
        server_port: Server port
        message: Message to send to server
        
    Returns:
        Reply from server or None if error
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_host}:{server_port}")
    
    try:
        print(f"Sending request: {message}")
        socket.send_string(message)
        
        # Wait for reply
        reply = socket.recv_string()
        print(f"Received reply: {reply}")
        return reply
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python req_rep.py [server|client] [args...]")
        print("  server [port]")
        print("  client [host] [port] [message]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555
        server(port)
    elif mode == "client":
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5555
        message = sys.argv[4] if len(sys.argv) > 4 else "Hello"
        client(host, port, message)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
