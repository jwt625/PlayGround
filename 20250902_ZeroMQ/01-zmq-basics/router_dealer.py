#!/usr/bin/env python3
"""
ZeroMQ Router-Dealer Pattern Example

This demonstrates the ROUTER-DEALER pattern for advanced routing:
- ROUTER socket can route messages to specific clients
- DEALER socket provides asynchronous REQ-REP-like behavior
- Enables load balancing and complex routing scenarios
- Useful for broker patterns and scalable architectures
"""

import zmq
import time
import random
import sys
import threading
import uuid
from typing import Dict, List


def broker(frontend_port: int = 5559, backend_port: int = 5560) -> None:
    """
    Message broker using ROUTER-DEALER pattern.
    Routes client requests to available workers.
    
    Args:
        frontend_port: Port for client connections (ROUTER)
        backend_port: Port for worker connections (DEALER)
    """
    context = zmq.Context()
    
    # Socket facing clients (ROUTER)
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(f"tcp://*:{frontend_port}")
    
    # Socket facing workers (DEALER)
    backend = context.socket(zmq.DEALER)
    backend.bind(f"tcp://*:{backend_port}")
    
    print(f"Broker started - Frontend: {frontend_port}, Backend: {backend_port}")
    
    try:
        # Simple proxy between frontend and backend
        zmq.proxy(frontend, backend)
    except KeyboardInterrupt:
        print("\nBroker shutting down...")
    finally:
        frontend.close()
        backend.close()
        context.term()


def worker(worker_id: str, broker_port: int = 5560) -> None:
    """
    Worker that connects to broker and processes requests.
    
    Args:
        worker_id: Unique identifier for this worker
        broker_port: Port to connect to broker backend
    """
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://localhost:{broker_port}")
    
    print(f"Worker {worker_id} ready")
    
    try:
        while True:
            # Receive request from broker
            message = socket.recv_multipart()
            
            # Simulate work
            work_time = random.uniform(0.1, 2.0)
            time.sleep(work_time)
            
            # Process the request (echo with worker info)
            if len(message) >= 1:
                request = message[-1].decode('utf-8')
                response = f"Worker {worker_id} processed: {request} (took {work_time:.2f}s)"
                
                # Send response back through broker
                message[-1] = response.encode('utf-8')
                socket.send_multipart(message)
                
                print(f"Worker {worker_id}: Processed '{request}'")
            
    except KeyboardInterrupt:
        print(f"\nWorker {worker_id} shutting down...")
    finally:
        socket.close()
        context.term()


def client(client_id: str, broker_port: int = 5559, num_requests: int = 5) -> None:
    """
    Client that sends requests through broker to workers.
    
    Args:
        client_id: Unique identifier for this client
        broker_port: Port to connect to broker frontend
        num_requests: Number of requests to send
    """
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    
    # Set identity for this client
    socket.setsockopt_string(zmq.IDENTITY, client_id)
    socket.connect(f"tcp://localhost:{broker_port}")
    
    print(f"Client {client_id} connected")
    
    try:
        for i in range(num_requests):
            # Send request
            request = f"Request {i+1} from {client_id}"
            socket.send_string(request)
            print(f"Client {client_id}: Sent '{request}'")
            
            # Receive response
            response = socket.recv_string()
            print(f"Client {client_id}: Received '{response}'")
            
            # Wait between requests
            time.sleep(random.uniform(0.5, 1.5))
            
    except KeyboardInterrupt:
        print(f"\nClient {client_id} shutting down...")
    finally:
        socket.close()
        context.term()


def load_balancing_broker(frontend_port: int = 5559, backend_port: int = 5560) -> None:
    """
    Advanced broker with load balancing and worker tracking.
    
    Args:
        frontend_port: Port for client connections
        backend_port: Port for worker connections
    """
    context = zmq.Context()
    
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(f"tcp://*:{frontend_port}")
    
    backend = context.socket(zmq.ROUTER)
    backend.bind(f"tcp://*:{backend_port}")
    
    # Track available workers
    available_workers: List[bytes] = []
    
    print(f"Load balancing broker started")
    
    poller = zmq.Poller()
    poller.register(backend, zmq.POLLIN)
    
    try:
        while True:
            socks = dict(poller.poll(1000))
            
            # Handle worker messages
            if backend in socks:
                # Get worker identity and message
                worker_id = backend.recv()
                empty = backend.recv()
                
                # Add worker to available list if not already there
                if worker_id not in available_workers:
                    available_workers.append(worker_id)
                    print(f"Worker {worker_id.decode()} registered")
                
                # Check if this is a client response
                try:
                    client_id = backend.recv()
                    if client_id:
                        # Forward response to client
                        empty2 = backend.recv()
                        response = backend.recv()
                        
                        frontend.send_multipart([client_id, b'', response])
                        print(f"Forwarded response from worker {worker_id.decode()} to client {client_id.decode()}")
                except:
                    # Worker just registered, no client response
                    pass
            
            # Handle client requests if workers are available
            if available_workers:
                poller.register(frontend, zmq.POLLIN)
                socks = dict(poller.poll(0))  # Non-blocking
                
                if frontend in socks:
                    # Get client request
                    client_id = frontend.recv()
                    empty = frontend.recv()
                    request = frontend.recv()
                    
                    # Send to next available worker
                    worker_id = available_workers.pop(0)
                    backend.send_multipart([worker_id, b'', client_id, b'', request])
                    
                    print(f"Routed request from client {client_id.decode()} to worker {worker_id.decode()}")
                
                poller.unregister(frontend)
            
    except KeyboardInterrupt:
        print("\nLoad balancing broker shutting down...")
    finally:
        frontend.close()
        backend.close()
        context.term()


def lb_worker(worker_id: str, broker_port: int = 5560) -> None:
    """
    Worker for load balancing broker.
    
    Args:
        worker_id: Unique identifier for this worker
        broker_port: Port to connect to broker
    """
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt_string(zmq.IDENTITY, worker_id)
    socket.connect(f"tcp://localhost:{broker_port}")
    
    # Register with broker
    socket.send_string("")
    
    print(f"LB Worker {worker_id} ready")
    
    try:
        while True:
            # Receive request
            message = socket.recv_multipart()
            
            # Simulate work
            work_time = random.uniform(0.1, 1.0)
            time.sleep(work_time)
            
            # Send response back
            response = f"LB Worker {worker_id} processed request (took {work_time:.2f}s)"
            message[-1] = response.encode('utf-8')
            socket.send_multipart(message)
            
            print(f"LB Worker {worker_id}: Processed request")
            
    except KeyboardInterrupt:
        print(f"\nLB Worker {worker_id} shutting down...")
    finally:
        socket.close()
        context.term()


def demo_router_dealer() -> None:
    """
    Demonstrate ROUTER-DEALER pattern with broker and multiple workers/clients.
    """
    print("Starting ROUTER-DEALER demo...")
    
    # Start broker
    broker_thread = threading.Thread(target=broker)
    broker_thread.daemon = True
    broker_thread.start()
    
    # Give broker time to start
    time.sleep(1)
    
    # Start workers
    worker_threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(f"worker-{i}",))
        thread.daemon = True
        thread.start()
        worker_threads.append(thread)
    
    # Give workers time to connect
    time.sleep(1)
    
    # Start clients
    client_threads = []
    for i in range(2):
        thread = threading.Thread(target=client, args=(f"client-{i}", 5559, 3))
        thread.daemon = True
        thread.start()
        client_threads.append(thread)
    
    # Wait for clients to finish
    for thread in client_threads:
        thread.join()
    
    print("Demo completed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python router_dealer.py [broker|worker|client|lb_broker|lb_worker|demo] [args...]")
        print("  broker [frontend_port] [backend_port]")
        print("  worker [worker_id] [broker_port]")
        print("  client [client_id] [broker_port] [num_requests]")
        print("  lb_broker [frontend_port] [backend_port] - load balancing broker")
        print("  lb_worker [worker_id] [broker_port] - worker for load balancing")
        print("  demo - runs complete demo")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "broker":
        frontend = int(sys.argv[2]) if len(sys.argv) > 2 else 5559
        backend = int(sys.argv[3]) if len(sys.argv) > 3 else 5560
        broker(frontend, backend)
    elif mode == "worker":
        worker_id = sys.argv[2] if len(sys.argv) > 2 else str(uuid.uuid4())[:8]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5560
        worker(worker_id, port)
    elif mode == "client":
        client_id = sys.argv[2] if len(sys.argv) > 2 else str(uuid.uuid4())[:8]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5559
        requests = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        client(client_id, port, requests)
    elif mode == "lb_broker":
        frontend = int(sys.argv[2]) if len(sys.argv) > 2 else 5559
        backend = int(sys.argv[3]) if len(sys.argv) > 3 else 5560
        load_balancing_broker(frontend, backend)
    elif mode == "lb_worker":
        worker_id = sys.argv[2] if len(sys.argv) > 2 else str(uuid.uuid4())[:8]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5560
        lb_worker(worker_id, port)
    elif mode == "demo":
        demo_router_dealer()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
