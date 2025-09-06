#!/usr/bin/env python3
"""
ZeroMQ Push-Pull Pattern Example

This demonstrates the PUSH-PULL pattern for work distribution:
- Ventilator pushes work to workers
- Workers pull work and process it
- Sink collects results from workers
- Load balancing is automatic
"""

import zmq
import time
import random
import sys
import threading
from typing import List


def ventilator(num_tasks: int = 100, worker_port: int = 5557) -> None:
    """
    Ventilator that distributes work to workers using PUSH socket.
    
    Args:
        num_tasks: Number of work items to distribute
        worker_port: Port where workers are listening
    """
    context = zmq.Context()
    
    # Socket to send messages to workers
    sender = context.socket(zmq.PUSH)
    sender.bind(f"tcp://*:{worker_port}")
    
    # Socket to signal start of batch to sink
    sink = context.socket(zmq.PUSH)
    sink.connect("tcp://localhost:5558")
    
    print(f"Distributing {num_tasks} tasks to workers...")
    input("Press Enter when workers are ready: ")
    
    # Signal start of batch
    sink.send_string("0")
    
    # Send tasks to workers
    total_msec = 0
    for task_nbr in range(num_tasks):
        # Random workload from 1 to 100 msecs
        workload = random.randint(1, 100)
        total_msec += workload
        
        sender.send_string(str(workload))
    
    print(f"Total expected cost: {total_msec} msec")
    
    sender.close()
    sink.close()
    context.term()


def worker(worker_id: int, ventilator_port: int = 5557, sink_port: int = 5558) -> None:
    """
    Worker that pulls work from ventilator and sends results to sink.
    
    Args:
        worker_id: Unique identifier for this worker
        ventilator_port: Port to connect to ventilator
        sink_port: Port to connect to sink
    """
    context = zmq.Context()
    
    # Socket to receive messages from ventilator
    receiver = context.socket(zmq.PULL)
    receiver.connect(f"tcp://localhost:{ventilator_port}")
    
    # Socket to send messages to sink
    sender = context.socket(zmq.PUSH)
    sender.connect(f"tcp://localhost:{sink_port}")
    
    print(f"Worker {worker_id} ready")
    
    try:
        while True:
            # Receive work from ventilator
            work = receiver.recv_string()
            
            # Simulate work
            workload = int(work)
            time.sleep(workload / 1000.0)  # Convert to seconds
            
            # Send result to sink
            result = f"Worker {worker_id} completed task {work}ms"
            sender.send_string(result)
            print(f"Worker {worker_id}: {work}ms")
            
    except KeyboardInterrupt:
        print(f"\nWorker {worker_id} shutting down...")
    finally:
        receiver.close()
        sender.close()
        context.term()


def sink(expected_tasks: int = 100, sink_port: int = 5558) -> None:
    """
    Sink that collects results from workers using PULL socket.
    
    Args:
        expected_tasks: Number of tasks to expect
        sink_port: Port to bind sink socket to
    """
    context = zmq.Context()
    
    # Socket to receive messages from workers
    receiver = context.socket(zmq.PULL)
    receiver.bind(f"tcp://*:{sink_port}")
    
    # Wait for start of batch
    s = receiver.recv_string()
    
    # Start timing
    tstart = time.time()
    
    # Process results from workers
    for task_nbr in range(expected_tasks):
        result = receiver.recv_string()
        if task_nbr % 10 == 0:
            print(f"Processed {task_nbr} tasks")
    
    # Calculate and report duration
    tend = time.time()
    print(f"Total elapsed time: {(tend - tstart) * 1000:.0f} msec")
    
    receiver.close()
    context.term()


def run_workers(num_workers: int = 3) -> List[threading.Thread]:
    """
    Start multiple worker threads.
    
    Args:
        num_workers: Number of worker threads to start
        
    Returns:
        List of worker threads
    """
    threads = []
    for i in range(num_workers):
        thread = threading.Thread(target=worker, args=(i,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    return threads


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python push_pull.py [ventilator|worker|sink|demo] [args...]")
        print("  ventilator [num_tasks]")
        print("  worker [worker_id]")
        print("  sink [expected_tasks]")
        print("  demo - runs complete demo with multiple workers")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "ventilator":
        num_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        ventilator(num_tasks)
    elif mode == "worker":
        worker_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        worker(worker_id)
    elif mode == "sink":
        expected = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        sink(expected)
    elif mode == "demo":
        print("Starting demo with 3 workers...")
        
        # Start sink in background
        sink_thread = threading.Thread(target=sink, args=(20,))
        sink_thread.daemon = True
        sink_thread.start()
        
        # Start workers
        worker_threads = run_workers(3)
        
        # Give everything time to connect
        time.sleep(1)
        
        # Start ventilator
        ventilator(20)
        
        # Wait for completion
        sink_thread.join()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
