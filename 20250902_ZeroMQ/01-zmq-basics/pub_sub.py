#!/usr/bin/env python3
"""
ZeroMQ Publish-Subscribe Pattern Example

This demonstrates the PUB-SUB pattern for broadcasting:
- Publisher broadcasts messages to multiple subscribers
- Subscribers can filter messages by topic
- Communication is asynchronous and one-to-many
- Late subscribers miss earlier messages
"""

import zmq
import time
import random
import sys
import threading
from typing import Optional


def publisher(port: int = 5556, duration: int = 10) -> None:
    """
    Publisher that broadcasts weather updates.
    
    Args:
        port: Port to bind publisher socket to
        duration: How long to publish messages (seconds)
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    
    print(f"Publisher starting on port {port}")
    print(f"Publishing for {duration} seconds...")
    
    # Give subscribers time to connect
    time.sleep(1)
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Generate random weather data
            zipcode = random.randint(10000, 99999)
            temperature = random.randint(-20, 40)
            humidity = random.randint(10, 90)
            
            # Create message with topic (zipcode) and data
            message = f"{zipcode} {temperature} {humidity}"
            socket.send_string(message)
            print(f"Published: {message}")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nPublisher shutting down...")
    finally:
        socket.close()
        context.term()


def subscriber(publisher_host: str = "localhost", publisher_port: int = 5556,
               topic_filter: str = "", duration: int = 15) -> None:
    """
    Subscriber that receives filtered messages from publisher.
    
    Args:
        publisher_host: Publisher hostname or IP
        publisher_port: Publisher port
        topic_filter: Topic prefix to filter on (empty = all messages)
        duration: How long to listen (seconds)
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{publisher_host}:{publisher_port}")
    
    # Set topic filter
    socket.setsockopt_string(zmq.SUBSCRIBE, topic_filter)
    
    print(f"Subscriber listening for topic '{topic_filter}'")
    print(f"Listening for {duration} seconds...")
    
    start_time = time.time()
    message_count = 0
    
    try:
        while time.time() - start_time < duration:
            try:
                # Non-blocking receive with timeout
                message = socket.recv_string(zmq.NOBLOCK)
                message_count += 1
                
                # Parse weather data
                parts = message.split()
                if len(parts) == 3:
                    zipcode, temp, humidity = parts
                    print(f"Weather update {message_count}: "
                          f"Zipcode {zipcode}, {temp}°C, {humidity}% humidity")
                else:
                    print(f"Received: {message}")
                    
            except zmq.Again:
                # No message available, continue
                time.sleep(0.1)
                continue
                
    except KeyboardInterrupt:
        print(f"\nSubscriber shutting down... Received {message_count} messages")
    finally:
        socket.close()
        context.term()


def multi_topic_publisher(port: int = 5556, duration: int = 10) -> None:
    """
    Publisher that broadcasts multiple types of messages.
    
    Args:
        port: Port to bind publisher socket to
        duration: How long to publish messages (seconds)
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    
    print(f"Multi-topic publisher starting on port {port}")
    
    # Give subscribers time to connect
    time.sleep(1)
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Publish different types of messages
            topics = ["weather", "news", "sports", "tech"]
            topic = random.choice(topics)
            
            if topic == "weather":
                zipcode = random.randint(10000, 99999)
                temp = random.randint(-20, 40)
                message = f"weather {zipcode} {temp}°C"
            elif topic == "news":
                headline = f"Breaking news item {random.randint(1, 100)}"
                message = f"news {headline}"
            elif topic == "sports":
                score = f"{random.randint(0, 10)}-{random.randint(0, 10)}"
                message = f"sports Game result: {score}"
            else:  # tech
                version = f"v{random.randint(1, 5)}.{random.randint(0, 9)}"
                message = f"tech New release {version}"
            
            socket.send_string(message)
            print(f"Published: {message}")
            
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("\nPublisher shutting down...")
    finally:
        socket.close()
        context.term()


def demo_pub_sub() -> None:
    """
    Demonstrate pub-sub with multiple subscribers filtering different topics.
    """
    print("Starting pub-sub demo...")
    
    # Start subscribers for different topics
    subscribers = []
    topics = ["weather", "news", "sports"]
    
    for topic in topics:
        thread = threading.Thread(
            target=subscriber, 
            args=("localhost", 5556, topic, 12)
        )
        thread.daemon = True
        thread.start()
        subscribers.append(thread)
    
    # Start subscriber for all messages
    all_thread = threading.Thread(
        target=subscriber,
        args=("localhost", 5556, "", 12)
    )
    all_thread.daemon = True
    all_thread.start()
    subscribers.append(all_thread)
    
    # Give subscribers time to connect
    time.sleep(1)
    
    # Start publisher
    multi_topic_publisher(5556, 10)
    
    # Wait for subscribers to finish
    for thread in subscribers:
        thread.join()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pub_sub.py [publisher|subscriber|demo] [args...]")
        print("  publisher [port] [duration]")
        print("  subscriber [host] [port] [topic_filter] [duration]")
        print("  demo - runs complete demo with multiple subscribers")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "publisher":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5556
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        publisher(port, duration)
    elif mode == "subscriber":
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5556
        topic = sys.argv[4] if len(sys.argv) > 4 else ""
        duration = int(sys.argv[5]) if len(sys.argv) > 5 else 15
        subscriber(host, port, topic, duration)
    elif mode == "demo":
        demo_pub_sub()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
