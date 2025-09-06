#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import zmq
        print(f"✓ ZeroMQ: {zmq.zmq_version()} (PyZMQ: {zmq.pyzmq_version()})")
    except ImportError as e:
        print(f"✗ ZeroMQ import failed: {e}")
        return False
    
    try:
        import msgpack
        print(f"✓ msgpack: {msgpack.version}")
    except ImportError as e:
        print(f"✗ msgpack import failed: {e}")
        return False
    
    try:
        import psutil
        print(f"✓ psutil: {psutil.__version__}")
    except ImportError as e:
        print(f"✗ psutil import failed: {e}")
        return False
    
    try:
        import click
        try:
            version = click.__version__
        except AttributeError:
            import importlib.metadata
            version = importlib.metadata.version("click")
        print(f"✓ click: {version}")
    except ImportError as e:
        print(f"✗ click import failed: {e}")
        return False

    try:
        import rich
        try:
            from rich import __version__ as rich_version
            print(f"✓ rich: {rich_version}")
        except ImportError:
            import importlib.metadata
            version = importlib.metadata.version("rich")
            print(f"✓ rich: {version}")
    except ImportError as e:
        print(f"✗ rich import failed: {e}")
        return False

    try:
        import typer
        try:
            version = typer.__version__
        except AttributeError:
            import importlib.metadata
            version = importlib.metadata.version("typer")
        print(f"✓ typer: {version}")
    except ImportError as e:
        print(f"✗ typer import failed: {e}")
        return False
    
    return True


def test_zmq_basic():
    """Test basic ZeroMQ functionality."""
    print("\nTesting basic ZeroMQ functionality...")
    
    try:
        import zmq
        import time
        import threading
        
        # Create context
        context = zmq.Context()
        
        # Test REQ-REP pattern
        def server():
            socket = context.socket(zmq.REP)
            socket.bind("tcp://127.0.0.1:5555")
            message = socket.recv_string()
            socket.send_string(f"Echo: {message}")
            socket.close()
        
        # Start server in background
        server_thread = threading.Thread(target=server)
        server_thread.daemon = True
        server_thread.start()
        
        # Give server time to start
        time.sleep(0.1)
        
        # Test client
        client_socket = context.socket(zmq.REQ)
        client_socket.connect("tcp://127.0.0.1:5555")
        client_socket.send_string("Hello ZeroMQ")
        reply = client_socket.recv_string()
        client_socket.close()
        
        context.term()
        
        if reply == "Echo: Hello ZeroMQ":
            print("✓ ZeroMQ REQ-REP pattern working")
            return True
        else:
            print(f"✗ ZeroMQ test failed: unexpected reply '{reply}'")
            return False
            
    except Exception as e:
        print(f"✗ ZeroMQ test failed: {e}")
        return False


def test_async_support():
    """Test async support."""
    print("\nTesting async support...")

    try:
        import asyncio
        import zmq.asyncio

        async def async_test():
            context = zmq.asyncio.Context()

            # Create a simple PUSH-PULL pair for testing
            push_socket = context.socket(zmq.PUSH)
            pull_socket = context.socket(zmq.PULL)

            push_socket.bind("tcp://127.0.0.1:5556")
            pull_socket.connect("tcp://127.0.0.1:5556")

            # Give sockets time to connect
            await asyncio.sleep(0.1)

            # Send and receive a message
            await push_socket.send_string("async test")
            message = await pull_socket.recv_string()

            push_socket.close()
            pull_socket.close()
            context.term()

            return message == "async test"

        result = asyncio.run(async_test())

        if result:
            print("✓ ZeroMQ async support working")
            return True
        else:
            print("✗ ZeroMQ async test failed")
            return False

    except Exception as e:
        print(f"✗ ZeroMQ async test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("ZeroMQ Learning Environment Setup Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_zmq_basic():
        tests_passed += 1
    
    if test_async_support():
        tests_passed += 1
    
    print(f"\nTest Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your environment is ready for ZeroMQ learning.")
        print("\nNext steps:")
        print("1. Try: python 01-zmq-basics/req_rep.py server")
        print("2. In another terminal: python 01-zmq-basics/req_rep.py client")
        print("3. Follow the learning path in docs/learning-notes.md")
        return True
    else:
        print("❌ Some tests failed. Please check your installation.")
        return False


if __name__ == "__main__":
    main()
