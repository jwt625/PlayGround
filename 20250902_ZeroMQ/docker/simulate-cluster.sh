#!/bin/bash

# ZeroMQ Cluster Simulation Script
# This script helps simulate various cluster scenarios for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if Docker and Docker Compose are available
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    success "Dependencies check passed"
}

# Function to build the Docker image
build_image() {
    log "Building ZeroMQ cluster Docker image..."
    cd "$PROJECT_DIR"
    docker build -t zmq-cluster -f docker/Dockerfile .
    success "Docker image built successfully"
}

# Function to start the cluster
start_cluster() {
    log "Starting ZeroMQ cluster..."
    cd "$PROJECT_DIR"
    docker-compose -f docker/docker-compose.yml up -d coordinator worker1 worker2 worker3 monitor
    
    # Wait for services to be ready
    log "Waiting for services to start..."
    sleep 10
    
    # Check if services are running
    if docker-compose -f docker/docker-compose.yml ps | grep -q "Up"; then
        success "Cluster started successfully"
        show_cluster_status
    else
        error "Failed to start cluster"
        docker-compose -f docker/docker-compose.yml logs
        exit 1
    fi
}

# Function to stop the cluster
stop_cluster() {
    log "Stopping ZeroMQ cluster..."
    cd "$PROJECT_DIR"
    docker-compose -f docker/docker-compose.yml down
    success "Cluster stopped"
}

# Function to show cluster status
show_cluster_status() {
    log "Cluster Status:"
    cd "$PROJECT_DIR"
    docker-compose -f docker/docker-compose.yml ps
    
    echo ""
    log "Service endpoints:"
    echo "  - Coordinator: localhost:5570 (worker tasks)"
    echo "  - Status updates: localhost:5571"
    echo "  - Monitor API: localhost:5572"
    echo ""
}

# Function to show logs
show_logs() {
    local service=${1:-""}
    cd "$PROJECT_DIR"
    
    if [ -n "$service" ]; then
        log "Showing logs for $service..."
        docker-compose -f docker/docker-compose.yml logs -f "$service"
    else
        log "Showing logs for all services..."
        docker-compose -f docker/docker-compose.yml logs -f
    fi
}

# Function to simulate node failure
simulate_failure() {
    local node=${1:-"worker1"}
    log "Simulating failure of $node..."
    cd "$PROJECT_DIR"
    
    docker-compose -f docker/docker-compose.yml stop "$node"
    warning "$node has been stopped (simulating failure)"
    
    echo ""
    log "To recover the node, run: $0 recover $node"
}

# Function to recover a failed node
recover_node() {
    local node=${1:-"worker1"}
    log "Recovering $node..."
    cd "$PROJECT_DIR"
    
    docker-compose -f docker/docker-compose.yml start "$node"
    success "$node has been recovered"
}

# Function to simulate network partition
simulate_partition() {
    local node=${1:-"worker1"}
    log "Simulating network partition for $node..."
    
    # Use tc (traffic control) to add network delay/loss
    docker exec "zmq-$node" tc qdisc add dev eth0 root netem delay 1000ms loss 50%
    warning "Network partition simulated for $node (high latency and packet loss)"
    
    echo ""
    log "To restore network, run: $0 restore-network $node"
}

# Function to restore network
restore_network() {
    local node=${1:-"worker1"}
    log "Restoring network for $node..."
    
    docker exec "zmq-$node" tc qdisc del dev eth0 root netem 2>/dev/null || true
    success "Network restored for $node"
}

# Function to run basic tests
run_tests() {
    log "Running basic cluster tests..."
    cd "$PROJECT_DIR"
    
    # Start test client
    docker-compose -f docker/docker-compose.yml --profile testing up -d client
    
    # Submit test tasks
    log "Submitting test tasks..."
    docker exec zmq-client python -c "
import asyncio
import json
import zmq
import zmq.asyncio

async def submit_tasks():
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect('tcp://coordinator:5570')
    
    # Submit various test tasks
    tasks = [
        {'type': 'echo', 'payload': {'message': 'Hello from test'}},
        {'type': 'compute', 'payload': {'operation': 'add', 'numbers': [1, 2, 3, 4, 5]}},
        {'type': 'sleep', 'payload': {'duration': 2.0}},
    ]
    
    for i, task in enumerate(tasks):
        task_msg = {
            'type': 'task',
            'task_id': f'test-task-{i}',
            'task_type': task['type'],
            'payload': task['payload']
        }
        await socket.send_string(json.dumps(task_msg))
        print(f'Submitted task {i}: {task[\"type\"]}')
    
    socket.close()
    context.term()

asyncio.run(submit_tasks())
"
    
    success "Test tasks submitted. Check logs to see results."
}

# Function to clean up everything
cleanup() {
    log "Cleaning up ZeroMQ cluster environment..."
    cd "$PROJECT_DIR"
    
    # Stop all services
    docker-compose -f docker/docker-compose.yml --profile testing down -v
    
    # Remove Docker image
    docker rmi zmq-cluster 2>/dev/null || true
    
    # Clean up logs
    rm -rf logs/*
    
    success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "ZeroMQ Cluster Simulation Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build                    Build the Docker image"
    echo "  start                    Start the cluster"
    echo "  stop                     Stop the cluster"
    echo "  status                   Show cluster status"
    echo "  logs [service]           Show logs (all services or specific service)"
    echo "  test                     Run basic functionality tests"
    echo "  fail <node>              Simulate node failure (default: worker1)"
    echo "  recover <node>           Recover a failed node (default: worker1)"
    echo "  partition <node>         Simulate network partition (default: worker1)"
    echo "  restore-network <node>   Restore network for node (default: worker1)"
    echo "  cleanup                  Clean up all resources"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 start     # Build and start cluster"
    echo "  $0 fail worker2          # Simulate worker2 failure"
    echo "  $0 logs coordinator      # Show coordinator logs"
    echo "  $0 test                  # Run basic tests"
    echo ""
}

# Main script logic
main() {
    case "${1:-help}" in
        "build")
            check_dependencies
            build_image
            ;;
        "start")
            check_dependencies
            start_cluster
            ;;
        "stop")
            stop_cluster
            ;;
        "status")
            show_cluster_status
            ;;
        "logs")
            show_logs "$2"
            ;;
        "test")
            run_tests
            ;;
        "fail")
            simulate_failure "$2"
            ;;
        "recover")
            recover_node "$2"
            ;;
        "partition")
            simulate_partition "$2"
            ;;
        "restore-network")
            restore_network "$2"
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
