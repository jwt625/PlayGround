# ZeroMQ Cluster Learning Repository

A comprehensive learning environment for integrating ZeroMQ into cluster_runner for improved distributed coordination and performance.

## 🎯 Project Goals

- **Learn ZeroMQ fundamentals** through hands-on examples
- **Develop cluster coordination patterns** for distributed systems
- **Create protocol-compatible implementations** for cluster_runner integration
- **Validate performance and reliability** through simulation
- **Plan safe integration strategy** with minimal risk to production systems

## 📁 Repository Structure

```
zmq-cluster-learning/
├── 01-zmq-basics/              # ZeroMQ fundamental patterns
│   ├── req_rep.py              # Request-Reply pattern
│   ├── push_pull.py            # Pipeline pattern for work distribution
│   ├── pub_sub.py              # Publish-Subscribe for broadcasting
│   └── router_dealer.py        # Advanced routing and load balancing
├── 02-cluster-patterns/        # Distributed coordination patterns
│   ├── coordinator.py          # Central coordination node
│   ├── worker.py               # Worker node implementation
│   ├── monitor.py              # Cluster monitoring and health
│   └── heartbeat.py            # Node failure detection
├── 03-cluster-runner-simulation/  # Protocol-compatible implementations
│   ├── zmq_operational_context.py    # ZeroMQ OperationalContextProtocol
│   ├── zmq_node_executor.py          # ZeroMQ NodeExecutorProtocol
│   ├── zmq_journal.py                # ZeroMQ AsyncJournalWriterProtocol
│   ├── test_coordination.py          # Coordination pattern tests
│   └── performance_comparison.py     # SSH vs ZeroMQ performance
├── docker/                     # Multi-node simulation environment
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Multi-node cluster setup
│   └── simulate-cluster.sh     # Cluster simulation scripts
├── docs/                       # Documentation and learning notes
│   ├── learning-notes.md       # Progress tracking and insights
│   ├── architecture-decisions.md   # Design decisions and rationale
│   └── integration-timeline.md     # Planned integration milestones
├── integration-plan/           # Production integration strategy
│   ├── protocol-mapping.md     # ZeroMQ to cluster_runner protocol mapping
│   ├── migration-strategy.md   # Step-by-step integration plan
│   └── rollback-plan.md        # Fallback strategy
└── examples/                   # Complete implementation examples
    ├── cluster-coordinator/    # Full coordinator implementation
    ├── worker-nodes/           # Worker node implementations
    └── monitoring/             # Monitoring and metrics collection
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- ZeroMQ library (`libzmq`)

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd zmq-cluster-learning
   pip install -r requirements.txt
   ```

2. **Build Docker environment**:
   ```bash
   ./docker/simulate-cluster.sh build
   ```

3. **Start cluster simulation**:
   ```bash
   ./docker/simulate-cluster.sh start
   ```

### Basic Examples

#### 1. Request-Reply Pattern
```bash
# Terminal 1: Start server
python 01-zmq-basics/req_rep.py server

# Terminal 2: Send request
python 01-zmq-basics/req_rep.py client localhost 5555 "Hello ZeroMQ"
```

#### 2. Work Distribution
```bash
# Terminal 1: Start sink
python 01-zmq-basics/push_pull.py sink

# Terminal 2: Start workers
python 01-zmq-basics/push_pull.py worker 1 &
python 01-zmq-basics/push_pull.py worker 2 &

# Terminal 3: Distribute work
python 01-zmq-basics/push_pull.py ventilator 50
```

#### 3. Cluster Simulation
```bash
# Start full cluster
./docker/simulate-cluster.sh start

# Run tests
./docker/simulate-cluster.sh test

# Simulate node failure
./docker/simulate-cluster.sh fail worker1

# Monitor cluster
./docker/simulate-cluster.sh logs monitor
```

## 📚 Learning Path

### Week 1: ZeroMQ Fundamentals
- **Focus**: Core patterns and concepts
- **Files**: `01-zmq-basics/*.py`
- **Key Topics**: REQ-REP, PUSH-PULL, PUB-SUB, ROUTER-DEALER
- **Resource**: [ZeroMQ Guide](http://zguide.zeromq.org/) Chapters 1-4

### Week 2: Cluster Patterns
- **Focus**: Distributed coordination
- **Files**: `02-cluster-patterns/*.py`
- **Key Topics**: Work distribution, status broadcasting, failure detection
- **Deliverable**: Working cluster simulation

### Week 3: Protocol Integration
- **Focus**: cluster_runner compatibility
- **Files**: `03-cluster-runner-simulation/*.py`
- **Key Topics**: Protocol implementations, performance testing
- **Deliverable**: Drop-in ZeroMQ replacements

### Week 4: Integration Planning
- **Focus**: Production integration strategy
- **Files**: `integration-plan/*.md`
- **Key Topics**: Migration plan, risk mitigation, rollback strategy
- **Deliverable**: Detailed integration roadmap

## 🔧 Simulation Capabilities

### What Can Be Fully Tested (95% coverage)
- ✅ **All ZeroMQ patterns**: REQ-REP, PUSH-PULL, PUB-SUB, ROUTER-DEALER
- ✅ **Multi-process simulation**: Docker containers simulating cluster nodes
- ✅ **Network failure simulation**: Docker network manipulation
- ✅ **Coordination logic**: All distributed coordination patterns
- ✅ **Failure detection**: Heartbeat and recovery mechanisms
- ✅ **Load balancing**: Automatic work distribution
- ✅ **Performance metrics**: Latency and throughput measurement

### Limitations and Workarounds
- ❌ **Real InfiniBand performance**: Focus on coordination logic
- ❌ **Hardware-level failures**: Use synthetic failure injection
- ❌ **Large-scale congestion**: Use traffic control simulation

## 🎛️ Cluster Simulation Commands

```bash
# Cluster management
./docker/simulate-cluster.sh start          # Start cluster
./docker/simulate-cluster.sh stop           # Stop cluster
./docker/simulate-cluster.sh status         # Show status

# Testing and monitoring
./docker/simulate-cluster.sh test           # Run basic tests
./docker/simulate-cluster.sh logs [service] # Show logs

# Failure simulation
./docker/simulate-cluster.sh fail worker1       # Simulate node failure
./docker/simulate-cluster.sh recover worker1    # Recover failed node
./docker/simulate-cluster.sh partition worker2  # Simulate network partition

# Cleanup
./docker/simulate-cluster.sh cleanup        # Clean up all resources
```

## 📊 Monitoring and Metrics

### Service Endpoints
- **Coordinator**: `localhost:5570` (worker tasks)
- **Status Updates**: `localhost:5571` (heartbeats, status)
- **Monitor API**: `localhost:5572` (metrics, health)

### Available Metrics
- Node health and status
- Task completion rates
- Response time distributions
- Failure detection latency
- Network partition recovery time

## 🔗 Integration with cluster_runner

### Key Integration Points
```python
# Existing cluster_runner protocols that need ZeroMQ implementations
OperationalContextProtocol  # Main coordination interface
NodeExecutorProtocol        # Node command execution
AsyncJournalWriterProtocol  # Result collection
```

### Integration Strategy
1. **Phase 1**: Protocol-compatible ZeroMQ implementations
2. **Phase 2**: Side-by-side testing with SSH fallback
3. **Phase 3**: Gradual migration with A/B testing
4. **Phase 4**: Full ZeroMQ adoption with SSH backup

## 📈 Success Metrics

### Learning Phase
- [ ] Complete ZeroMQ Guide chapters 1-4
- [ ] Build working multi-node Docker simulation
- [ ] Implement protocol-compatible ZeroMQ versions
- [ ] Demonstrate performance parity with SSH approach

### Integration Phase
- [ ] Zero regression in existing test functionality
- [ ] Improved coordination performance (latency/throughput)
- [ ] Enhanced fault tolerance and recovery
- [ ] Successful multi-node cluster coordination

## 🛠️ Development Tools

### Code Quality
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

### Monitoring
- **psutil**: System metrics
- **Custom dashboards**: Real-time cluster monitoring

## 📖 Documentation

- **[Learning Notes](docs/learning-notes.md)**: Progress tracking and insights
- **[Architecture Decisions](docs/architecture-decisions.md)**: Design rationale
- **[Integration Timeline](docs/integration-timeline.md)**: Planned milestones

## 🤝 Contributing

This is a learning repository focused on ZeroMQ integration with cluster_runner. Contributions should align with the learning objectives and integration goals.

## 📄 License

This project is part of the cluster_runner ecosystem and follows the same licensing terms.

---

**Next Steps**: Start with `01-zmq-basics/req_rep.py` and follow the learning path outlined in `docs/learning-notes.md`.
