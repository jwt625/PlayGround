# ZeroMQ Learning Notes

**Date Started**: 2025-09-02  
**Goal**: Learn ZeroMQ for cluster_runner scaling

## Learning Progress

### Week 1: ZeroMQ Fundamentals ⏳

#### Day 1: Basic Patterns
- [ ] **REQ-REP Pattern** (`01-zmq-basics/req_rep.py`)
  - Simple request-reply communication
  - Synchronous, alternating pattern
  - Good for: Simple client-server interactions
  
- [ ] **PUSH-PULL Pattern** (`01-zmq-basics/push_pull.py`)
  - Work distribution pipeline
  - Load balancing automatic
  - Good for: Distributing tasks to workers
  
- [ ] **PUB-SUB Pattern** (`01-zmq-basics/pub_sub.py`)
  - One-to-many broadcasting
  - Topic-based filtering
  - Good for: Status updates, notifications
  
- [ ] **ROUTER-DEALER Pattern** (`01-zmq-basics/router_dealer.py`)
  - Advanced routing and load balancing
  - Asynchronous REQ-REP-like behavior
  - Good for: Broker patterns, scalable architectures

#### Key Concepts Learned
- [ ] **Message Framing**: How ZeroMQ handles message boundaries
- [ ] **High Water Marks (HWM)**: Flow control and backpressure
- [ ] **Connection Management**: bind() vs connect(), when to use each
- [ ] **Socket Types**: Understanding when to use each socket type
- [ ] **Message Patterns**: Synchronous vs asynchronous communication

#### Essential Resources
- **ZeroMQ Guide**: http://zguide.zeromq.org/ (Chapters 1-4)
- **PyZMQ Documentation**: https://pyzmq.readthedocs.io/

### Week 2: Cluster-Specific Patterns ⏳

#### Cluster Coordination Components
- [ ] **Coordinator** (`02-cluster-patterns/coordinator.py`)
  - Central coordination node
  - Task distribution and result collection
  - Node membership management
  
- [ ] **Worker** (`02-cluster-patterns/worker.py`)
  - Worker node implementation
  - Task processing and status reporting
  - Graceful shutdown and error handling
  
- [ ] **Monitor** (`02-cluster-patterns/monitor.py`)
  - Cluster health monitoring
  - Performance metrics collection
  - Real-time status dashboard
  
- [ ] **Heartbeat** (`02-cluster-patterns/heartbeat.py`)
  - Node failure detection
  - Automatic recovery mechanisms
  - Network partition handling

#### Patterns for Cluster Runner
- [ ] **Work Distribution**: Using PUSH-PULL for task distribution
- [ ] **Status Broadcasting**: Using PUB-SUB for status updates
- [ ] **Heartbeat and Failure Detection**: Robust node monitoring
- [ ] **Load Balancing**: Automatic and manual load balancing strategies

### Week 3: Cluster Runner Simulation ⏳

#### Protocol-Compatible Implementations
- [ ] **ZMQ Operational Context** (`03-cluster-runner-simulation/zmq_operational_context.py`)
  - ZeroMQ implementation of `OperationalContextProtocol`
  - Drop-in replacement for SSH-based context
  
- [ ] **ZMQ Node Executor** (`03-cluster-runner-simulation/zmq_node_executor.py`)
  - ZeroMQ implementation of `NodeExecutorProtocol`
  - Remote command execution over ZeroMQ
  
- [ ] **ZMQ Journal** (`03-cluster-runner-simulation/zmq_journal.py`)
  - ZeroMQ implementation of `AsyncJournalWriterProtocol`
  - Distributed result collection

#### Testing and Validation
- [ ] **Coordination Tests** (`03-cluster-runner-simulation/test_coordination.py`)
  - Test cluster coordination patterns
  - Validate failure scenarios
  
- [ ] **Performance Comparison** (`03-cluster-runner-simulation/performance_comparison.py`)
  - Compare SSH vs ZeroMQ performance
  - Latency and throughput measurements

### Week 4: Integration Planning ⏳

#### Integration Strategy
- [ ] **Protocol Mapping** (`integration-plan/protocol-mapping.md`)
  - How ZeroMQ maps to existing cluster_runner protocols
  - Interface compatibility analysis
  
- [ ] **Migration Strategy** (`integration-plan/migration-strategy.md`)
  - Step-by-step integration plan
  - Risk mitigation strategies
  
- [ ] **Rollback Plan** (`integration-plan/rollback-plan.md`)
  - Fallback strategy if issues arise
  - Maintaining SSH as backup option

## Docker Simulation Environment

### Setup and Usage
```bash
# Build the environment
./docker/simulate-cluster.sh build

# Start the cluster
./docker/simulate-cluster.sh start

# Run tests
./docker/simulate-cluster.sh test

# Simulate failures
./docker/simulate-cluster.sh fail worker1

# Monitor cluster
./docker/simulate-cluster.sh logs monitor
```

### What Can Be Tested
- ✅ **All ZeroMQ patterns**: REQ-REP, PUSH-PULL, PUB-SUB, ROUTER-DEALER
- ✅ **Multi-process simulation**: Multiple containers simulating cluster nodes
- ✅ **Network failure simulation**: Using Docker network manipulation
- ✅ **Coordination logic**: All distributed coordination patterns
- ✅ **Failure detection**: Heartbeat and recovery mechanisms

### Limitations
- ❌ **Real InfiniBand performance**: Cannot test actual hardware performance
- ❌ **True hardware-level failures**: Simulated failures only
- ❌ **Large-scale network congestion**: Limited to Docker network simulation

## Key Insights and Decisions

### Architecture Decisions
- **Separate Learning Repo**: Keeps learning isolated from production code
- **Protocol-Based Design**: Maintains compatibility with existing cluster_runner interfaces
- **Docker Simulation**: Enables comprehensive testing without real cluster hardware
- **Incremental Integration**: Reduces risk through gradual adoption

### Performance Considerations
- **Message Serialization**: JSON for human-readable, msgpack for performance
- **Connection Pooling**: Reuse connections to reduce overhead
- **Backpressure Handling**: Use High Water Marks to prevent memory issues
- **Async Operations**: Leverage asyncio for concurrent operations

### Failure Handling
- **Heartbeat Mechanisms**: Detect node failures quickly
- **Graceful Degradation**: Continue operating with reduced capacity
- **Automatic Recovery**: Reconnect and redistribute work automatically
- **Circuit Breaker Pattern**: Prevent cascading failures

## Next Steps

### Immediate Actions
1. **Complete ZeroMQ Fundamentals**: Finish implementing and testing basic patterns
2. **Test Docker Environment**: Validate simulation setup works correctly
3. **Study Cluster Runner Code**: Understand existing protocols and interfaces
4. **Begin Protocol Implementations**: Start building ZeroMQ versions of cluster_runner protocols

### Success Metrics
- [ ] All basic ZeroMQ patterns working correctly
- [ ] Docker simulation environment functional
- [ ] Protocol-compatible ZeroMQ implementations
- [ ] Performance parity with SSH approach demonstrated
- [ ] Integration plan validated through simulation

## Resources and References

### Essential Learning Materials
- **ZeroMQ Guide**: http://zguide.zeromq.org/
- **PyZMQ Documentation**: https://pyzmq.readthedocs.io/
- **ZeroMQ RFC Specifications**: For advanced patterns

### Cluster Runner Integration Points
- `src/cluster_runner/operational_context/ssh.py`
- `src/cluster_runner/operational_context/node_executor/ssh.py`
- `src/cluster_runner/schedulers/schedules.py`
- `src/cluster_runner/performance_tests/run_cluster_perf.py`

### Code Examples and Patterns
- All examples in `01-zmq-basics/` for fundamental patterns
- Cluster patterns in `02-cluster-patterns/` for distributed coordination
- Integration examples in `03-cluster-runner-simulation/` for protocol compatibility
