# DevLog-000: ZeroMQ Learning and Integration Strategy for Cluster Runner Scaling

**Date**: 2025-09-02  
**Author**: Augment & Wentao
**Status**: Planning Phase  
**Goal**: Scale cluster_runner using ZeroMQ for improved coordination and performance

## Executive Summary

We are scaling the cluster_runner system and have decided to use ZeroMQ for distributed coordination. This document outlines a learning strategy and integration plan that minimizes risk to the existing production system while enabling effective ZeroMQ adoption.

## Current System Analysis

### Existing Architecture
The cluster_runner currently uses:
- **SSH-based coordination** via `NodeExecutorSSH` and `asyncssh`
- **Async task scheduling** with `nodes_concurrency_set` and `run_perf_test_scheduled`
- **Journal-based result collection** using `AsyncPerfTestJournalFileWriter`
- **Protocol-based abstractions** (`OperationalContextProtocol`, `NodeExecutorProtocol`)

### Current Hardware Configuration
From test results in `/home/ubuntu/perftests-single/`:
- 2 high-end GPUs with 18 NVLink connections each (26.562 GB/s per link)
- Single network interface (mlx5_0) with 100 Gbps Ethernet
- Single NUMA node configuration
- Mixed test results: GPU tests passed, NCCL all_reduce failed

### Key Integration Points
```python
# Critical protocols for ZeroMQ integration
OperationalContextProtocol  # Main coordination interface
NodeExecutorProtocol        # Node command execution
AsyncJournalWriterProtocol  # Result collection
```

## Learning Strategy: Separate Repository Approach

### Rationale for Separate Repo
1. **Clean Learning Environment**: No interference with production cluster_runner
2. **Risk Mitigation**: Existing system continues working during learning phase
3. **Protocol Compatibility**: Existing abstractions make integration cleaner
4. **Team Collaboration**: Easier to share learning progress

### Recommended Repository Structure
```
zmq-cluster-learning/
├── 01-zmq-basics/
│   ├── req_rep.py              # Request-Reply pattern
│   ├── push_pull.py            # Pipeline pattern for work distribution
│   ├── pub_sub.py              # Publish-Subscribe for status updates
│   └── router_dealer.py        # Advanced routing and load balancing
├── 02-cluster-patterns/
│   ├── coordinator.py          # Central coordination node
│   ├── worker.py               # Worker node implementation
│   ├── monitor.py              # Cluster monitoring and health
│   └── heartbeat.py            # Node failure detection
├── 03-cluster-runner-simulation/
│   ├── zmq_operational_context.py    # ZeroMQ OperationalContextProtocol impl
│   ├── zmq_node_executor.py          # ZeroMQ NodeExecutorProtocol impl
│   ├── zmq_journal.py                # ZeroMQ AsyncJournalWriterProtocol impl
│   ├── test_coordination.py          # Test cluster coordination patterns
│   └── performance_comparison.py     # Compare SSH vs ZeroMQ performance
├── docker/
│   ├── Dockerfile                    # Based on cluster_runner's container
│   ├── docker-compose.yml           # Multi-node simulation
│   ├── simulate-cluster.sh           # Cluster simulation scripts
│   └── network-configs/              # Network topology simulation
├── docs/
│   ├── learning-notes.md             # ZeroMQ learning progress
│   ├── architecture-decisions.md     # Design decisions and rationale
│   └── integration-timeline.md       # Planned integration milestones
├── integration-plan/
│   ├── protocol-mapping.md           # How ZeroMQ maps to existing protocols
│   ├── migration-strategy.md         # Step-by-step integration plan
│   └── rollback-plan.md              # Fallback strategy if issues arise
└── examples/
    ├── cluster-coordinator/          # Full coordinator implementation
    ├── worker-nodes/                 # Worker node implementations
    └── monitoring/                   # Monitoring and metrics collection
```

## ZeroMQ Learning Path

### Week 1: ZeroMQ Fundamentals
**Focus**: Core patterns and concepts
- **Essential Resource**: ZeroMQ Guide (http://zguide.zeromq.org/)
- **Key Patterns**: REQ-REP, PUSH-PULL, PUB-SUB, ROUTER-DEALER
- **Critical Topics**: Message framing, High Water Marks, connection management

### Week 2: Cluster-Specific Patterns
**Focus**: Distributed coordination patterns
- Work distribution using PUSH-PULL
- Status broadcasting with PUB-SUB
- Heartbeat and failure detection
- Load balancing and routing

### Week 3: Cluster Runner Simulation
**Focus**: Protocol-compatible implementations
- Build ZeroMQ versions of cluster_runner protocols
- Test with Docker-based multi-node simulation
- Performance comparison with SSH-based approach

### Week 4: Integration Planning
**Focus**: Production integration strategy
- Analyze integration points in cluster_runner codebase
- Design adapter interfaces
- Plan migration timeline and rollback strategy

## Simulation Capabilities on Current Hardware

### What Can Be Fully Tested (95% coverage)
- **All ZeroMQ patterns**: REQ-REP, PUSH-PULL, PUB-SUB, ROUTER-DEALER
- **Multi-process simulation**: Multiple Python processes on localhost
- **Container-based cluster**: Docker containers simulating cluster nodes
- **Network failure simulation**: Using Docker network manipulation
- **Coordination logic**: All distributed coordination patterns

### Docker-Based Simulation Strategy
```yaml
# docker-compose.yml example
version: '3.8'
services:
  coordinator:
    build: .
    ports:
      - "5555:5555"
    environment:
      - ROLE=coordinator
  
  worker1:
    build: .
    environment:
      - ROLE=worker
      - COORDINATOR_HOST=coordinator
  
  worker2:
    build: .
    environment:
      - ROLE=worker
      - COORDINATOR_HOST=coordinator
```

### Limitations and Workarounds
**Cannot Test**:
- Real InfiniBand performance
- True hardware-level failures
- Large-scale network congestion

**Workarounds**:
- Focus on coordination logic rather than raw performance
- Use synthetic workloads for testing patterns
- Simulate network latency with traffic control tools

## Integration Strategy

### Phase 1: Protocol Compatibility (Week 4)
```python
# Implement ZeroMQ versions of existing protocols
class ZMQOperationalContext:
    """ZeroMQ implementation of OperationalContextProtocol"""
    
    def journal(self) -> AsyncJournalWriterProtocol:
        return self.zmq_journal
    
    def executor(self, node_id: str) -> NodeExecutorProtocol:
        return self.zmq_executors[node_id]
    
    def system_config(self, node_id: str) -> SystemConfiguration:
        return self.cached_configs[node_id]
```

### Phase 2: Side-by-Side Testing (Week 5)
```python
# Add ZeroMQ as optional transport in cluster_runner
if args.transport == "zmq":
    opcon = await build_zmq_context(descriptor, filter_fn, journal)
else:
    opcon = await build_ssh_context(descriptor, filter_fn, journal)  # existing
```

### Phase 3: Gradual Migration (Weeks 6+)
- Replace specific operations incrementally
- Maintain SSH as fallback option
- A/B test performance and reliability
- Monitor for regressions

## Key Focus Areas for Cluster Runner Integration

### 1. Async Coordination Patterns
```python
# cluster_runner's async patterns that need ZeroMQ equivalents
async with nodes_concurrency_set() as task_runner:
    await task_runner(nodes=group, coroutine=group_call(group))
```

### 2. Result Collection Patterns
```python
# Journal-based streaming that needs ZeroMQ transport
await journal.write(test_result)
```

### 3. Complex Scheduling Patterns
```python
# Scheduling logic that needs distributed coordination
schedule_all_servers()
schedule_leaf_to_leaf_pairs()
schedule_increasing_size_groups()
```

## Success Metrics

### Learning Phase Success Criteria
- [ ] Complete ZeroMQ Guide chapters 1-4
- [ ] Build working multi-node Docker simulation
- [ ] Implement protocol-compatible ZeroMQ versions
- [ ] Demonstrate performance parity with SSH approach

### Integration Phase Success Criteria
- [ ] Zero regression in existing test functionality
- [ ] Improved coordination performance (latency/throughput)
- [ ] Enhanced fault tolerance and recovery
- [ ] Successful multi-node cluster coordination

## Risk Mitigation

### Technical Risks
- **Performance regression**: Mitigated by A/B testing and fallback to SSH
- **Compatibility issues**: Mitigated by protocol-based design
- **Learning curve**: Mitigated by structured learning approach

### Operational Risks
- **Production disruption**: Mitigated by separate learning repo
- **Timeline delays**: Mitigated by incremental integration approach
- **Team knowledge gaps**: Mitigated by documentation and knowledge sharing

## Next Steps

1. **Create separate learning repository** with structure outlined above
2. **Begin ZeroMQ fundamentals learning** using ZeroMQ Guide
3. **Set up Docker-based simulation environment**
4. **Build protocol-compatible ZeroMQ implementations**
5. **Plan integration timeline** based on learning progress

## Resources

### Essential Learning Resources
- **ZeroMQ Guide**: http://zguide.zeromq.org/
- **PyZMQ Documentation**: https://pyzmq.readthedocs.io/
- **ZeroMQ RFC Specifications**: For advanced patterns

### Key Libraries
```python
import zmq
import zmq.asyncio  # For async operations
import msgpack     # For efficient serialization
import json        # For human-readable messages
```

### Integration Reference Files
- `src/cluster_runner/operational_context/ssh.py`
- `src/cluster_runner/operational_context/node_executor/ssh.py`
- `src/cluster_runner/schedulers/schedules.py`
- `src/cluster_runner/performance_tests/run_cluster_perf.py`

---

**Note**: This document should be copied to the new learning repository as the foundation for the ZeroMQ integration project.
