# DevLog-000: GraphRAG Setup with Custom Endpoints

**Date**: 2025-12-05  
**Status**: Complete  
**Project**: GraphRAG Implementation

## Objective

Set up Microsoft GraphRAG with custom OpenAI-compatible endpoints for knowledge graph-based retrieval augmented generation.

## Environment Setup

### Virtual Environment
- Created Python virtual environment using `uv venv`
- Installed GraphRAG 2.7.0 with 138 dependencies
- Python version: 3.12.10

### Project Structure
- Project root: `./christmas/`
- Input directory: `./christmas/input/`
- Output directory: `./christmas/output/`
- Configuration: `./christmas/settings.yaml`

## Configuration

### Endpoints Configured

**Chat Model (Primary)**
- Endpoint: Kimi-K2 (local vLLM instance)
- Model: kimi-k2
- Features: Large context window, conversational reasoning

**Chat Model (Alternative)**
- Endpoint: Internal inference server
- Model: Llama-4-Maverick-17B-128E-Instruct-FP8
- Features: Fast inference, lightweight

**Embedding Model**
- Endpoint: Lambda embedding service
- Model: bge-m3 (BAAI/bge-m3)
- Features: Multilingual support, 8192 token context

### Key Configuration Details

Modified `settings.yaml` to support custom endpoints:
- Set `model_provider: openai` for OpenAI-compatible endpoints
- Configured `api_base` for each model endpoint
- Set `concurrent_requests: 5` to manage endpoint load
- Configured `async_mode: threaded` for parallel processing

## Implementation Steps

1. Initialized GraphRAG project with `graphrag init --root ./christmas`
2. Configured environment variables in `.env` file
3. Updated model configurations in `settings.yaml`
4. Verified embedding endpoint compatibility (tested bge-m3 model)
5. Created sample input document for testing
6. Executed indexing pipeline
7. Validated with local and global search queries

## Indexing Pipeline Results

Successfully completed all workflow stages:
- Document loading and chunking
- Entity and relationship extraction
- Graph construction and finalization
- Community detection and clustering
- Community report generation
- Text embedding generation

### Output Artifacts
- `communities.parquet`: Hierarchical community structures
- `entities.parquet`: Extracted entities
- `relationships.parquet`: Entity relationships
- `community_reports.parquet`: Community summaries
- `text_units.parquet`: Chunked text units
- LanceDB vector stores for embeddings

## Testing and Validation

### Test Query
Query: "What is GraphRAG?"  
Method: Local search

### Results
- Successfully retrieved contextual information
- Generated comprehensive response with proper citations
- Response included:
  - System overview and architecture
  - Developer information
  - Core components explanation
  - Search capabilities description
  - Integration details

## Technical Notes

### Embedding Endpoint Verification
- Initial test with kimi-k2 endpoint confirmed no embedding support
- Lambda embedding endpoint tested and verified functional
- BGE-M3 model returns 1024-dimensional embeddings

### Model Compatibility
- All endpoints confirmed OpenAI API compatible
- Authentication handled via Bearer token in headers
- JSON response format validated

## Usage Instructions

### Running Indexing
```bash
source .venv/bin/activate
graphrag index --root ./christmas
```

### Querying Data
Local search (entity-focused):
```bash
graphrag query --root ./christmas --method local --query "your question"
```

Global search (community-level summaries):
```bash
graphrag query --root ./christmas --method global --query "your question"
```

### Switching Models
To use Llama-4 instead of Kimi-K2, modify `settings.yaml`:
- Change `model_id: default_chat_model` to `model_id: llama_chat_model` in workflow sections

## Lessons Learned

1. GraphRAG requires separate embedding endpoint if chat model does not support embeddings API
2. Custom endpoints must implement OpenAI-compatible API structure
3. BGE-M3 provides effective multilingual embedding support
4. Local search provides more detailed responses than global search for specific queries
5. Concurrent request limits should be tuned based on endpoint capacity

## Next Steps

- Add production documents to input directory
- Tune chunking parameters for specific document types
- Optimize concurrent request settings based on endpoint performance
- Evaluate query performance across different search methods
- Consider implementing custom prompts for domain-specific extraction

## References

- GraphRAG Documentation: https://microsoft.github.io/graphrag/
- Configuration Reference: https://microsoft.github.io/graphrag/config/yaml/
- GraphRAG Version: 2.7.0

