"""vLLM-based inference server for GLM-4.5-Air."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import json

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
import torch

from .config import InferenceConfig


logger = logging.getLogger(__name__)


class GLMVLLMServer:
    """vLLM-based inference server for GLM-4.5-Air."""
    
    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the vLLM server.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self._startup_time: Optional[float] = None
        
    async def initialize(self) -> None:
        """Initialize the vLLM engine."""
        logger.info("Initializing vLLM engine for GLM-4.5-Air...")
        start_time = time.time()
        
        # Configure engine arguments
        engine_args = AsyncEngineArgs(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            trust_remote_code=True,  # Required for GLM models
            dtype="auto",  # Let vLLM choose the best dtype
            enforce_eager=False,  # Use CUDA graphs for better performance
            disable_log_stats=False,
            enable_prefix_caching=True,  # Enable prefix caching for better efficiency
        )
        
        # Create the async engine
        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._startup_time = time.time() - start_time
            
            logger.info(f"vLLM engine initialized successfully in {self._startup_time:.2f} seconds")
            
            # Log engine configuration
            await self._log_engine_info()
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    async def _log_engine_info(self) -> None:
        """Log engine configuration and model information."""
        if not self.engine:
            return
            
        # Get model configuration
        model_config = self.engine.engine.model_config
        
        logger.info("=== vLLM Engine Configuration ===")
        logger.info(f"Model: {model_config.model}")
        logger.info(f"Model Type: {getattr(model_config.hf_config, 'model_type', 'unknown')}")
        logger.info(f"Tensor Parallel Size: {self.config.tensor_parallel_size}")
        logger.info(f"Max Model Length: {model_config.max_model_len}")
        logger.info(f"Max Num Seqs: {self.config.max_num_seqs}")
        logger.info(f"GPU Memory Utilization: {self.config.gpu_memory_utilization}")
        logger.info(f"Vocab Size: {getattr(model_config.hf_config, 'vocab_size', 'unknown')}")
        logger.info("================================")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Generate text using the GLM-4.5-Air model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated text response or async generator for streaming
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            stop=stop or [],
            **kwargs
        )
        
        # Generate request ID
        request_id = random_uuid()
        
        if stream:
            return self._stream_generate(prompt, sampling_params, request_id)
        else:
            return await self._generate_single(prompt, sampling_params, request_id)
    
    async def _generate_single(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> Dict[str, Any]:
        """Generate a single response."""
        start_time = time.time()
        
        # Add the request to the engine
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        # Get the final result
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if not final_output:
            raise RuntimeError("No output generated")
        
        # Extract the generated text
        generated_text = final_output.outputs[0].text
        finish_reason = final_output.outputs[0].finish_reason
        
        generation_time = time.time() - start_time
        
        return {
            "id": request_id,
            "object": "text_completion",
            "created": int(start_time),
            "model": self.config.model_path,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "finish_reason": finish_reason,
                "logprobs": None
            }],
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids)
            },
            "generation_time": generation_time
        }
    
    async def _stream_generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response."""
        start_time = time.time()
        
        # Add the request to the engine
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        previous_text = ""
        async for request_output in results_generator:
            if request_output.outputs:
                current_text = request_output.outputs[0].text
                delta_text = current_text[len(previous_text):]
                
                if delta_text:
                    yield {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(start_time),
                        "model": self.config.model_path,
                        "choices": [{
                            "text": delta_text,
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": None
                        }]
                    }
                    previous_text = current_text
        
        # Send final chunk with finish reason
        if request_output and request_output.outputs:
            yield {
                "id": request_id,
                "object": "text_completion",
                "created": int(start_time),
                "model": self.config.model_path,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "finish_reason": request_output.outputs[0].finish_reason,
                    "logprobs": None
                }],
                "usage": {
                    "prompt_tokens": len(request_output.prompt_token_ids),
                    "completion_tokens": len(request_output.outputs[0].token_ids),
                    "total_tokens": len(request_output.prompt_token_ids) + len(request_output.outputs[0].token_ids)
                }
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the server."""
        if not self.engine:
            return {
                "status": "unhealthy",
                "message": "Engine not initialized"
            }
        
        try:
            # Simple generation test
            test_prompt = "Hello"
            result = await self.generate(
                prompt=test_prompt,
                max_tokens=5,
                temperature=0.0
            )
            
            return {
                "status": "healthy",
                "model": self.config.model_path,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "startup_time": self._startup_time,
                "test_generation": {
                    "prompt": test_prompt,
                    "response": result["choices"][0]["text"][:50] + "..." if len(result["choices"][0]["text"]) > 50 else result["choices"][0]["text"]
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}"
            }
    
    async def shutdown(self) -> None:
        """Shutdown the vLLM engine."""
        if self.engine:
            logger.info("Shutting down vLLM engine...")
            # vLLM doesn't have an explicit shutdown method, but we can clean up
            self.engine = None
            logger.info("vLLM engine shutdown complete")


async def create_glm_server(config: InferenceConfig) -> GLMVLLMServer:
    """Create and initialize a GLM vLLM server.
    
    Args:
        config: Inference configuration
        
    Returns:
        Initialized GLM vLLM server
    """
    server = GLMVLLMServer(config)
    await server.initialize()
    return server
