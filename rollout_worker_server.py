#!/usr/bin/env python3
"""
Rollout Worker Server

A standalone HTTP server that exposes AReaLite rollout worker functionality.
You can send requests to this server and it will process them using the 
configured inference engine and workflow.

Usage:
    python rollout_worker_server.py --config config.yaml --port 8080
"""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arealite.api.cli_args import InferenceEngineConfig, GenerationHyperparameters
from arealite.api.engine_api import InferenceEngine
from arealite.api.io_struct import LLMRequest, FinetuneSpec
from arealite.api.workflow_api import RolloutWorkflow
from arealite.engine.sglang_remote import RemoteSGLangEngine
from arealite.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RolloutWorkerServer")

# Global reward function (needs to be module-level for multiprocessing)
def simple_reward_fn(prompt, completions, prompt_ids, completion_ids, answer=None, **kwargs):
    """Simple reward function - can be customized based on your needs"""
    if answer is None:
        # Default reward based on completion length
        return len(completion_ids) / 100.0
    else:
        # Simple string matching reward
        return 1.0 if answer.lower() in completions.lower() else 0.0

# Pydantic models for API
class RolloutRequest(BaseModel):
    """Request model for rollout processing"""
    messages: List[Dict[str, str]]  # Chat messages
    answer: str = None  # Expected answer for reward computation
    query_id: str = None  # Optional query identifier
    # Generation parameters (optional overrides)
    temperature: float = None
    max_new_tokens: int = None
    n_samples: int = None

class RolloutResponse(BaseModel):
    """Response model for rollout processing"""
    query_id: str
    completions: List[str]
    rewards: List[float]
    input_tokens: List[List[int]]
    output_tokens: List[List[int]]
    logprobs: List[List[float]]
    seqlens: List[int]

class SGLangServerConfig(BaseModel):
    """SGLang server configuration"""
    model_path: str  # Path to model (local or HuggingFace)
    host: str = "127.0.0.1"
    port: int = 30000
    tp_size: int = 1
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True
    additional_args: List[str] = []

class ServerConfig(BaseModel):
    """Server configuration"""
    sglang_servers: List[str] = None  # List of existing SGLang server addresses (optional)
    auto_start_sglang: List[SGLangServerConfig] = None  # Auto-start SGLang servers (optional)
    tokenizer_path: str  # Path to tokenizer
    model_name: str = "qwen"  # Model identifier  
    generation_config: Dict[str, Any] = {
        "temperature": 1.0,
        "max_new_tokens": 2048,
        "n_samples": 1,
        "top_p": 0.9,
        "top_k": 50,
        "greedy": False
    }

class SGLangServerManager:
    """Manages SGLang server processes"""
    
    def __init__(self):
        self.processes = []
        self.server_addresses = []
    
    def start_server(self, config: SGLangServerConfig) -> str:
        """Start a single SGLang server"""
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", config.model_path,
            "--host", config.host,
            "--port", str(config.port),
            "--tp-size", str(config.tp_size),
            "--mem-fraction-static", str(config.gpu_memory_utilization),
            "--tokenizer-mode", "auto",
            "--load-format", "auto",
            "--skip-tokenizer-init",  # This is crucial for AReaL compatibility
            "--dtype", "bfloat16",
            "--context-length", "32768",
        ]
        
        if config.trust_remote_code:
            cmd.append("--trust-remote-code")
        
        cmd.extend(config.additional_args)
        
        logger.info(f"Starting SGLang server: {' '.join(cmd)}")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid  # Create new process group for easier cleanup
        )
        
        self.processes.append(process)
        server_addr = f"{config.host}:{config.port}"
        self.server_addresses.append(server_addr)
        
        # Wait for server to be ready
        self._wait_for_server(server_addr)
        
        return server_addr
    
    def _wait_for_server(self, address: str, timeout: int = 300):
        """Wait for SGLang server to be ready"""
        logger.info(f"Waiting for SGLang server at {address} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{address}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ SGLang server at {address} is ready!")
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        raise RuntimeError(f"SGLang server at {address} failed to start within {timeout}s")
    
    def start_all_servers(self, configs: List[SGLangServerConfig]) -> List[str]:
        """Start all configured SGLang servers"""
        addresses = []
        for config in configs:
            addr = self.start_server(config)
            addresses.append(addr)
        return addresses
    
    def shutdown_all(self):
        """Shutdown all SGLang servers"""
        logger.info("Shutting down SGLang servers...")
        for process in self.processes:
            if process.poll() is None:  # Process is still running
                try:
                    # Kill the entire process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    # Force kill if still running
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        
        self.processes.clear()
        self.server_addresses.clear()
        logger.info("All SGLang servers shut down")

class RolloutWorkerServer:
    """Main server class that handles rollout worker functionality"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.inference_engine: InferenceEngine = None
        self.workflow: RolloutWorkflow = None
        self.tokenizer = None
        self.sglang_manager = SGLangServerManager()
        
    async def initialize(self):
        """Initialize the rollout worker components"""
        try:
            logger.info("Initializing rollout worker server...")
            
            # Determine SGLang server addresses
            sglang_addresses = []
            
            # Option 1: Auto-start SGLang servers
            if self.config.auto_start_sglang:
                logger.info("Auto-starting SGLang servers...")
                sglang_addresses = self.sglang_manager.start_all_servers(self.config.auto_start_sglang)
            
            # Option 2: Use existing SGLang servers
            elif self.config.sglang_servers:
                sglang_addresses = self.config.sglang_servers
                logger.info(f"Using existing SGLang servers: {sglang_addresses}")
            
            else:
                raise ValueError("Either 'sglang_servers' or 'auto_start_sglang' must be configured")
            
            # Set environment variable for SGLang servers
            os.environ["AREAL_LLM_SERVER_ADDRS"] = ",".join(sglang_addresses)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.config.tokenizer_path}")
            self.tokenizer = load_hf_tokenizer(self.config.tokenizer_path)
            
            # Create inference engine config
            engine_config = InferenceEngineConfig(
                consumer_batch_size=8,
                max_concurrent_rollouts=16,
                queue_size=64,
                request_timeout=300,
                request_retries=3,
                setup_timeout=60,
                schedule_policy="round_robin"
            )
            
            # Initialize inference engine
            logger.info("Initializing inference engine...")
            self.inference_engine = RemoteSGLangEngine(engine_config)
            self.inference_engine.initialize(None, None)
            
            # Create generation config
            gconfig = GenerationHyperparameters(
                temperature=self.config.generation_config.get("temperature", 1.0),
                max_new_tokens=self.config.generation_config.get("max_new_tokens", 2048),
                n_samples=self.config.generation_config.get("n_samples", 1),
                top_p=self.config.generation_config.get("top_p", 0.9),
                top_k=self.config.generation_config.get("top_k", 50),
                greedy=self.config.generation_config.get("greedy", False),
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
            )
            
            logger.info("Creating workflow...")
            self.workflow = RLVRWorkflow(
                reward_fn=simple_reward_fn,
                gconfig=gconfig,
                tokenizer=self.tokenizer,
                enable_thinking=False,
                dump_dir=None  # No dumping for server mode
            )
            
            logger.info("Rollout worker server initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            traceback.print_exc()
            raise
    
    async def process_rollout(self, request: RolloutRequest) -> RolloutResponse:
        """Process a rollout request"""
        try:
            # Prepare data dict
            data = {
                "messages": request.messages,
                "query_id": request.query_id or "unknown",
            }
            if request.answer:
                data["answer"] = request.answer
            
            # Override generation parameters if provided
            gconfig = self.workflow.gconfig
            if request.temperature is not None:
                gconfig = gconfig.new(temperature=request.temperature)
            if request.max_new_tokens is not None:
                gconfig = gconfig.new(max_new_tokens=request.max_new_tokens)
            if request.n_samples is not None:
                gconfig = gconfig.new(n_samples=request.n_samples)
            
            # Temporarily update workflow config
            original_gconfig = self.workflow.gconfig
            self.workflow.gconfig = gconfig
            
            try:
                # Run the episode
                logger.info(f"Processing rollout for query_id: {data['query_id']}")
                result = await self.workflow.arun_episode(self.inference_engine, data)
                
                # Extract results
                completions = []
                rewards = result["rewards"].tolist()
                input_tokens = []
                output_tokens = []
                logprobs = []
                seqlens = []
                
                # Process each sample in the batch
                for i in range(result.batch_size[0]):
                    # Get input/output tokens
                    input_ids = result["input_ids"][i].tolist()
                    loss_mask = result["loss_mask"][i].tolist()
                    
                    # Split into input and output tokens based on loss_mask
                    input_len = loss_mask.index(1) if 1 in loss_mask else len(input_ids)
                    inp_tokens = input_ids[:input_len]
                    out_tokens = input_ids[input_len:]
                    
                    # Decode completion
                    completion = self.tokenizer.decode(out_tokens)
                    completions.append(completion)
                    
                    input_tokens.append(inp_tokens)
                    output_tokens.append(out_tokens)
                    logprobs.append(result["logprobs"][i].tolist())
                    seqlens.append(len(input_ids))
                
                return RolloutResponse(
                    query_id=data["query_id"],
                    completions=completions,
                    rewards=rewards,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    logprobs=logprobs,
                    seqlens=seqlens
                )
                
            finally:
                # Restore original config
                self.workflow.gconfig = original_gconfig
                
        except Exception as e:
            logger.error(f"Error processing rollout: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.inference_engine:
            self.inference_engine.destroy()
        
        # Shutdown SGLang servers if we started them
        if self.config.auto_start_sglang:
            self.sglang_manager.shutdown_all()

# Global server instance
server_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global server_instance
    # Startup
    if server_instance:
        await server_instance.initialize()
    
    yield
    
    # Shutdown
    if server_instance:
        await server_instance.shutdown()

# FastAPI app
app = FastAPI(title="Rollout Worker Server", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Rollout worker server is running"}

@app.post("/process", response_model=RolloutResponse)
async def process_rollout_endpoint(request: RolloutRequest):
    """Process a rollout request"""
    if not server_instance:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    return await server_instance.process_rollout(request)

@app.get("/status")
async def status():
    """Get server status"""
    if not server_instance or not server_instance.inference_engine:
        return {"status": "not_initialized"}
    
    # Get SGLang server addresses
    sglang_servers = []
    if server_instance.config.auto_start_sglang:
        sglang_servers = server_instance.sglang_manager.server_addresses
    elif server_instance.config.sglang_servers:
        sglang_servers = server_instance.config.sglang_servers
    
    return {
        "status": "ready",
        "sglang_servers": sglang_servers,
        "auto_started_sglang": bool(server_instance.config.auto_start_sglang),
        "model": server_instance.config.model_name,
        "version": server_instance.inference_engine.get_version()
    }

def load_config(config_path: str) -> ServerConfig:
    """Load configuration from file"""
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError("Config file must be JSON or YAML")
    
    return ServerConfig(**config_dict)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        if server_instance:
            # Use asyncio to run the async shutdown
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server_instance.shutdown())
            loop.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Rollout Worker Server")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create server instance
    global server_instance
    server_instance = RolloutWorkerServer(config)
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Run server
    logger.info(f"Starting rollout worker server on {args.host}:{args.port}")
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        # Ensure cleanup
        if server_instance:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server_instance.shutdown())
            loop.close()

if __name__ == "__main__":
    main()