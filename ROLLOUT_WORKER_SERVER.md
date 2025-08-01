# Rollout Worker Server

A standalone HTTP server that exposes AReaLite rollout worker functionality. You can send requests to this server and it will process them using the configured inference engine and workflow.

## Features

- **HTTP API**: RESTful endpoints for submitting rollout requests
- **Async Processing**: Supports concurrent request processing
- **Configurable**: YAML/JSON configuration for easy customization  
- **SGLang Integration**: Works with remote SGLang servers for inference
- **Custom Workflows**: Supports custom reward functions and generation parameters
- **Health Monitoring**: Built-in health check and status endpoints

## Prerequisites

1. **SGLang Servers**: You need running SGLang servers for inference
2. **Dependencies**: Install required packages:
   ```bash
   pip install fastapi uvicorn aiohttp pyyaml
   ```

## Quick Start

### Option A: Auto-Start SGLang (Recommended)

**Single command setup** - the server will automatically start SGLang servers for you:

#### 1. Configure with Auto-Start

Create a configuration file (`rollout_server_config_autostart.yaml`):
```yaml
# Auto-start SGLang servers (recommended for single-command setup)
auto_start_sglang:
  - model_path: "Qwen/Qwen2.5-7B-Instruct"
    host: "127.0.0.1"
    port: 30000
    tp_size: 1
    gpu_memory_utilization: 0.85
    trust_remote_code: true

# Tokenizer path (required)
tokenizer_path: "Qwen/Qwen2.5-7B-Instruct"

# Model identifier
model_name: "qwen2.5-7b"

# Generation configuration
generation_config:
  temperature: 1.0
  max_new_tokens: 2048
  n_samples: 1
  top_p: 0.9
  top_k: 50
  greedy: false
```

#### 2. Start the Server (One Command!)

```bash
python rollout_worker_server.py --config rollout_server_config_autostart.yaml --port 8080
```

The server will:
1. Automatically start SGLang server(s)
2. Wait for them to be ready
3. Start the rollout worker HTTP API
4. Automatically clean up SGLang servers on shutdown

### Option B: Use Existing SGLang Servers

If you prefer to manage SGLang servers manually:

#### 1. Start SGLang Servers

```bash
# Server 1
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1

# Server 2 (optional, for load balancing)  
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30001 \
    --tp-size 1
```

#### 2. Configure for Existing Servers

Create a configuration file (`rollout_server_config.yaml`):
```yaml
# Use existing SGLang servers
sglang_servers:
  - "localhost:30000" 
  - "localhost:30001"

# Tokenizer path (required)
tokenizer_path: "Qwen/Qwen2.5-7B-Instruct"

# Model identifier
model_name: "qwen2.5-7b"

# Generation configuration
generation_config:
  temperature: 1.0
  max_new_tokens: 2048
  n_samples: 1
  top_p: 0.9
  top_k: 50
  greedy: false
```

#### 3. Start the Rollout Worker Server

```bash
python rollout_worker_server.py --config rollout_server_config.yaml --port 8080
```

### 4. Test the Server

Check if the server is running:
```bash
curl http://localhost:8080/health
```

Get server status:
```bash
curl http://localhost:8080/status
```

## API Endpoints

### POST /process

Process a rollout request.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "What is 2 + 2?"}
  ],
  "answer": "4",
  "query_id": "math_001",
  "temperature": 0.8,
  "max_new_tokens": 1024,
  "n_samples": 2
}
```

**Response:**
```json
{
  "query_id": "math_001",
  "completions": ["The answer is 4.", "2 + 2 equals 4."],
  "rewards": [1.0, 1.0],
  "input_tokens": [[1, 2, 3, ...]],
  "output_tokens": [[10, 11, 12, ...]],
  "logprobs": [[-0.1, -0.2, ...]], 
  "seqlens": [45, 42]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Rollout worker server is running"
}
```

### GET /status

Get server status and configuration.

**Response:**
```json
{
  "status": "ready",
  "sglang_servers": ["localhost:30000", "localhost:30001"],
  "model": "qwen2.5-7b",
  "version": 0
}
```

## Using the Client

### Python Client

Use the provided client example:

```bash
# Run synchronous examples
python rollout_client_example.py --server http://localhost:8080

# Run asynchronous examples
python rollout_client_example.py --server http://localhost:8080 --async_enabled
```

### Python Code

```python
import requests

# Simple request
response = requests.post("http://localhost:8080/process", json={
    "messages": [
        {"role": "user", "content": "Write a hello world program in Python"}
    ],
    "query_id": "code_001"
})

result = response.json()
print(f"Completion: {result['completions'][0]}")
print(f"Reward: {result['rewards'][0]}")
```

### cURL

```bash
curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "query_id": "geo_001"
  }'
```

## Configuration Options

### Server Configuration

**Core Settings:**
- `tokenizer_path`: Path to the tokenizer (local path or HuggingFace model ID)
- `model_name`: Model identifier for logging/status
- `generation_config`: Default generation parameters

**SGLang Server Options (choose one):**

**Option 1: Auto-start SGLang servers (recommended)**
- `auto_start_sglang`: List of SGLang server configurations to auto-start
  - `model_path`: Path to model (local or HuggingFace)
  - `host`: Server host (default: "127.0.0.1")
  - `port`: Server port (default: 30000)
  - `tp_size`: Tensor parallelism size (default: 1)
  - `gpu_memory_utilization`: GPU memory fraction (default: 0.85)
  - `trust_remote_code`: Trust remote code (default: true)
  - `additional_args`: Extra SGLang arguments (default: [])

**Option 2: Use existing SGLang servers**
- `sglang_servers`: List of existing SGLang server addresses

### Generation Parameters

You can override these per request:

- `temperature`: Sampling temperature (0.0 = greedy)
- `max_new_tokens`: Maximum tokens to generate
- `n_samples`: Number of completions to generate
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `greedy`: Whether to use greedy decoding

## Customization

### Custom Reward Functions

The server uses a simple reward function by default. You can modify the `simple_reward_fn` in `rollout_worker_server.py`:

```python
def simple_reward_fn(prompt, completions, prompt_ids, completion_ids, answer=None, **kwargs):
    """Custom reward function"""
    if answer is None:
        # Default reward based on completion length
        return len(completion_ids) / 100.0
    else:
        # Custom logic here
        return 1.0 if answer.lower() in completions.lower() else 0.0
```

### Custom Workflows

You can replace the `RLVRWorkflow` with your own workflow implementation by subclassing `RolloutWorkflow`:

```python
class CustomWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine: InferenceEngine, data):
        # Your custom logic here
        pass
```

## Troubleshooting

### Common Issues

1. **Server won't start**: Check that SGLang servers are running and accessible
2. **Timeout errors**: Increase request timeout in configuration
3. **Memory issues**: Reduce `max_concurrent_rollouts` or `n_samples`
4. **Import errors**: Install missing dependencies

### Logging

The server logs important events. Increase log level for debugging:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tuning

- **Concurrent rollouts**: Adjust `max_concurrent_rollouts` in the engine config
- **Queue size**: Increase `queue_size` for better throughput
- **Multiple SGLang servers**: Add more servers for load balancing
- **Request batching**: Use async client for better performance

## Examples

See `rollout_client_example.py` for comprehensive examples including:

- Math problem solving with answer verification
- Code generation tasks
- Multi-turn conversations
- Async batch processing
- Custom generation parameters

## Architecture

The server consists of:

1. **FastAPI Web Server**: Handles HTTP requests
2. **RemoteSGLangEngine**: Communicates with SGLang servers
3. **RLVRWorkflow**: Processes requests and computes rewards
4. **WorkflowExecutor**: Manages async execution

```
Client Request → FastAPI → RolloutWorkerServer → InferenceEngine → SGLang Servers
                                ↓
              WorkflowExecutor → RLVRWorkflow → Reward Function
```