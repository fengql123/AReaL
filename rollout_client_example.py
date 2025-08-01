#!/usr/bin/env python3
"""
Rollout Worker Client Example

This script demonstrates how to send requests to the rollout worker server
and process the responses.

Usage:
    python rollout_client_example.py --server http://localhost:8080
"""

import argparse
import asyncio
import json
import time
from typing import List, Dict

import aiohttp
import requests

class RolloutClient:
    """Client for communicating with the rollout worker server"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
    
    def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_status(self) -> dict:
        """Get server status"""
        response = requests.get(f"{self.server_url}/status", timeout=10)
        response.raise_for_status()
        return response.json()
    
    async def process_rollout_async(self, messages: List[Dict[str, str]], 
                                   answer: str = None,
                                   query_id: str = None,
                                   **generation_params) -> dict:
        """Process a rollout request asynchronously"""
        payload = {
            "messages": messages,
            "query_id": query_id,
        }
        if answer:
            payload["answer"] = answer
        
        # Add generation parameters
        for key, value in generation_params.items():
            if key in ["temperature", "max_new_tokens", "n_samples"] and value is not None:
                payload[key] = value
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/process",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    def process_rollout(self, messages: List[Dict[str, str]], 
                       answer: str = None,
                       query_id: str = None,
                       **generation_params) -> dict:
        """Process a rollout request synchronously"""
        payload = {
            "messages": messages,
            "query_id": query_id,
        }
        if answer:
            payload["answer"] = answer
        
        # Add generation parameters
        for key, value in generation_params.items():
            if key in ["temperature", "max_new_tokens", "n_samples"] and value is not None:
                payload[key] = value
        
        response = requests.post(
            f"{self.server_url}/process",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json()

def example_math_problem():
    """Example: Math problem solving"""
    return {
        "messages": [
            {"role": "user", "content": "What is 15 + 27?"}
        ],
        "answer": "42",
        "query_id": "math_001"
    }

def example_coding_problem():
    """Example: Coding problem"""
    return {
        "messages": [
            {"role": "user", "content": "Write a Python function to calculate factorial of a number."}
        ],
        "query_id": "code_001"
    }

def example_chat():
    """Example: Multi-turn chat"""
    return {
        "messages": [
            {"role": "user", "content": "Hello! Can you help me understand what photosynthesis is?"},
        ],
        "query_id": "chat_001"
    }

async def run_async_examples(client: RolloutClient):
    """Run examples asynchronously"""
    print("\n=== Running Async Examples ===")
    
    examples = [
        example_math_problem(),
        example_coding_problem(), 
        example_chat()
    ]
    
    # Process all examples concurrently
    tasks = []
    for example in examples:
        task = client.process_rollout_async(**example, n_samples=2)
        tasks.append(task)
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f"Processed {len(examples)} requests in {end_time - start_time:.2f} seconds")
    
    for i, result in enumerate(results):
        print(f"\n--- Example {i+1} ---")
        print(f"Query ID: {result['query_id']}")
        print(f"Completions: {len(result['completions'])}")
        for j, (completion, reward) in enumerate(zip(result['completions'], result['rewards'])):
            print(f"  Sample {j+1} (reward: {reward:.3f}): {completion[:100]}...")

def run_sync_examples(client: RolloutClient):
    """Run examples synchronously"""
    print("\n=== Running Sync Examples ===")
    
    # Math problem
    print("\n--- Math Problem ---")
    result = client.process_rollout(**example_math_problem())
    print(f"Query ID: {result['query_id']}")
    print(f"Completion: {result['completions'][0]}")
    print(f"Reward: {result['rewards'][0]:.3f}")
    
    # Coding problem with multiple samples
    print("\n--- Coding Problem (Multiple Samples) ---")
    result = client.process_rollout(**example_coding_problem(), n_samples=2, temperature=0.8)
    print(f"Query ID: {result['query_id']}")
    for i, (completion, reward) in enumerate(zip(result['completions'], result['rewards'])):
        print(f"Sample {i+1} (reward: {reward:.3f}): {completion[:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Rollout Worker Client Example")
    parser.add_argument("--server", default="http://localhost:8080", 
                       help="Server URL")
    parser.add_argument("--async_enabled", action="store_true", 
                       help="Run async examples")
    
    args = parser.parse_args()
    
    client = RolloutClient(args.server)
    
    # Check server health
    print("Checking server health...")
    if not client.health_check():
        print("ERROR: Server is not healthy!")
        return
    
    print("✓ Server is healthy")
    
    # Get server status
    try:
        status = client.get_status()
        print(f"Server status: {status['status']}")
        print(f"SGLang servers: {status.get('sglang_servers', 'unknown')}")
        print(f"Model: {status.get('model', 'unknown')}")
    except Exception as e:
        print(f"Could not get server status: {e}")
    
    # Run examples
    if args.async_enabled:
        asyncio.run(run_async_examples(client))
    else:
        run_sync_examples(client)
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
    