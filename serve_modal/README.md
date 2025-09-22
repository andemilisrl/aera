# Serving ÆRA-4B on Modal with vLLM

This directory contains the setup to serve ÆRA-4B through vLLM on Modal with custom tool calling support.

## Files

- `toolcall_parser.py` - Custom tool parsing logic for ÆRA's `<tool_call>` format
- `vllm_aera4b_inference.py` - Modal inference service (configured for A10G GPU)

## Deployment Steps

1. Set environment variables:
   ```bash
   export MODAL_TOKEN_ID="your_token_id"
   export MODAL_TOKEN_SECRET="your_token_secret"
   ```

2. Install and configure Modal:
   ```bash
   pip install --quiet --upgrade modal
   modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
   ```

3. Deploy the service:
   ```bash
   modal deploy vllm_aera4b_inference.py
   ```

The service will be available at the Modal-provided URL with OpenAI-compatible endpoints.