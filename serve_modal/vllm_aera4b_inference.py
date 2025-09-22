"""
Serve the `and-emili/aera-4b` model behind an OpenAI-compatible
endpoint on Modal with vLLM 0.8.5.post1

  • GET   /health                     – liveness check
  • POST  /v1/chat/completions        – OpenAI chat API
  • Swagger docs at /docs
"""

import modal
import subprocess
import os


# keep this in the same repo as your Modal app

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
      # ➊ libs as before …
      .pip_install(
          "torch==2.6.0",
          index_url="https://download.pytorch.org/whl/cu124",
      )
      .pip_install(
          "vllm==0.8.5.post1",
          "flashinfer-python==0.2.5",
          "huggingface_hub[hf_transfer]",
          extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6",
      )
      # ➋ COPY the local plugin into the image
      .add_local_file(
          "modal_serve/toolcall_parser.py",       # <-- path on your laptop
          "/workspace/parsers/toolcall_parser.py",  # <-- path inside container
          copy=True,
      )
      .env(
          {
              "HF_HUB_ENABLE_HF_TRANSFER": "1",
              "HF_HUB_DOWNLOAD_TIMEOUT": "180",      # seconds for file blobs
              "HF_HUB_ETAG_TIMEOUT": "60",           # seconds for the metadata HEAD
              "HF_HUB_DOWNLOAD_RETRIES": "10",       # default is 3
              "VLLM_ATTENTION_BACKEND": "FLASHINFER",
          }
      )
)



app = modal.App("aera-4b-openai-compatible")

N_GPU = 1                    
GPU_TYPE = "A10G"            
API_KEY = os.environ.get("AERA4B_API_KEY", "aera-temporary-key") # change this to the actual API key you want to use


MODEL_NAME = "and-emili/aera-4b"
MODEL_REVISION = None        

CACHE_DIR = "/root/.cache"
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

PORT = 8000
MINUTES = 60  



@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    min_containers=1,
    timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
    volumes={
        f"{CACHE_DIR}/huggingface": hf_cache,
        f"{CACHE_DIR}/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=PORT, startup_timeout=5 * MINUTES)
def serve():
    """Launch vLLM in OpenAI-compatible mode."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--api-key",
        API_KEY,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "toolcall",
        "--tool-parser-plugin",
        "/workspace/parsers/toolcall_parser.py",
        "--max-model-len",
        "20000"
    ]

    
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    
    subprocess.Popen(cmd, shell=False)



@app.local_entrypoint()
def test(timeout=10 * MINUTES):
    """
    Spin up a fresh replica, wait for /health, then send a minimal
    chat completion request.
    """
    import time, urllib.request, json

    url = serve.get_web_url()
    print("Waiting for server:", url)

    start = time.time()
    while True:
        try:
            with urllib.request.urlopen(f"{url}/health") as r:
                if r.status == 200:
                    break
        except Exception:
            if time.time() - start > timeout:
                raise RuntimeError("Health check timeout")
            time.sleep(5)

    print("✔️  Server up")

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Ciao come ti chiami?"}],
            }
        ).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as r:
        print("Response:", json.loads(r.read().decode()))
