import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # Optional for CSS/JS later
from pydantic import BaseModel
import httpx # Async HTTP client
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm # For server-side progress logging (optional)
import queue # For SSE communication
import json
import logging
from chat_template import format_with_chatml
import gc # Garbage collector
#import function from folder above
import sys
import requests
import os
import glob
import re
import threading
from typing import List, Optional
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_PAGES_TO_CRAWL = 100
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
MODEL_NAME = "jinaai/jina-embeddings-v3"
# FAISS_INDEX_PATH = "vector_store.faiss" # Optional: path to save/load index - Will be dynamic
# METADATA_PATH = "metadata.json"        # Optional: path to save/load chunk metadata - Will be dynamic
MAX_CONTEXT_CHARS = 24000  # Cap context passed to the LLM (~3-4k tokens depending on text)
TOP_K_CHUNKS = 12           # Number of top chunks to include in the context

# --- Global State (In-memory - Suitable for simple demos) ---
# Warning: This will be reset if the server restarts.
# For production, consider persistent storage (DB, files)
vector_index = None
chunk_metadata = [] # List to store {"chunk_text": "...", "source_url": "..."}
is_ready = False
progress_queue = queue.Queue() # Queue for SSE updates

# --- Helper Functions (Your provided functions adapted) ---


def to_aera(messages, tools=[], model="and emili/aera/aera4b-instruct-q4_k_m.gguf", stream=False, max_tokens=32768, repeat_penalty=1.1, 
              top_p=0.95, top_k=40, stop="<|im_end|>", temp=0, response_format=None):
    prompt=format_with_chatml(messages, add_generation_prompt=True, tools=tools)
    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": repeat_penalty - 1.0,  # Converting repeat_penalty
        "stream": stream,
    }

    # add "response_format": response_format if it is not None
    if response_format != None:
        data["response_format"] = response_format
    print("response_format", response_format)
    # Add optional parameters if provided
    if stop:
        data["stop"] = stop

    response = requests.post('http://localhost:1234/v1/completions', 
                            headers=headers, 
                            json=data, 
                            stream=stream)
    
    if stream:
        return stream_response2(response)
    else:
        response_data = response.json()
        return response_data['choices'][0]['text']

def stream_response2(response):
    """Process a streaming response from the completions API."""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                json_str = line[6:]  # Remove 'data: ' prefix
                if json_str.strip() == '[DONE]':
                    break
                try:
                    chunk = json.loads(json_str)
                    content = chunk.get('choices', [{}])[0].get('text', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue


async def stream_aera_response(messages, tools=None, **kwargs):
    """Stream model output without blocking the event loop."""
    loop = asyncio.get_running_loop()
    message_queue: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for chunk in to_aera(messages, tools=tools or [], stream=True, **kwargs):
                if chunk:
                    loop.call_soon_threadsafe(message_queue.put_nowait, ("chunk", chunk))
            loop.call_soon_threadsafe(message_queue.put_nowait, ("end", None))
        except Exception as exc:  # pragma: no cover - defensive
            loop.call_soon_threadsafe(message_queue.put_nowait, ("error", str(exc)))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, payload = await message_queue.get()
        if kind == "chunk":
            if payload is None:
                continue
            yield payload
        elif kind == "end":
            break
        elif kind == "error":
            raise RuntimeError(payload or "Unknown streaming error")

def get_project_name_from_url(url: str) -> str:
    """Derives a sanitized project name from a URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    # Remove port if present
    domain = domain.split(':')[0]

    name = domain
    if name.startswith("www."):
        name = name[4:]
    
    project_base_name = name.split('.')[0]
    
    # Sanitize: keep alphanumeric, hyphen, underscore. Replace others.
    project_base_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project_base_name)
    if not project_base_name: # Handle cases like "..." -> "_" resulting in empty string
        project_base_name = "default_project"
    return project_base_name.lower()

def chunk_text_by_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunks text into overlapping segments based on word count."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start_index = 0
    while start_index < len(words):
        end_index = min(start_index + chunk_size, len(words))
        chunk = " ".join(words[start_index:end_index])
        chunks.append(chunk)
        start_index += chunk_size - overlap
        if start_index >= len(words):
             break
        # Ensure progress even if overlap is large relative to chunk size
        if start_index <= end_index - chunk_size:
            start_index = end_index - chunk_size + 1

    # Add the very last bit if it wasn't captured
    # Check if the last computed end_index covered the whole text
    if end_index < len(words) and start_index < len(words):
        last_chunk = " ".join(words[start_index:])
        if last_chunk: # Ensure it's not empty
             chunks.append(last_chunk)


    # Simple deduplication
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)

    return unique_chunks

def find_urls_from_sitemap(base_url: str) -> List[str]:
    """
    Given a base URL, find and parse sitemaps to extract all URLs.

    Args:
        base_url: The base URL of the website

    Returns:
        A list of URLs found in the sitemaps, or empty list if none found
    """
    # Ensure the base URL has a scheme
    if not base_url.startswith(('http://', 'https://')):
        base_url = 'https://' + base_url

    # Remove trailing slash if present
    base_url = base_url.rstrip('/')

    all_urls = []

    # Method 1: Check robots.txt for sitemap declarations
    sitemap_urls = _find_sitemaps_from_robots_txt(base_url)
    for sitemap_url in sitemap_urls:
        urls = _extract_urls_from_sitemap(sitemap_url, base_url)
        all_urls.extend(urls)

    # Method 2: Try common sitemap locations
    common_locations = [
        '/sitemap.xml',
        '/sitemap_index.xml',
        '/sitemap/sitemap.xml',
        '/wp-sitemap.xml',  # WordPress
        '/sitemap.php',
        '/sitemaps/sitemap.xml',
        '/sitemap.xml.gz',
        '/sitemap.gz',
        '/admin/config/search/xmlsitemap',  # Drupal
        '/sitemap/index.xml',
        '/sitemap1.xml',
        '/sitemap0.xml',
        '/feed/sitemap.xml',
        '/sitemap/index',  # Some custom implementations
        '/sitemaps.xml',
        '/sitemap-index.xml',
        '/xmlsitemap.xml',
        '/sitemapindex.xml',
        '/sitemap/sitemap-index.xml'
    ]

    for location in common_locations:
        sitemap_url = base_url + location
        urls = _extract_urls_from_sitemap(sitemap_url, base_url)
        all_urls.extend(urls)

    # Method 3: Try alternative naming patterns
    alternative_patterns = [
        '/sitemap_{}.xml',
        '/sitemap-{}.xml',
        '/sitemap{}.xml',
        '/sitemaps/{}.xml',
        '/{}-sitemap.xml'
    ]

    # Try patterns with common identifiers
    identifiers = ['main', 'posts', 'pages', 'categories', 'tags', 'products', 'services']
    for pattern in alternative_patterns:
        for identifier in identifiers:
            sitemap_url = base_url + pattern.format(identifier)
            urls = _extract_urls_from_sitemap(sitemap_url, base_url)
            all_urls.extend(urls)

    # Method 4: Try to find sitemap links from the homepage
    try:
        print("Trying to find sitemap links from homepage...")
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            sitemap_links = _extract_sitemap_links_from_html(response.text, base_url)
            for sitemap_url in sitemap_links:
                urls = _extract_urls_from_sitemap(sitemap_url, base_url)
                all_urls.extend(urls)
    except requests.RequestException as e:
        print(f"Error checking homepage: {e}")

    # Remove duplicates and return
    unique_urls = list(set(all_urls))
    print(f"Found {len(unique_urls)} unique URLs from sitemaps")
    return unique_urls

def _extract_urls_from_sitemap(sitemap_url: str, base_url: str) -> List[str]:
    """Extract all URLs from a sitemap file."""
    urls = []
    try:
        print(f"Fetching sitemap: {sitemap_url}")
        response = requests.get(sitemap_url, timeout=15)
        if response.status_code == 200:
            try:
                # Parse XML content
                root = ET.fromstring(response.content)

                # Handle sitemap index files (which contain links to other sitemaps)
                if root.tag.endswith('sitemapindex'):
                    # Find all <sitemap> elements and extract their <loc> children
                    for sitemap_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                        loc_elem = sitemap_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                        if loc_elem is not None and loc_elem.text:
                            # Recursively extract URLs from nested sitemaps
                            nested_urls = _extract_urls_from_sitemap(loc_elem.text.strip(), base_url)
                            urls.extend(nested_urls)
                else:
                    # Handle regular sitemap files
                    # Find all <url> elements and extract their <loc> children
                    for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                        loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                        if loc_elem is not None and loc_elem.text:
                            url = loc_elem.text.strip()
                            if url.startswith('http'):
                                urls.append(url)
                            else:
                                urls.append(urljoin(base_url, url))

                print(f"Extracted {len(urls)} URLs from {sitemap_url}")
            except ET.ParseError as e:
                print(f"Error parsing XML from {sitemap_url}: {e}")
                # Try alternative parsing with BeautifulSoup as fallback
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'xml')
                    loc_elements = soup.find_all('loc')
                    for loc_elem in loc_elements:
                        url = loc_elem.text.strip()
                        if url.startswith('http'):
                            urls.append(url)
                        else:
                            urls.append(urljoin(base_url, url))
                    print(f"Extracted {len(urls)} URLs from {sitemap_url} using BeautifulSoup fallback")
                except Exception as bs_error:
                    print(f"BeautifulSoup fallback also failed for {sitemap_url}: {bs_error}")
        else:
            print(f"Sitemap not found at {sitemap_url} (status: {response.status_code})")
    except requests.RequestException as e:
        print(f"Error fetching {sitemap_url}: {e}")

    return urls

def _find_sitemaps_from_robots_txt(base_url: str) -> List[str]:
    """Extract sitemap URLs from robots.txt file."""
    sitemap_urls = []
    try:
        robots_url = base_url + '/robots.txt'
        print(f"Checking robots.txt: {robots_url}")
        response = requests.get(robots_url, timeout=10)
        if response.status_code == 200:
            for line in response.text.split('\n'):
                line = line.strip()
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line[8:].strip()
                    if sitemap_url:
                        sitemap_urls.append(sitemap_url)
                        print(f"Found sitemap in robots.txt: {sitemap_url}")
    except requests.RequestException as e:
        print(f"Error checking robots.txt: {e}")

    return sitemap_urls

def _extract_sitemap_links_from_html(html_content: str, base_url: str) -> List[str]:
    """Extract sitemap URLs from HTML content using basic parsing."""
    sitemap_urls = []

    # Look for sitemap links in various formats
    patterns = [
        r'href=["\']([^"\']*sitemap[^"\']*\.xml?)["\']',
        r'href=["\']([^"\']*\.xml[^"\']*sitemap[^"\']*)["\']',
        r'href=["\']([^"\']*sitemap[^"\']*)["\']'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            if match.startswith('http'):
                sitemap_urls.append(match)
            elif match.startswith('/'):
                sitemap_urls.append(base_url + match)
            else:
                sitemap_urls.append(base_url + '/' + match)

    return list(set(sitemap_urls))  # Remove duplicates

# --- Load Model (Done once at startup) ---
logger.info(f"Loading sentence transformer model: {MODEL_NAME}...")
try:
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Sentence Transformer model: {e}")
    # Depending on requirements, you might want to exit or run without embedding
    raise RuntimeError(f"Fatal error: Could not load embedding model {MODEL_NAME}") from e


# --- FastAPI App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static") # If you have static files

# --- Pydantic Models ---
class CrawlRequest(BaseModel):
    url: str

# --- Background Task Functions ---

async def send_progress(percent: int, message: str, status: str = "progress"):
    """Helper to put progress update into the SSE queue."""
    progress_queue.put(json.dumps({
        "status": status,
        "percent": percent,
        "message": message
    }))

async def crawl_and_index_task(start_url: str):
    """The background task performing crawling, embedding, and indexing."""
    global vector_index, chunk_metadata, is_ready

    project_name = get_project_name_from_url(start_url)
    logger.info(f"Starting crawl task for: {start_url} (Project: {project_name})")
    await send_progress(0, f"Initializing crawler for {project_name}...")

    # --- Reset state for new crawl ---
    vector_index = None
    chunk_metadata = []
    is_ready = False
    gc.collect() # Suggest garbage collection before intensive task

    visited_urls = set()
    urls_to_crawl = asyncio.Queue()
    pages_processed = 0
    all_page_data = [] # List to hold {"text": "...", "url": "..."}

    # First, try to get URLs from sitemaps
    await send_progress(5, "Checking for sitemaps...")
    logger.info("Attempting to find URLs from sitemaps...")
    sitemap_urls = find_urls_from_sitemap(start_url)

    if sitemap_urls:
        logger.info(f"Found {len(sitemap_urls)} URLs from sitemaps")
        await send_progress(10, f"Found {len(sitemap_urls)} URLs from sitemaps, starting crawl...")

        # Add sitemap URLs to the crawl queue (limit to avoid overwhelming)
        urls_added = 0
        for url in sitemap_urls:
            if urls_added >= MAX_PAGES_TO_CRAWL:
                break
            if url not in visited_urls:
                await urls_to_crawl.put(url)
                visited_urls.add(url)
                urls_added += 1

        logger.info(f"Added {urls_added} URLs from sitemaps to crawl queue")
    else:
        # Fallback to original behavior - start with the provided URL
        logger.info("No sitemaps found, starting with provided URL")
        await urls_to_crawl.put(start_url)
        visited_urls.add(start_url)

    # Use httpx.AsyncClient for efficient async requests
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        while not urls_to_crawl.empty() and pages_processed < MAX_PAGES_TO_CRAWL:
            current_url = await urls_to_crawl.get()
            logger.info(f"Processing URL ({pages_processed + 1}/{MAX_PAGES_TO_CRAWL}): {current_url}")
            await send_progress(
                int((pages_processed / MAX_PAGES_TO_CRAWL) * 30), # Crawling is ~30%
                f"Crawling page {pages_processed + 1}/{MAX_PAGES_TO_CRAWL}: {current_url}"
            )

            try:
                response = await client.get(current_url)
                response.raise_for_status() # Raise exception for 4xx/5xx status codes

                # Basic content type check
                content_type = response.headers.get("content-type", "").lower()
                if "html" not in content_type:
                    logger.warning(f"Skipping non-HTML content at {current_url} (type: {content_type})")
                    urls_to_crawl.task_done()
                    continue # Skip non-html pages

                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract text (simple version - might need refinement)
                page_text = ' '.join(soup.stripped_strings) # Get text nodes and join
                if not page_text:
                    page_text = soup.get_text(separator=' ', strip=True) # Fallback

                if page_text:
                    all_page_data.append({"text": page_text, "url": current_url})
                else:
                     logger.warning(f"No text extracted from {current_url}")


                pages_processed += 1

                # Find and add new links (simple domain check)
                base_domain = urlparse(start_url).netloc
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(current_url, href)
                    parsed_url = urlparse(absolute_url)

                    # Basic validation and stay within domain/subdomain
                    # Also, ignore common non-HTML file extensions
                    file_path = parsed_url.path.lower()
                    ignored_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.rar', '.tar', '.gz', '.mp3', '.mp4', '.avi', '.mov', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.css', '.js', '.xml', '.json', '.svg', '.webp')
                    if (parsed_url.scheme in ['http', 'https'] and
                        parsed_url.netloc and
                        (parsed_url.netloc == base_domain or parsed_url.netloc.endswith('.' + base_domain)) and
                        absolute_url not in visited_urls and
                        not file_path.endswith(ignored_extensions) and # Ignore specified file extensions
                        urls_to_crawl.qsize() + pages_processed < MAX_PAGES_TO_CRAWL * 2 # Limit queue size
                        ):
                        await urls_to_crawl.put(absolute_url)
                        visited_urls.add(absolute_url)

            except httpx.RequestError as e:
                logger.error(f"HTTP error crawling {current_url}: {e}")
                await send_progress(
                    int((pages_processed / MAX_PAGES_TO_CRAWL) * 30),
                    f"Error crawling {current_url}: {e}",
                     status="error" # Send error status if needed by JS
                )
                 # Decide whether to stop the whole crawl on error or just skip the page
                 # For now, we just skip.
            except Exception as e:
                 logger.error(f"Unexpected error processing {current_url}: {e}")
                 await send_progress(
                     int((pages_processed / MAX_PAGES_TO_CRAWL) * 30),
                     f"Error processing {current_url}: {e}",
                     status="error"
                 )

            finally:
                urls_to_crawl.task_done()
                await asyncio.sleep(0.1) # Small delay to prevent overwhelming server


    logger.info(f"Crawling finished. Found text from {len(all_page_data)} pages.")
    if not all_page_data:
        logger.error("No text data extracted from any page. Stopping.")
        await send_progress(100, "Failed: No text could be extracted.", status="error")
        return # Exit the task

    # --- 2. Chunking ---
    await send_progress(35, "Starting text chunking...")
    logger.info("Chunking text...")
    all_chunks_with_source = []
    combined_text_length = sum(len(page['text']) for page in all_page_data)
    processed_length = 0

    for i, page_data in enumerate(all_page_data):
        page_text = page_data["text"]
        source_url = page_data["url"]
        chunks = chunk_text_by_words(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
             # Store chunk text and its original URL
            all_chunks_with_source.append({"chunk_text": chunk, "source_url": source_url})

        processed_length += len(page_text)
        chunking_progress = int((processed_length / combined_text_length) * 25) # Chunking is ~25%
        await send_progress(35 + chunking_progress, f"Chunking page {i+1}/{len(all_page_data)}")
        await asyncio.sleep(0.01) # Yield control briefly


    if not all_chunks_with_source:
        logger.error("No chunks generated from the text. Stopping.")
        await send_progress(100, "Failed: Could not generate text chunks.", status="error")
        return

    # --- Simple Deduplication of chunks ---
    seen_chunks = set()
    unique_chunks_with_source = []
    for item in all_chunks_with_source:
        if item['chunk_text'] not in seen_chunks:
            unique_chunks_with_source.append(item)
            seen_chunks.add(item['chunk_text'])

    chunk_metadata = unique_chunks_with_source # Store metadata globally
    logger.info(f"Generated {len(chunk_metadata)} unique chunks.")
    await send_progress(60, f"Generated {len(chunk_metadata)} unique chunks. Starting embedding...")


    # --- 3. Embedding ---
    logger.info("Embedding chunks...")
    chunk_texts = [item['chunk_text'] for item in chunk_metadata]
    total_chunks_to_embed = len(chunk_texts)
    
    # Initialize embeddings as an empty array with the correct dimension
    # This ensures 'embeddings' is always defined, even if no chunks to embed.
    embeddings = np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    if total_chunks_to_embed > 0:
        try:
            embeddings_list = []
            # Define a batch size for encoding and progress updates.
            # This determines how frequently progress is reported.
            progress_batch_size = 32  # Tunable parameter

            for i in range(0, total_chunks_to_embed, progress_batch_size):
                batch_chunk_texts = chunk_texts[i:i+progress_batch_size]
                
                if not batch_chunk_texts: # Should ideally not happen with correct loop logic
                    continue

                # Run the synchronous model.encode in a separate thread
                # to avoid blocking the asyncio event loop.
                # Pass the batch of texts to encode.
                # show_progress_bar=False for these individual calls as we're sending SSE.
                # The batch_size for model.encode itself can be len(batch_chunk_texts)
                # to process the given batch.
                batch_embeddings = await asyncio.to_thread(
                    model.encode,
                    batch_chunk_texts,
                    task="retrieval.passage",
                    prompt_name="retrieval.passage",
                    show_progress_bar=False,
                    batch_size=len(batch_chunk_texts) 
                )
                embeddings_list.append(batch_embeddings)

                processed_chunks = min(i + len(batch_chunk_texts), total_chunks_to_embed)
                
                # Calculate progress within the embedding phase (0-100%)
                embedding_phase_completion_ratio = processed_chunks / total_chunks_to_embed
                
                # Embedding phase contributes 35% to total progress (from 60% to 95%)
                current_total_progress = 60 + int(embedding_phase_completion_ratio * 35)
                
                await send_progress(
                    current_total_progress,
                    f"Embedding: {processed_chunks}/{total_chunks_to_embed} chunks ({current_total_progress}%)"
                )
                await asyncio.sleep(0.01) # Yield control briefly

            if embeddings_list:
                embeddings = np.concatenate(embeddings_list, axis=0)
            # If embeddings_list is empty, 'embeddings' remains the initialized empty array.

        except Exception as e:
            logger.error(f"Error during embedding: {e}")
            await send_progress(100, f"Failed: Error during text embedding: {e}", status="error")
            chunk_metadata = [] # Clear potentially partial metadata
            vector_index = None # Ensure vector_index is also cleared
            is_ready = False
            return # Exit the task
    else:
        logger.info("No chunks to embed. Skipping embedding phase.")
        # If embedding is skipped, progress should jump to the start of the next phase (95%)
        await send_progress(95, "No chunks to embed. Proceeding to indexing.")


    # --- 4. FAISS Indexing ---
    logger.info("Building FAISS index...")
    # This progress message marks the beginning of the FAISS indexing phase.
    # If embedding was skipped, we've already set progress to 95%.
    # If embedding happened, it should have concluded around 95%.
    await send_progress(95, "Building search index...")
    
    # Get embedding dimension directly from the model, safer if 'embeddings' array is empty.
    dimension = model.get_sentence_embedding_dimension()
    
    # Using IndexFlatIP for cosine similarity after normalization.
    index = faiss.IndexFlatIP(dimension)
    
    # Check if there are embeddings to add to the index.
    # embeddings.ndim == 2 ensures it's a 2D array.
    # embeddings.shape[0] > 0 ensures there's at least one vector.
    if embeddings.ndim == 2 and embeddings.shape[0] > 0:
        # Ensure embeddings are float32, SentenceTransformer usually returns float32.
        # np.array creates a copy if not already float32, which is fine.
        embeddings_float32 = np.array(embeddings, dtype='float32')
        
        # Normalize embeddings for IndexFlatIP. This is crucial for cosine similarity.
        # faiss.normalize_L2 modifies the array in-place.
        faiss.normalize_L2(embeddings_float32)
        
        index.add(embeddings_float32) # Add normalized embeddings to the index
        logger.info(f"FAISS index populated with {index.ntotal} vectors.")
    else:
        logger.info("FAISS index built with 0 vectors as no embeddings were provided or generated.")

    vector_index = index # Store index globally
    is_ready = True # Mark the system as ready for chat
    logger.info(f"FAISS index built successfully for project {project_name}. Total vectors: {index.ntotal}.")

    # Save index and metadata
    PROJECT_FAISS_PATH = f"{project_name}.faiss"
    PROJECT_METADATA_PATH = f"{project_name}.json"
    try:
        logger.info(f"Attempting to save index to {PROJECT_FAISS_PATH} and metadata to {PROJECT_METADATA_PATH}")
        # Run synchronous I/O in a separate thread
        await asyncio.to_thread(faiss.write_index, vector_index, PROJECT_FAISS_PATH)
        with open(PROJECT_METADATA_PATH, 'w') as f:
            await asyncio.to_thread(json.dump, chunk_metadata, f)
        logger.info(f"Saved FAISS index to {PROJECT_FAISS_PATH} and metadata to {PROJECT_METADATA_PATH}")
    except Exception as e:
        logger.error(f"Error saving index/metadata for project {project_name}: {e}")
        # Potentially send a specific progress update about save failure
        await send_progress(100, f"Crawling finished, but failed to save project '{project_name}': {e}", status="error")
        # Note: is_ready is True, system might be usable but not saved.


    await send_progress(100, f"Crawling and indexing for '{project_name}' finished!", status="completed")
    logger.info(f"Background task for project '{project_name}' completed.")


# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-crawl")
async def start_crawl_endpoint(crawl_request: CrawlRequest, background_tasks: BackgroundTasks):
    """Starts the background crawling and indexing task."""
    global is_ready, vector_index, chunk_metadata
    # Basic validation
    parsed_url = urlparse(crawl_request.url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
         return JSONResponse(status_code=400, content={"detail": "Invalid URL provided."})

    project_name = get_project_name_from_url(crawl_request.url)
    logger.info(f"Received crawl request for: {crawl_request.url} (Project: {project_name})")

    # Clear previous results immediately if a new crawl starts
    is_ready = False
    vector_index = None
    chunk_metadata = []
    gc.collect()
    # Empty the progress queue
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except queue.Empty:
            break
    logger.info(f"Cleared state for new crawl/load of project: {project_name}")


    background_tasks.add_task(crawl_and_index_task, crawl_request.url)
    return {"status": "Crawl started", "project_name": project_name}


# Server-Sent Events for Progress Updates
@app.get("/crawl-progress")
async def crawl_progress_sse():
    """Endpoint for Server-Sent Events to stream progress updates."""
    from fastapi.responses import StreamingResponse
    import asyncio

    async def event_stream():
        while True:
            try:
                # Use get_nowait() or timeout to prevent blocking indefinitely
                # If using asyncio, a dedicated async queue would be better.
                # For simplicity with standard queue, we check frequently.
                message = progress_queue.get_nowait()
                yield f"data: {message}\n\n"
                # Check if the message indicates completion or error to stop streaming
                data = json.loads(message)
                if data.get("status") in ["completed", "error"]:
                    logger.info(f"SSE stream closing due to status: {data.get('status')}")
                    break
            except queue.Empty:
                await asyncio.sleep(0.2) # Wait a bit before checking again
            except Exception as e:
                logger.error(f"Error in SSE stream: {e}")
                yield f"data: {json.dumps({'status': 'error', 'message': 'SSE internal error'})}\n\n"
                break # Stop streaming on error

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for the chat interface."""
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    async def send_json_message(payload: dict):
        await websocket.send_text(json.dumps(payload))

    if not is_ready or vector_index is None or not chunk_metadata:
        await send_json_message({
            "type": "response_error",
            "message": "Sorry, the system is not ready. Please crawl a website first."
        })
        await websocket.close()
        logger.warning("WebSocket closed because system is not ready.")
        return

    try:
        while True:
            query = await websocket.receive_text()
            logger.info(f"Received query: {query}")

            if not query:
                continue

            # The old prompt structure is no longer needed with Jina v3's task/prompt_name
            # task = 'Given a question, retrieve Wikipedia passages that answer the question'
            # prompted_query = f"Instruct: {task}\nQuery: {query}"

            try:
                # 1. Embed the query
                # Use "retrieval_query" for the task and prompt_name
                query_embedding = model.encode(
                    [query], # Pass the raw query
                    task="retrieval.query",
                    prompt_name="retrieval.query",
                    show_progress_bar=False,
                    batch_size=1
                )
                # Normalize query embedding for IP search
                faiss.normalize_L2(query_embedding)


                # 2. Search the FAISS index for top chunks
                k = max(1, min(TOP_K_CHUNKS, vector_index.ntotal))
                distances, indices = vector_index.search(np.array(query_embedding, dtype='float32'), k)

                # 3. Build context from top chunks (not whole pages), with a char budget
                selected_chunks = []
                seen_texts = set()
                total_chars = 0
                unique_urls = set()

                if indices.size > 0:
                    top_indices = indices[0]
                    for idx in top_indices:
                        if 0 <= idx < len(chunk_metadata):
                            meta = chunk_metadata[idx]
                            chunk_text = (meta.get('chunk_text') or '').strip()
                            source_url = meta.get('source_url') or ''
                            if not chunk_text or chunk_text in seen_texts:
                                continue
                            projected_chars = total_chars + len(chunk_text)
                            if projected_chars > MAX_CONTEXT_CHARS:
                                break
                            selected_chunks.append({"text": chunk_text, "url": source_url})
                            seen_texts.add(chunk_text)
                            total_chars = projected_chars
                            if source_url:
                                unique_urls.add(source_url)
                        else:
                            logger.warning(f"Invalid index {idx} returned from FAISS search.")

                # 4. Send the results back to the client
                if selected_chunks:
                    logger.info(f"Selected {len(selected_chunks)} chunk(s) from {len(unique_urls)} URL(s) for context.")

                    context_parts = []
                    for i, ch in enumerate(selected_chunks, 1):
                        context_parts.append(f"[{i}] Source: {ch['url']}\n{ch['text']}")
                    context_str = "\n\n---\n\n".join(context_parts)

                    system_prompt = (
                        "Sei Aera 4B. Rispondi in italiano usando esclusivamente il CONTENUTO seguente. "
                        "Cita i riferimenti tra parentesi quadre [n] quando possibile. "
                        "Se l'informazione non è presente, dichiara che non è presente.\n\n"
                        "Formatta la risposta in markdown.\n\n"
                        f"CONTENUTO:\n{context_str}"
                    )
                    await send_json_message({"type": "response_start"})

                    try:
                        async for chunk in stream_aera_response([
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ]):
                            await send_json_message({
                                "type": "response_chunk",
                                "content": chunk,
                            })
                        await send_json_message({"type": "response_end"})
                    except Exception as stream_error:
                        logger.error(f"Error streaming model response: {stream_error}")
                        await send_json_message({
                            "type": "response_error",
                            "message": f"An error occurred while generating the answer: {stream_error}",
                        })

                else:
                    logger.info("No relevant chunks found.")
                    await send_json_message({
                        "type": "info",
                        "message": "Sorry, I couldn't find a relevant page for that query in the crawled content."
                    })

            except Exception as e:
                 logger.error(f"Error processing chat query '{query}': {e}")
                 await send_json_message({
                     "type": "response_error",
                     "message": f"An error occurred while processing your request: {e}"
                 })


    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        try:
             # Try to inform the client about the error before closing
             await send_json_message({
                 "type": "response_error",
                 "message": f"A server error occurred: {e}"
             })
             await websocket.close(code=1011) # Internal Error code
        except:
             pass # Ignore errors during cleanup


# --- Project Listing and Loading ---
def list_available_projects():
    """Scans for .faiss files and returns a list of project names."""
    # Assumes CWD is where main.py is, and where projects are saved.
    project_files = glob.glob("*.faiss")
    project_names = sorted([os.path.splitext(os.path.basename(f))[0] for f in project_files])
    logger.info(f"Found projects: {project_names}")
    return project_names

@app.get("/list-projects", response_model=list[str])
async def get_list_projects_endpoint():
    """Endpoint to get a list of saved project names."""
    return list_available_projects()

async def load_project_data(project_name: str):
    """Loads a project's FAISS index and metadata. Returns (success_bool, message_str)."""
    global vector_index, chunk_metadata, is_ready
    
    project_faiss_path = f"{project_name}.faiss"
    project_metadata_path = f"{project_name}.json"

    if not os.path.exists(project_faiss_path):
        logger.error(f"FAISS index file not found: {project_faiss_path}")
        return False, f"Index file for '{project_name}' not found."
    if not os.path.exists(project_metadata_path):
        logger.error(f"Metadata file not found: {project_metadata_path}")
        return False, f"Metadata file for '{project_name}' not found."

    try:
        logger.info(f"Loading project '{project_name}' from {project_faiss_path} and {project_metadata_path}...")
        
        # Load FAISS index (synchronous I/O, run in thread)
        loaded_index = await asyncio.to_thread(faiss.read_index, project_faiss_path)
        
        # Load metadata (synchronous I/O, run in thread)
        with open(project_metadata_path, 'r') as f:
            # json.load is synchronous, wrap it if it could block significantly
            # For typical metadata sizes, it might be fine, but for consistency:
            loaded_metadata = await asyncio.to_thread(json.load, f)
        
        vector_index = loaded_index
        chunk_metadata = loaded_metadata
        is_ready = True
        
        logger.info(f"Successfully loaded project '{project_name}'. Index vectors: {vector_index.ntotal}, Metadata entries: {len(chunk_metadata)}")
        return True, f"Project '{project_name}' loaded successfully."
    except Exception as e:
        logger.error(f"Error loading project '{project_name}': {e}")
        vector_index = None # Ensure state is clean on error
        chunk_metadata = []
        is_ready = False
        return False, f"Error loading project '{project_name}': {str(e)}"

@app.post("/load-project/{project_name_param}")
async def load_project_endpoint(project_name_param: str):
    """Loads an existing project by its name."""
    global is_ready, vector_index, chunk_metadata # For clearing state
    
    logger.info(f"Received request to load project: {project_name_param}")

    # Clear previous state before loading
    is_ready = False
    vector_index = None
    chunk_metadata = []
    gc.collect() # Suggest garbage collection
    # Empty the progress queue as it's for crawling, not loading
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except queue.Empty:
            break
    logger.info(f"Cleared state before attempting to load project: {project_name_param}")

    success, message = await load_project_data(project_name_param)
    
    if success:
        # Send a progress update to notify client if necessary, or rely on HTTP response.
        # For simplicity, client will handle UI update based on this HTTP response.
        return {"status": "Project loaded", "project_name": project_name_param, "message": message}
    else:
        return JSONResponse(status_code=500, content={"status": "Error loading project", "project_name": project_name_param, "detail": message})


# --- Test function for sitemap functionality ---
def test_sitemap_functionality():
    """Test the sitemap URL extraction functionality."""
    test_urls = [
        "https://example.com",
        "https://github.com",
        "https://www.python.org"
    ]

    print("Testing sitemap URL extraction...")
    for test_url in test_urls:
        print(f"\nTesting: {test_url}")
        try:
            urls = find_urls_from_sitemap(test_url)
            print(f"Found {len(urls)} URLs")
            if urls:
                print("Sample URLs:")
                for i, url in enumerate(urls[:5]):  # Show first 5 URLs
                    print(f"  {i+1}. {url}")
                if len(urls) > 5:
                    print(f"  ... and {len(urls) - 5} more")
        except Exception as e:
            print(f"Error testing {test_url}: {e}")

# --- Main execution (for running with uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    import sys

    # If test argument is provided, run tests instead of starting server
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_sitemap_functionality()
    else:
        logger.info("Starting FastAPI server...")
        # Reload=True is useful for development but might interfere with background tasks
        # Use reload=False for more stable background task execution
        uvicorn.run(app, host="0.0.0.0", port=8000)
        # For production use: uvicorn main:app --host 0.0.0.0 --port 8000
