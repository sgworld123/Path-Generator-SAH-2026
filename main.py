from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Optional
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np
import google.generativeai as genai
import json
import re
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from urllib.parse import urlparse

app = FastAPI()

# -------------------------
# INITIALIZATION
# -------------------------
genai.configure(api_key="")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

kw_model = KeyBERT()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

difficulty_cache: dict = {}
executor = ThreadPoolExecutor(max_workers=4)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LearningPathBot/1.0)"}


# -------------------------
# URL HELPERS
# -------------------------
def is_github_url(url: str) -> bool:
    return "github.com" in urlparse(url).netloc


def github_readme_raw_url(url: str) -> str:
    """
    Convert any GitHub repo URL to its raw README URL.
    e.g. https://github.com/user/repo  ->  https://raw.githubusercontent.com/user/repo/HEAD/README.md
    """
    path = urlparse(url).path.strip("/")           # "user/repo" or "user/repo/tree/branch"
    parts = path.split("/")
    user, repo = parts[0], parts[1]
    return f"https://raw.githubusercontent.com/{user}/{repo}/HEAD/README.md"


async def fetch_blog_text(url: str) -> str:
    """Fetch a blog/article URL and extract readable text via BeautifulSoup."""
    async with httpx.AsyncClient(headers=HEADERS, timeout=15, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Try common article containers first, fall back to full body
    for selector in ["article", "main", ".post-content", ".entry-content", ".content", "body"]:
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return text

    return soup.get_text(separator=" ", strip=True)


async def fetch_github_readme(url: str) -> str:
    """Fetch README.md raw text from a GitHub repo URL."""
    raw_url = github_readme_raw_url(url)
    async with httpx.AsyncClient(headers=HEADERS, timeout=15, follow_redirects=True) as client:
        resp = await client.get(raw_url)
        if resp.status_code == 404:
            # Try README.rst as fallback
            raw_url_rst = raw_url.replace("README.md", "README.rst")
            resp = await client.get(raw_url_rst)
        resp.raise_for_status()
    return resp.text


async def fetch_url_text(url: str) -> tuple[str, str]:
    """
    Auto-detect URL type and return (name, text).
    Returns a clean name derived from the URL slug.
    """
    if is_github_url(url):
        text = await fetch_github_readme(url)
        # Name = "user/repo"
        parts = urlparse(url).path.strip("/").split("/")
        name = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else url
    else:
        text = await fetch_blog_text(url)
        # Name = last path segment or domain
        path = urlparse(url).path.strip("/")
        slug = path.split("/")[-1] or urlparse(url).netloc
        # Clean slug: remove file extensions, replace hyphens/underscores
        name = re.sub(r'\.[a-z]{2,4}$', '', slug).replace("-", " ").replace("_", " ") or url
    return name, text


# -------------------------
# CONCEPT EXTRACTION
# -------------------------
def extract_concepts(text: str, top_n: int = 10) -> list:
    text = text[:3000]
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=False,
        top_n=top_n
    )
    seen, unique = set(), []
    for kw, _ in keywords:
        c = kw.lower().strip()
        if c not in seen and len(c) > 2:
            seen.add(c)
            unique.append(c)
    return unique


# -------------------------
# GEMINI DIFFICULTY SCORING
# -------------------------
def _call_gemini_sync(prompt: str) -> str:
    return gemini_model.generate_content(prompt).text


async def get_difficulty_score(name: str, text: str) -> int:
    cache_key = text[:500]
    if cache_key in difficulty_cache:
        return difficulty_cache[cache_key]

    prompt = f"""Rate the technical difficulty of the following content from 1 to 10.
Return ONLY a single integer, nothing else.

Content:
{text[:1500]}"""

    loop = asyncio.get_event_loop()
    try:
        raw = await asyncio.wait_for(
            loop.run_in_executor(executor, _call_gemini_sync, prompt),
            timeout=15
        )
        score = int(re.search(r'\d+', raw).group())
        score = max(1, min(score, 10))
    except Exception:
        score = 5

    difficulty_cache[cache_key] = score
    return score


# -------------------------
# SHARED PROCESSING PIPELINE
# -------------------------
async def process_source(name: str, text: str) -> tuple[str, dict]:
    """Given a name + raw text, extract concepts, embed, score difficulty."""
    if not text or len(text.strip()) < 50:
        raise ValueError(f"Content for '{name}' is too short or empty.")

    loop = asyncio.get_event_loop()
    concepts  = await loop.run_in_executor(executor, extract_concepts, text)
    embedding = await loop.run_in_executor(executor, embed_model.encode, text)
    difficulty = await get_difficulty_score(name, text)

    return name, {
        "concepts": concepts,
        "embedding": embedding.tolist(),
        "difficulty": difficulty,
    }


# -------------------------
# GRAPH CONSTRUCTION
# -------------------------
def build_dependency_graph(files_data: dict):
    names = list(files_data.keys())
    graph = {name: [] for name in names}
    indegree = {name: 0 for name in names}

    concept_embeddings = {
        name: embed_model.encode(" ".join(files_data[name]["concepts"]))
        for name in names
    }

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            f1, f2 = names[i], names[j]
            sim = cosine_similarity(
                concept_embeddings[f1].reshape(1, -1),
                concept_embeddings[f2].reshape(1, -1)
            )[0][0]

            if sim < 0.3:
                continue

            d1 = files_data[f1]["difficulty"]
            d2 = files_data[f2]["difficulty"]

            if d1 == d2:
                set1 = set(files_data[f1]["concepts"])
                set2 = set(files_data[f2]["concepts"])
                if not set1 & set2:
                    continue
                prerequisite = f1 if len(set1) <= len(set2) else f2
                dependent = f2 if prerequisite == f1 else f1
            else:
                prerequisite = f1 if d1 < d2 else f2
                dependent = f2 if prerequisite == f1 else f1

            if dependent not in graph[prerequisite]:
                graph[prerequisite].append(dependent)
                indegree[dependent] += 1

    return graph, indegree


# -------------------------
# CYCLE REMOVAL
# -------------------------
def remove_cycles(graph: dict, indegree: dict):
    from collections import deque
    indegree_copy = indegree.copy()
    queue = deque(n for n in indegree_copy if indegree_copy[n] == 0)
    visited = set()
    while queue:
        node = queue.popleft()
        visited.add(node)
        for nb in graph[node]:
            indegree_copy[nb] -= 1
            if indegree_copy[nb] == 0:
                queue.append(nb)

    cycle_nodes = set(graph.keys()) - visited
    for node in cycle_nodes:
        graph[node] = [nb for nb in graph[node] if nb not in cycle_nodes]

    new_indegree = {n: 0 for n in graph}
    for node in graph:
        for nb in graph[node]:
            new_indegree[nb] += 1

    return graph, new_indegree


# -------------------------
# TOPO SORT VARIANTS
# -------------------------
def topo_sort_easy_first(graph, indegree, difficulty):
    heap, result, ind = [], [], indegree.copy()
    for n in ind:
        if ind[n] == 0:
            heapq.heappush(heap, (difficulty[n], n))
    while heap:
        _, node = heapq.heappop(heap)
        result.append(node)
        for nb in graph[node]:
            ind[nb] -= 1
            if ind[nb] == 0:
                heapq.heappush(heap, (difficulty[nb], nb))
    return result

def topo_sort_hard_first(graph, indegree, difficulty):
    heap, result, ind = [], [], indegree.copy()
    for n in ind:
        if ind[n] == 0:
            heapq.heappush(heap, (-difficulty[n], n))
    while heap:
        _, node = heapq.heappop(heap)
        result.append(node)
        for nb in graph[node]:
            ind[nb] -= 1
            if ind[nb] == 0:
                heapq.heappush(heap, (-difficulty[nb], nb))
    return result

def topo_sort_balanced(graph, indegree, difficulty):
    available, result, ind = [], [], indegree.copy()
    for n in ind:
        if ind[n] == 0:
            available.append(n)
    while available:
        available.sort(key=lambda n: difficulty[n])
        node = available.pop(len(available) // 2)
        result.append(node)
        for nb in graph[node]:
            ind[nb] -= 1
            if ind[nb] == 0:
                available.append(nb)
    return result


# -------------------------
# GEMINI JSON PARSER
# -------------------------
def parse_gemini_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON object found in response")


# -------------------------
# PATH ANALYSIS
# -------------------------
async def analyze_paths(files_data: dict, paths: dict) -> dict:
    summary = {
        name: {
            "difficulty": files_data[name]["difficulty"],
            "concepts": files_data[name]["concepts"]
        }
        for name in files_data
    }

    prompt = f"""You are a curriculum designer. Analyze these learning paths.

Sources (name, difficulty 1-10, key concepts):
{json.dumps(summary, indent=2)}

Learning Paths (ordered sequences):
{json.dumps(paths, indent=2)}

For each path give pros, cons, best learner type, and a score 1-10.
Pick the best path and explain WHY, referencing specific source names and difficulty scores.

Return ONLY valid JSON, no markdown fences:
{{
  "easy_first": {{"pros": [], "cons": [], "best_for": "", "score": 0}},
  "hard_first": {{"pros": [], "cons": [], "best_for": "", "score": 0}},
  "balanced":   {{"pros": [], "cons": [], "best_for": "", "score": 0}},
  "best_path": "",
  "reason": ""
}}"""

    loop = asyncio.get_event_loop()
    try:
        raw = await asyncio.wait_for(
            loop.run_in_executor(executor, _call_gemini_sync, prompt),
            timeout=20
        )
        return parse_gemini_json(raw)
    except asyncio.TimeoutError:
        return {
            "analysis_error": "Gemini timed out",
            "best_path": "easy_first",
            "reason": "Defaulting to easy_first: progressive difficulty is best for most learners."
        }
    except Exception as e:
        return {"analysis_error": str(e)}


# -------------------------
# SHARED RESPONSE BUILDER
# -------------------------
async def build_response(files_data: dict) -> dict:
    graph, indegree = build_dependency_graph(files_data)
    graph, indegree = remove_cycles(graph, indegree)

    difficulty_map = {n: files_data[n]["difficulty"] for n in files_data}
    all_sources = set(files_data.keys())

    paths = {
        "easy_first": topo_sort_easy_first(graph, indegree.copy(), difficulty_map),
        "hard_first":  topo_sort_hard_first(graph, indegree.copy(), difficulty_map),
        "balanced":    topo_sort_balanced(graph, indegree.copy(), difficulty_map),
    }

    for path_name, path in paths.items():
        missing = sorted(all_sources - set(path), key=lambda n: difficulty_map[n])
        paths[path_name].extend(missing)

    analysis = await analyze_paths(files_data, paths)

    return {
        "sources": {
            n: {"difficulty": files_data[n]["difficulty"], "concepts": files_data[n]["concepts"]}
            for n in files_data
        },
        "graph": graph,
        "difficulty_scores": difficulty_map,
        "paths": paths,
        "analysis": analysis,
    }


# -------------------------
# SINGLE ENDPOINT: files + URLs via form-data
# -------------------------
@app.post("/generate-paths/")
async def generate_paths(
    files: Optional[List[UploadFile]] = File(default=None),
    urls: Optional[List[str]] = Form(default=None),
):
    """
    Send files and/or URLs in the same multipart form-data request.

    curl example:
      curl -X POST http://localhost:8000/generate-paths/ \
        -F "files=@intro.txt" \
        -F "files=@advanced.txt" \
        -F "urls=https://github.com/user/repo" \
        -F "urls=https://some-blog.com/article"
    """
    tasks = []

    if files:
        for file in files:
            async def _file(f=file):
                content = await f.read()
                text = content.decode("utf-8")
                name = f.filename.replace(".txt", "")
                return await process_source(name, text)
            tasks.append(_file())

    if urls:
        for url in urls:
            async def _url(u=url):
                name, text = await fetch_url_text(u.strip())
                return await process_source(name, text)
            tasks.append(_url())

    if not tasks:
        raise HTTPException(status_code=400, detail="Provide at least one file or URL.")

    results = await asyncio.gather(*tasks)
    files_data = {name: data for name, data in results}
    return await build_response(files_data)
