"""
FastAPI Todo Management System with Concurrent API Calls.

Demonstrates async operations, caching, and data processing.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Concurrent Todo API Gateway",
    description="A FastAPI application that demonstrates concurrent API calls to JSONPlaceholder",  # noqa: E501
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache
CACHE_TTL = 300  # 5 minutes

# Base URL for JSONPlaceholder API
BASE_URL = "https://jsonplaceholder.typicode.com"


# Pydantic models
class Todo(BaseModel):
    """Todo item model."""

    userId: int
    id: int
    title: str
    completed: bool


class TodoCreate(BaseModel):
    """Todo creation model."""

    title: str = Field(..., min_length=1, max_length=200)
    completed: bool = False


class TodoUpdate(BaseModel):
    """Todo update model."""

    title: Optional[str] = Field(None, min_length=1, max_length=200)
    completed: Optional[bool] = None


class UserTodoStats(BaseModel):
    """User todo statistics model."""

    user_id: int
    total_todos: int
    completed_todos: int
    pending_todos: int
    completion_percentage: float


class BatchRequest(BaseModel):
    """Batch request model for multiple users."""

    user_ids: List[int] = Field(..., min_items=1, max_items=10)


class CacheInfo(BaseModel):
    """Cache information model."""

    cached_items: int
    cache_hit_rate: float
    last_updated: Optional[datetime]


# Cache utilities
class CacheManager:
    """Cache manager with TTL and statistics."""

    def __init__(self):
        """Initialize cache manager."""
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with TTL check."""
        if key in self.cache:
            item, timestamp = self.cache[key]
            if time.time() - timestamp < CACHE_TTL:
                self.hits += 1
                return item
            del self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any):
        """Set item in cache with timestamp."""
        self.cache[key] = (value, time.time())

    def clear(self):
        """Clear cache and reset statistics."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "cached_items": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total,
        }


cache_manager = CacheManager()


# HTTP client
async def get_http_client():
    """Get HTTP client with timeout configuration."""
    return httpx.AsyncClient(timeout=30.0)


# Utility functions
async def fetch_todos_for_user(client: httpx.AsyncClient, user_id: int) -> List[Dict]:
    """Fetch todos for a specific user with error handling."""
    try:
        cache_key = f"user_todos_{user_id}"
        cached_data = cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        response = await client.get(f"{BASE_URL}/todos", params={"userId": user_id})
        response.raise_for_status()
        todos = response.json()

        cache_manager.set(cache_key, todos)
        return todos
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.error("Error fetching todos for user %d: %s", user_id, exc)
        return []


async def fetch_all_todos(client: httpx.AsyncClient) -> List[Dict]:
    """Fetch all todos with caching."""
    cache_key = "all_todos"
    cached_data = cache_manager.get(cache_key)
    if cached_data:
        return cached_data

    try:
        response = await client.get(f"{BASE_URL}/todos")
        response.raise_for_status()
        todos = response.json()

        cache_manager.set(cache_key, todos)
        return todos
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.error("Error fetching all todos: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch todos") from exc


async def concurrent_user_fetch(
    client: httpx.AsyncClient, user_ids: List[int]
) -> Dict[int, List[Dict]]:
    """Fetch todos for multiple users concurrently."""
    tasks = [fetch_todos_for_user(client, user_id) for user_id in user_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    user_todos = {}
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Error for user %d: %s", user_ids[i], result)
            user_todos[user_ids[i]] = []
        else:
            user_todos[user_ids[i]] = result

    return user_todos


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Concurrent Todo API Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/todos/1", timeout=5.0)
            external_api_status = (
                "healthy" if response.status_code == 200 else "unhealthy"
            )
    except Exception:  # pylint: disable=broad-exception-caught
        external_api_status = "unhealthy"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "external_api": external_api_status,
        "cache_stats": cache_manager.get_stats(),
    }


@app.get("/todos", response_model=List[Todo])
async def get_all_todos(
    limit: Optional[int] = Query(
        None, ge=1, le=200, description="Limit number of results"
    ),
    completed: Optional[bool] = Query(None, description="Filter by completion status"),
):
    """Get all todos with optional filtering."""
    async with await get_http_client() as client:
        todos = await fetch_all_todos(client)

        # Apply filters
        if completed is not None:
            todos = [todo for todo in todos if todo["completed"] == completed]

        if limit:
            todos = todos[:limit]

        return todos


@app.get("/todos/user/{user_id}", response_model=List[Todo])
async def get_user_todos(user_id: int):
    """Get todos for a specific user."""
    if user_id < 1 or user_id > 10:
        raise HTTPException(status_code=400, detail="User ID must be between 1 and 10")

    async with await get_http_client() as client:
        todos = await fetch_todos_for_user(client, user_id)
        if not todos:
            raise HTTPException(status_code=404, detail="No todos found for this user")
        return todos


@app.post("/todos/users/batch", response_model=Dict[int, List[Todo]])
async def get_batch_user_todos(batch_request: BatchRequest):
    """Get todos for multiple users concurrently."""
    start_time = time.time()

    # Validate user IDs
    invalid_users = [uid for uid in batch_request.user_ids if uid < 1 or uid > 10]
    if invalid_users:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user IDs: {invalid_users}. Must be between 1 and 10",  # noqa: E501
        )

    async with await get_http_client() as client:
        user_todos = await concurrent_user_fetch(client, batch_request.user_ids)

    execution_time = time.time() - start_time
    logger.info("Batch fetch completed in %.2f seconds", execution_time)

    return user_todos


@app.get("/todos/stats", response_model=List[UserTodoStats])
async def get_todo_statistics():
    """Get comprehensive statistics for all users."""
    async with await get_http_client() as client:
        # Fetch all users concurrently
        user_ids = list(range(1, 11))
        user_todos = await concurrent_user_fetch(client, user_ids)

        stats = []
        for user_id, todos in user_todos.items():
            if todos:
                total = len(todos)
                completed = sum(1 for todo in todos if todo["completed"])
                pending = total - completed
                completion_percentage = (completed / total * 100) if total > 0 else 0

                stats.append(
                    UserTodoStats(
                        user_id=user_id,
                        total_todos=total,
                        completed_todos=completed,
                        pending_todos=pending,
                        completion_percentage=round(completion_percentage, 2),
                    )
                )

        return sorted(stats, key=lambda x: x.completion_percentage, reverse=True)


@app.get("/todos/search")
async def search_todos(
    q: str = Query(..., min_length=2, description="Search term"),
    user_id: Optional[int] = Query(None, ge=1, le=10, description="Filter by user ID"),
):
    """Search todos by title."""
    async with await get_http_client() as client:
        if user_id:
            todos = await fetch_todos_for_user(client, user_id)
        else:
            todos = await fetch_all_todos(client)

        # Search in titles
        search_results = [todo for todo in todos if q.lower() in todo["title"].lower()]

        return {
            "query": q,
            "user_id": user_id,
            "results": search_results,
            "total_found": len(search_results),
        }


@app.get("/todos/analytics")
async def get_analytics():
    """Get comprehensive analytics across all users."""
    start_time = time.time()

    async with await get_http_client() as client:
        user_ids = list(range(1, 11))
        user_todos = await concurrent_user_fetch(client, user_ids)

        # Aggregate analytics
        total_todos = 0
        total_completed = 0
        user_stats = []
        word_frequency = defaultdict(int)

        for user_id, todos in user_todos.items():
            if todos:
                completed_count = sum(1 for todo in todos if todo["completed"])
                total_todos += len(todos)
                total_completed += completed_count

                user_stats.append(
                    {
                        "user_id": user_id,
                        "todo_count": len(todos),
                        "completion_rate": (
                            completed_count / len(todos) if todos else 0
                        ),
                    }
                )

                # Word frequency analysis
                for todo in todos:
                    words = todo["title"].lower().split()
                    for word in words:
                        if len(word) > 3:  # Only count significant words
                            word_frequency[word] += 1

        # Top words
        top_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        execution_time = time.time() - start_time

        return {
            "execution_time_seconds": round(execution_time, 2),
            "total_todos": total_todos,
            "total_completed": total_completed,
            "overall_completion_rate": (
                round(total_completed / total_todos * 100, 2) if total_todos > 0 else 0
            ),
            "user_statistics": sorted(
                user_stats, key=lambda x: x["completion_rate"], reverse=True
            ),
            "most_common_words": [
                {"word": word, "frequency": freq} for word, freq in top_words
            ],
            "cache_performance": cache_manager.get_stats(),
        }


@app.get("/cache/info", response_model=Dict[str, Any])
async def get_cache_info():
    """Get cache information and statistics."""
    return cache_manager.get_stats()


@app.delete("/cache/clear")
async def clear_cache():
    """Clear the application cache."""
    cache_manager.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/todos/performance-test")
async def performance_test(concurrent_users: int = Query(5, ge=1, le=10)):
    """Test concurrent performance with multiple user requests."""
    start_time = time.time()

    async with await get_http_client() as client:
        # Create multiple concurrent requests
        tasks = []
        for _ in range(concurrent_users):
            for user_id in range(1, 11):
                tasks.append(fetch_todos_for_user(client, user_id))

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successes = sum(1 for result in results if not isinstance(result, Exception))
        failures = len(results) - successes

        execution_time = time.time() - start_time

        return {
            "concurrent_users_simulated": concurrent_users,
            "total_requests": len(tasks),
            "successful_requests": successes,
            "failed_requests": failures,
            "execution_time_seconds": round(execution_time, 2),
            "requests_per_second": round(len(tasks) / execution_time, 2),
            "cache_stats": cache_manager.get_stats(),
        }


# Background task example
@app.post("/todos/background-analysis")
async def trigger_background_analysis(background_tasks: BackgroundTasks):
    """Trigger a background analysis task."""
    task_id = str(uuid.uuid4())

    async def analyze_todos():
        """Background task to analyze all todos."""
        logger.info("Starting background analysis %s", task_id)

        async with await get_http_client() as client:
            todos = await fetch_all_todos(client)

            # Simulate heavy processing
            await asyncio.sleep(2)

            analysis = {
                "task_id": task_id,
                "total_todos": len(todos),
                "completed": sum(1 for todo in todos if todo["completed"]),
                "analysis_completed_at": datetime.now().isoformat(),
            }

            # In a real app, you'd store this in a database
            cache_manager.set(f"analysis_{task_id}", analysis)
            logger.info("Background analysis %s completed", task_id)

    background_tasks.add_task(analyze_todos)

    return {
        "message": "Background analysis started",
        "task_id": task_id,
        "check_result_url": f"/todos/analysis/{task_id}",
    }


@app.get("/todos/analysis/{task_id}")
async def get_analysis_result(task_id: str):
    """Get the result of a background analysis."""
    result = cache_manager.get(f"analysis_{task_id}")
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
