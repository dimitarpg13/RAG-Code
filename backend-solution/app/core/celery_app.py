from celery import Celery

celery_app = Celery(
    "rag_indexer",
    # The message broker URL. Tasks are queued in Redis DB 0 on localhost:6379.
    broker="redis://localhost:6379/0",
    # Tells Celery to import this module on startup so @celery_app.task functions are discovered/registered.
    include=["app.indexing.tasks"]
)

# Update runtime configuration on the app
celery_app.conf.update(
    # Task arguments are serialized as JSON when sent to the broker (safe and language-agnostic).
    task_serializer="json",
    # Workers will only accept JSON messages (blocks unsafe formats like pickle).
    accept_content=["json"],
    # Don’t store task return values/status in a result backend.
    task_ignore_result=True,
    # If the broker isn’t up yet, Celery keeps retrying on startup instead of failing immediately—handy in dev/docker.
    broker_connection_retry_on_startup=True,
)