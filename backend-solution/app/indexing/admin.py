from sqladmin import ModelView
from app.indexing.models import IndexedRepo  # your SQLAlchemy model

class IndexedRepoAdmin(ModelView, model=IndexedRepo):
    # columns to show in list/detail forms
    column_list = [
        IndexedRepo.id,
        IndexedRepo.github_url,
        IndexedRepo.namespace,
        IndexedRepo.indexed_at,
        IndexedRepo.updated_at,
    ]
    # optional niceties
    column_searchable_list = [IndexedRepo.github_url]
    column_sortable_list = [IndexedRepo.indexed_at]
