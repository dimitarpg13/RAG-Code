from sqladmin import ModelView
from app.indexing.models import IndexedRepo  # your SQLAlchemy model

class IndexedRepoAdmin(ModelView, model=IndexedRepo):
    # columns to show in list/detail forms
    # TODO: Add the columns you want to be able to visualize in the Admin UI.
    column_list = [
        IndexedRepo.id,
    ]
    # optional niceties
    column_searchable_list = [IndexedRepo.github_url]
    column_sortable_list = [IndexedRepo.indexed_at]
