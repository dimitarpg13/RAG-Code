from sqladmin import ModelView
from app.chat.models import User, Message

class UserAdmin(ModelView, model=User):
    # TODO:  Add the columns you want to be able to visualize in the UI.
    column_list = []

class MessageAdmin(ModelView, model=Message):
    # TODO:  Add the columns you want to be able to visualize in the UI.
    column_list = []