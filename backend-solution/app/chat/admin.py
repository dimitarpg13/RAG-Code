from sqladmin import ModelView
from app.chat.models import User, Message

class UserAdmin(ModelView, model=User):
    column_list = [
        User.id,
        User.username,
        User.messages,
    ]

class MessageAdmin(ModelView, model=Message):
    column_list = [
        Message.id,
        Message.user_id,
        Message.message,
        Message.type,
        Message.timestamp,
        Message.user,
    ]