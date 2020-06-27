from flaskblog import app, db
from flaskblog.modals import User, Product, Final

user_obj=User.query.filter_by(email=id).all()
id=user_obj.id
age=user_obj.age
gender=user_obj.gender
mbc=user_obj.mbc
purpose=user_obj.purpose

