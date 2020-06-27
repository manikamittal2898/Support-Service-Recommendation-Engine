from flaskblog import db, login_manager
from flask_login import UserMixin
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(50), nullable=False)
    lastname = db.Column(db.String(50), nullable=False)
    purpose = db.Column(db.String(50), nullable=False)
    gender=db.Column(db.String(10), nullable=False)
    age=db.Column(db.Integer, nullable=False)
    mbc=db.Column(db.Integer, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    services = db.relationship('Service', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.firstname}', '{self.lastname}', '{self.purpose}', '{self.gender}','{self.age}', '{self.mbc}', '{self.email}')"
    
class Service(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    service_name=db.Column(db.String(100), nullable=False )
    cost = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Service('{self.service_name}', '{self.cost}')"

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prod_name=db.Column(db.String(100), nullable=False )
    prod_type=db.Column(db.String(100), nullable=False )
    
    def __repr__(self):
        return f"Product('{self.prod_name}', '{self.prod_type}')"   

class Final(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prod_id=db.Column(db.String(100), db.ForeignKey('product.id'), nullable=False ) 
    service=db.Column(db.String(100), nullable=False )
    cost = db.Column(db.Integer, nullable=False)
    expiry = db.Column(db.Integer, nullable=False)
    category = db.Column(db.Integer, nullable=False)
    
    def __repr__(self):
        return f"Final('{self.user_id}', '{self.prod_id}', '{self.service}','{self.cost}','{self.expiry}','{self.category}')"                     