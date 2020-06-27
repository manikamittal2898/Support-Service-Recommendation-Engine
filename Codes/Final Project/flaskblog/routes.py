from flask import escape, request, render_template, url_for, flash, redirect
from flaskblog.modals import User, Product, Final
from flaskblog import app, db, bcrypt
from flaskblog.forms import RegistrationForm, LoginForm
from flask_login import login_user, current_user, logout_user, login_required
import numpy as np
import pandas as pd
import csv
import os



email='yevette.mullis@gmail.com'

user_obj=User.query.filter_by(email=email).all()


id=user_obj[0].id
age1=user_obj[0].age
gender1=user_obj[0].gender
mbc=user_obj[0].mbc
purpose=user_obj[0].purpose

list_id=Final.query.filter_by(user_id=id).all()
list2=[]
for x in list_id:
    list2.append(x.prod_id)

list3=[]
for x in list2:
    pr=Product.query.filter_by(id=x).all()
    list3.append(pr[0].prod_type)  


dataset = pd.read_csv('/home/shivam/Desktop/Final Project/flaskblog/Final_Database.csv')
user_id=id
final_list=[]
list1=[]
for i, j in dataset.iterrows():
	if j['User ID']==user_id:
         
         X=j['Service']
         Y=j['Category']
         for index, row in dataset.iterrows():
	         if row['Category']==Y and X!=row['Service']:
		         list1.append(row['Service'])
	         elif row['Months to expire']<=5 and row['Service']==X:
		         list1.append(row['Service'])


for s in list1:
	if s not in final_list:
		final_list.append(s)

final_set= set(final_list)
final_list= list(final_set)
print(final_list)

@app.route('/')
@app.route('/home')
def home():
    print(User.query.all())
    return render_template('home.html')

@app.route('/acad')
def acad():
    return render_template('acad.html')

@app.route('/gaming')
def gaming():
    return render_template('gaming.html')

@app.route('/personal')
def personal():
    return render_template('pers.html')

@app.route('/small')
def small():
    return render_template('smallo.html')

@app.route('/big')
def big():
    return render_template('bigo.html')

@app.route('/recommendation')
def recommendation():
    print(final_list)
    return render_template('recommendation.html', lista=final_list)    


@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(firstname=form.firstname.data, lastname=form.lastname.data, purpose=form.purpose.data, gender=form.gender.data, age=form.age.data, mbc=form.mbc.data, email=form.email.data, password = form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Your Account has been created!','success')
        return redirect(url_for('login'))
    return render_template('register.html', form = form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        email=form.email.data
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.password==form.password.data:
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('Login.html', form = form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

# @app.route("/account")
# @login_required
# def account():
# return render_template('account.html', title='Account')        