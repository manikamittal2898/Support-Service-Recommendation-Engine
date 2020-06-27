from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flaskblog.modals import User

class RegistrationForm(FlaskForm):
    firstname = StringField('First Name',
                           validators=[DataRequired(), Length(min=2, max=20)])
    lastname = StringField('Last Name',
                           validators=[DataRequired(), Length(min=2, max=20)])
    gender = SelectField('Gender',  choices=[('M','Male'),('F','Female')]
                           )                       
    age = IntegerField('Age',
                           validators=[DataRequired()])
    mbc = IntegerField('Purchase Limit',
                           validators=[DataRequired()])
                       
    purpose = SelectField('Purpose',  choices=[('Personal Use','Personal Use'),('Academic purpose','Academic purpose'),('Big Organization','Big Organization'),('Small Organization','Small Organization'), ('Gaming Purpose','Gaming Purpose')])
    
    email = StringField('Email Id',
                        validators=[DataRequired(), Email()])

    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')