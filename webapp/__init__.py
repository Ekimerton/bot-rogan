from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_login import LoginManager
from webapp.config import Config

app = Flask(__name__)
app.config.from_object(Config)

# db = SQLAlchemy(app)
# bcrypt = Bcrypt(app)
# login_manager = LoginManager(app)
# login_manager.login_view = 'users.login'
# login_manager.login_message_category = 'info'

# db.init_app(app)
# bcrypt.init_app(app)
# login_manager.init_app(app)

from webapp.main.routes import main
from webapp.errors.handlers import errors
app.register_blueprint(main)
app.register_blueprint(errors)
