from functools import wraps
from flask import session, redirect, url_for

class AuthService:
    def __init__(self, app_config):
        self.config = app_config
    
    def login_required(self, f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('logged_in'):
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    
    def verify_credentials(self, username, password):
        """验证用户凭据"""
        return (username == self.config.ADMIN_USERNAME and 
                password == self.config.ADMIN_PASSWORD) 