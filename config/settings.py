import os

# 基础配置
class Config:
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = '970602'
    SECRET_KEY = 'your_secret_key_here'
    
    # 文件路径配置
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
    DATABASE_FOLDER = os.path.join(BASE_DIR, 'static/database')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} 