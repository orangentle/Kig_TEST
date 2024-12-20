from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import mediapipe as mp
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from config import Config
from services import AuthService, ImageService
from image_processor import ImageMatcher
from routes.remove_bg import bg_bp
import onnxruntime as ort

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 初始化 Flask 应用
app = Flask(__name__)
app.config.from_object(Config)

# 初始化 YOLO 和 MediaPipe
model = YOLO(os.path.join(MODELS_DIR, 'yolov8n-pose.pt'))
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

# 初始化服务
image_service = ImageService(Config, model, pose)
auth_service = AuthService(Config)

# 初始化 ImageMatcher
image_matcher = ImageMatcher()

# 注册蓝图
app.register_blueprint(bg_bp)

print("可用的执行提供程序:", ort.get_available_providers())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bmi')
def bmi_calculator():
    return render_template('bmi.html')

@app.route('/similarity-system')
def similarity_system():
    return render_template('similarity.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if auth_service.verify_credentials(
            request.form['username'], 
            request.form['password']
        ):
            session['logged_in'] = True
            return redirect(url_for('admin'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/admin')
@auth_service.login_required
def admin():
    return render_template('admin.html')

@app.route('/analyze-body-image', methods=['POST'])
def analyze_body_image():
    """处理 BMI 检测的图片上传"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择图片'}), 400
    
    try:
        # 使用 image_service 处理图片
        result = image_service.analyze_body_image(file)
        
        # 检查是否有错误
        if isinstance(result, tuple) and len(result) == 2 and 'error' in result[0]:
            return jsonify(result[0]), result[1]
            
        return jsonify(result)
        
    except Exception as e:
        print(f"分析图片时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """处理相似度检测的图片上传"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件被上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('static', 'uploads', filename)
        file.save(filepath)
        
        try:
            # 使用 ImageMatcher 查找相似图片
            similar_images = image_matcher.find_similar_images(
                filepath,
                top_k=5,
                similarity_threshold=0.5
            )
            
            # 简化存储在 session 中的数据
            simplified_results = {
                'uploaded_image': {
                    'filename': filename,
                    'path': os.path.join('uploads', filename)
                },
                'similar_images': [
                    {
                        'filename': img['filename'],
                        'similarity': round(img['similarity'], 3)
                    }
                    for img in similar_images
                ]
            }
            
            # 将简化后的结果存储在 session 中
            session['results'] = simplified_results
            
            return jsonify({
                'success': True,
                'message': '文件上传成功'
            })
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': '不允许的文件类型'}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/result')
def result():
    results = session.get('results')
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)