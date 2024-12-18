from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from functools import wraps
import os
from werkzeug.utils import secure_filename
from image_processor import ImageMatcher
from pathlib import Path
import shutil
import numpy as np
import cv2
from ultralytics import YOLO
import base64
import mediapipe as mp

# 初始化 YOLO 和 MediaPipe
model = YOLO('yolov8n-pose.pt')  # 使用YOLO的姿态检测模型
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

# 定义骨架连接和颜色
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                   (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = {
    'box': (0, 255, 0),      # 亮绿色边框
    'joints': (255, 50, 50),  # 亮红色关节点
    'bones': (50, 50, 255),   # 亮蓝色骨架
    'text': (255, 255, 255)   # 白色文字
}

THICKNESS = {
    'box': 6,        # 边框粗细
    'bones': 4,      # 骨架线条粗细
    'joints': 8,     # 关节点大小
    'joints_outline': 10  # 关节点外圈大小
}

def draw_detection(image, results, detection):
    """绘制检测结果，使用更粗的线条和更大的点"""
    h, w = image.shape[:2]
    # 创建稍微暗一点的背景以突出显示
    result_image = cv2.addWeighted(image.copy(), 0.9, np.zeros_like(image), 0.1, 0)
    
    # 绘制检测框
    x1, y1, x2, y2 = map(int, detection[:4])
    cv2.rectangle(result_image, 
                 (x1, y1), (x2, y2), 
                 COLORS['box'], 
                 THICKNESS['box'])
    
    # 获取姿态关键点
    keypoints = results[0].keypoints.data[0].cpu().numpy()
    
    # 绘制骨架
    for p1_idx, p2_idx in POSE_CONNECTIONS:
        if keypoints[p1_idx][2] > 0.5 and keypoints[p2_idx][2] > 0.5:
            p1 = tuple(map(int, keypoints[p1_idx][:2]))
            p2 = tuple(map(int, keypoints[p2_idx][:2]))
            # 先画粗白色线条作为外边框
            cv2.line(result_image, p1, p2, (255, 255, 255), 
                    THICKNESS['bones'] + 2)
            # 再画主色线条
            cv2.line(result_image, p1, p2, COLORS['bones'], 
                    THICKNESS['bones'])
    
    # 绘制关节点
    for kp in keypoints:
        if kp[2] > 0.5:  # 只绘制置信度高的关键点
            x, y = map(int, kp[:2])
            # 绘制白色外圈
            cv2.circle(result_image, (x, y), 
                      THICKNESS['joints_outline'], (255, 255, 255), -1)
            # 绘制内部实心圆
            cv2.circle(result_image, (x, y), 
                      THICKNESS['joints'], COLORS['joints'], -1)
    
    # 添加检测框标签
    label = f"Person {detection[4]:.2f}"
    font_scale = 1.2
    font_thickness = 3
    # 添加文字阴影效果
    cv2.putText(result_image, label, 
                (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), font_thickness+1, cv2.LINE_AA)
    cv2.putText(result_image, label, 
                (x1, y1-12), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, COLORS['text'], font_thickness, cv2.LINE_AA)
    
    return result_image

app = Flask(__name__)
app.config['ADMIN_USERNAME'] = 'admin'
app.config['ADMIN_PASSWORD'] = '970602'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE_FOLDER'] = 'static/database'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'your_secret_key_here'

# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if (request.form['username'] == app.config['ADMIN_USERNAME'] and 
            request.form['password'] == app.config['ADMIN_PASSWORD']):
            session['logged_in'] = True
            return redirect(url_for('admin'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/admin')
@login_required
def admin():
    db_images = image_matcher.get_all_images()
    return render_template('admin.html', images=db_images)

# 确保必要的文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 初始化图像匹配器
image_matcher = ImageMatcher()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/similarity')
def similarity_system():
    return render_template('similarity.html')  # 这是原来的index.html内容，改名为similarity.html

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"上传的文件保存在: {filepath}")

        # 获取最相似的图片
        similar_images = image_matcher.find_similar_images(filepath, top_k=5, similarity_threshold=0.5)
        print(f"找到的相似图片: {similar_images}")
        
        # 构建结果数据
        results = {
            'uploaded_image': {
                'path': f'/static/uploads/{filename}',
                'filename': filename
            },
            'similar_images': similar_images
        }

        return render_template('result.html', results=results)

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/add-to-database', methods=['POST'])
def add_to_database():
    if 'files[]' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('admin'))

    files = request.files.getlist('files[]')
    success_count = 0
    error_count = 0

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['DATABASE_FOLDER'], filename)

            try:
                # 保存文件
                file.save(filepath)
                print(f"文件已保存到: {filepath}")
                
                # 添加到数据库
                if image_matcher.add_image_to_db(filepath):
                    print(f"成功添加图片到数据库: {filename}")
                    success_count += 1
                else:
                    print(f"添加图片到数据库失败: {filename}")
                    error_count += 1
            except Exception as e:
                print(f"处理文件时出错 {filename}: {str(e)}")
                error_count += 1

    if success_count > 0:
        flash(f'Successfully added {success_count} images to database', 'success')
    if error_count > 0:
        flash(f'Failed to add {error_count} images', 'error')

    return redirect(url_for('admin'))


@app.route('/delete-from-database/<int:image_id>', methods=['POST'])
def delete_from_database(image_id):
    if image_matcher.delete_image(image_id):
        flash('Image successfully deleted', 'success')
    else:
        flash('Failed to delete image', 'error')
    return redirect(url_for('admin'))

@app.route('/cleanup-uploads', methods=['POST'])
def cleanup_uploads():
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])
        flash('Upload folder cleaned successfully', 'success')
    except Exception as e:
        flash(f'Error cleaning upload folder: {str(e)}', 'error')
    return redirect(url_for('admin'))

@app.route('/bmi')
def bmi_calculator():
    return render_template('bmi.html')

@app.route('/analyze-body-image', methods=['POST'])
def analyze_body_image():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择图片'}), 400

    try:
        # 读取图片
        image_stream = file.read()
        nparr = np.frombuffer(image_stream, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 使用YOLO进行姿态检测
        results = model(image)
        
        # 获取人体检测结果
        detections = results[0].boxes.data
        
        if len(detections) == 0:
            return jsonify({
                'error': '未检测到人体，请上传包含完整人体的正面照片'
            }), 400
            
        # 使用最大的检测框
        detection = max(detections, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        
        # 计算身体比例
        width = detection[2] - detection[0]
        height = detection[3] - detection[1]
        ratio = width / height
        
        # 估算BMI
        if ratio < 0.3:
            base_bmi = 17.5
            category = "竹节虫"
        elif ratio < 0.35:
            base_bmi = 20.5
            category = "正常娃"
        elif ratio < 0.4:
            base_bmi = 25.5
            category = "肉感烧娃"
        else:
            base_bmi = 29.0
            category = "冰箱"
            
        bmi = base_bmi + np.random.uniform(-1.0, 1.0)
            
        recommendation = {
            "竹节虫": "您瘦得像根针，下雨都淋不着您！",
            "正常娃": "恭喜！您的体型很标准，我们都应该像您学习！",
            "肉感烧娃": "不要盯着我看了，网站都要被您烧坏了！",
            "冰箱": "您就往这一站，我都能感觉到寒气！"
        }[category]

        # 绘制检测结果
        result_image = draw_detection(image, results, detection)
        
        # 转换图片为base64
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

        return jsonify({
            'bmi': f"{bmi:.1f}",
            'category': category,
            'recommendation': recommendation,
            'annotated_image': result_image_b64
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'图片处理出错: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)