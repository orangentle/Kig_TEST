from flask import Blueprint, render_template, request, jsonify
from rembg import remove, new_session
from PIL import Image
import io
import base64
import os

# 设置环境变量，指定模型路径
os.environ['U2NET_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models'

bg_bp = Blueprint('bg', __name__)

# 获取项目根目录和模型目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 定义模型配置
MODEL_PATHS = {
    'isnet-anime': {
        'name': 'isnet-anime',
        'path': os.path.join(MODELS_DIR, 'isnet-anime.onnx')
    },
    'u2net': {
        'name': 'u2net',
        'path': os.path.join(MODELS_DIR, 'u2net.onnx')
    }
}

def create_session(model_key):
    model_info = MODEL_PATHS[model_key]
    if not os.path.exists(model_info['path']):
        raise FileNotFoundError(f"模型文件不存在: {model_info['path']}")
    
    return new_session(
        model_name=model_info['name'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

# 创建两个会话：动漫风格和通用物体
anime_session = create_session('isnet-anime')
object_session = create_session('u2net')

@bg_bp.route('/remove-bg', methods=['GET', 'POST'])
def remove_bg():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择图片'})
        
        # 获取选择的模型类型
        model_type = request.form.get('model_type', 'anime')  # 默认使用anime模型
        
        try:
            # 读取上传的图片
            input_image = Image.open(file)
            
            # 根据选择使用不同的会话
            if model_type == 'anime':
                output_image = remove(input_image, session=anime_session)
            else:
                output_image = remove(input_image, session=object_session)
            
            # 将结果转换为base64
            buffered = io.BytesIO()
            output_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({'result': img_str})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return render_template('remove_bg.html')