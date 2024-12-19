import os
import cv2
import numpy as np
import base64
from pathlib import Path
import shutil
from utils.pose_visualizer import PoseVisualizer

class ImageService:
    def __init__(self, app_config, model, pose):
        self.config = app_config
        self.model = model
        self.pose = pose
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.config.DATABASE_FOLDER, exist_ok=True)
    
    def allowed_file(self, filename):
        """检查文件类型是否允许"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.config.ALLOWED_EXTENSIONS
    
    def analyze_body_image(self, image_file):
        """分析身体图像"""
        try:
            # 读取图片
            image_stream = image_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': '无法读取图片'}, 400
            
            # 使用YOLO进行姿态检测
            results = self.model(image)
            
            if not results or len(results) == 0:
                return {'error': '模型处理失败'}, 400
            
            # 获取人体检测结果
            detections = results[0].boxes.data.cpu()
            
            if len(detections) == 0:
                return {'error': '未检测到人体，请上传包含完整人体的正面照片'}, 400
                
            # 使用最大的检测框
            detection = max(detections, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            
            # 计算身体比例和BMI
            bmi_result = self._calculate_bmi(detection)
            
            # 绘制检测结果
            result_image = PoseVisualizer.draw_detection(image, results, detection)
            
            # 准备响应
            response = self._prepare_response(result_image, bmi_result)
            return response
            
        except Exception as e:
            print(f"Error in analyze_body_image: {str(e)}")
            return {'error': f'图片处理出错: {str(e)}'}, 500
    
    def _calculate_bmi(self, detection):
        """计算BMI和相关信息"""
        width = float(detection[2] - detection[0])
        height = float(detection[3] - detection[1])
        ratio = width / height
        
        def sigmoid(x, L=6, k=20):
            return L / (1 + np.exp(-k * (x - 0.35)))
        
        base_bmi = 18 + sigmoid(ratio)
        bmi = base_bmi + np.random.normal(0, 0.3)
        
        # 处理超出范围的值
        if bmi < 18.0:
            bmi = 18.0 + np.random.uniform(0, 0.2)
        elif bmi > 24.0:
            bmi = 24.0 + np.random.uniform(0, 0.2)
        
        # 确定类别
        if bmi < 18.5:
            category = "竹节虫"
        elif bmi < 22:
            category = "正常娃"
        else:
            category = "肉感烧娃"
            
        recommendation = {
            "竹节虫": "您瘦得像根针，下雨都淋不着您！",
            "正常娃": "恭喜！您的体型很标准，我们都应该像您学习！",
            "肉感烧娃": "不要盯着我看了，网站都要被您烧坏了！"
        }[category]
        
        return {
            'bmi': f"{bmi:.1f}",
            'category': category,
            'recommendation': recommendation
        }
    
    def _prepare_response(self, result_image, bmi_result):
        """准备响应数据"""
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
        
        return {
            'bmi': bmi_result['bmi'],
            'category': bmi_result['category'],
            'recommendation': bmi_result['recommendation'],
            'annotated_image': result_image_b64
        } 