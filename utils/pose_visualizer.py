import cv2
import numpy as np

# 定义骨架连接和颜色
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                   (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

COLORS = {
    'box': (0, 255, 0),      # 保持亮绿色边框
    'joints': (255, 100, 100),  # 调亮红色关节点
    'bones': (100, 100, 255),   # 调亮蓝色骨架
    'text': (255, 255, 255)   # 白色文字
}

THICKNESS = {
    'box': 8,              # 增加边框粗细
    'bones': 12,           # 显著增加骨架线条粗细
    'joints': 15,          # 增大关节点大小
    'joints_outline': 18   # 增大关节点外圈大小
}

class PoseVisualizer:
    @staticmethod
    def draw_detection(image, results, detection):
        """绘制检测结果，使用更粗的线条和更大的点，并且裁剪到检测框区域"""
        h, w = image.shape[:2]
        
        # 获取检测框坐标并确保在图像范围内
        x1, y1, x2, y2 = map(int, detection[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 扩大检测框范围
        padding = 20
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(w, x2 + padding)
        y2_padded = min(h, y2 + padding)
        
        # 裁剪原始图像
        cropped_image = image[y1_padded:y2_padded, x1_padded:x2_padded].copy()
        
        # 在裁剪后的图像上进行暗化处理
        result_image = cv2.addWeighted(cropped_image, 0.6, np.zeros_like(cropped_image), 0.4, 0)
        
        # 调整关键点坐标以适应裁剪后的图像
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        keypoints_adjusted = keypoints.copy()
        keypoints_adjusted[:, 0] -= x1_padded
        keypoints_adjusted[:, 1] -= y1_padded
        
        # 绘制骨架
        result_image = PoseVisualizer._draw_skeleton(result_image, keypoints_adjusted)
        
        # 绘制关节点
        result_image = PoseVisualizer._draw_joints(result_image, keypoints_adjusted)
        
        # 创建最终图像并添加标签
        final_image = PoseVisualizer._create_final_image(
            h, w, result_image, x1, y1, y1_padded, x1_padded, 
            y2_padded, x2_padded, detection
        )
        
        return final_image

    @staticmethod
    def _draw_skeleton(image, keypoints):
        """绘制骨架连接"""
        for p1_idx, p2_idx in POSE_CONNECTIONS:
            if keypoints[p1_idx][2] > 0.5 and keypoints[p2_idx][2] > 0.5:
                p1 = tuple(map(int, keypoints[p1_idx][:2]))
                p2 = tuple(map(int, keypoints[p2_idx][:2]))
                # 先画粗白色线条作为外边框
                cv2.line(image, p1, p2, (255, 255, 255), 
                        THICKNESS['bones'] + 2)
                # 再画主色线条
                cv2.line(image, p1, p2, COLORS['bones'], 
                        THICKNESS['bones'])
        return image

    @staticmethod
    def _draw_joints(image, keypoints):
        """绘制关节点"""
        for kp in keypoints:
            if kp[2] > 0.5:  # 只绘制置信度高的关键点
                x, y = map(int, kp[:2])
                # 绘制白色外圈
                cv2.circle(image, (x, y), 
                          THICKNESS['joints_outline'], (255, 255, 255), -1)
                # 绘制内部实心圆
                cv2.circle(image, (x, y), 
                          THICKNESS['joints'], COLORS['joints'], -1)
        return image

    @staticmethod
    def _create_final_image(h, w, result_image, x1, y1, y1_padded, x1_padded, 
                           y2_padded, x2_padded, detection):
        """创建最终图像并添加标签"""
        final_image = np.zeros((h, w, 3), dtype=np.uint8)
        final_image[y1_padded:y2_padded, x1_padded:x2_padded] = result_image
        
        # 添加检测框标签
        label = f"Person {detection[4]:.2f}"
        font_scale = 2.0
        font_thickness = 5
        
        # 添加文字阴影效果
        cv2.putText(final_image, label, 
                    (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness+1, cv2.LINE_AA)
        cv2.putText(final_image, label, 
                    (x1, y1-12), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, COLORS['text'], font_thickness, cv2.LINE_AA)
        
        return final_image 