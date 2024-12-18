# Kiger 图片相似度检测 Web 应用 v0.2

这是一个为 **Kiger** 开发的小程序，目前实现了推荐 Kiger 店铺和 BMI 检测功能。未来计划集成更多功能以提升用户体验和系统能力。

## 功能

### 当前版本 (v0.2)

- **推荐 Kiger 店铺**：用户可以上传图片，系统将分析图片并推荐数据库中相似的 Kiger 店铺图片。
- **BMI 检测**：
  - **自动检测**：用户上传全身照，系统自动分析并计算 BMI 指数
  - **手动输入**：用户手动输入身高体重数据，系统计算 BMI 指数

### 未来功能

- **订单查询**：用户可以查询已有订单的状态和详情。
- **新订单填写**：用户可以在线填写并提交新订单。
- **AI 辅助 Kiger 头壳的参考图生成**：通过输入文字或上传图片，AI 将生成 Kiger 头壳的参考图。
- **AI 辅助 3D 成品图或视频参考**：生成 3D 成品图或视频的参考资料，帮助用户更好地了解产品。
- **AI 辅助 Kiger 照片抠图和打光换场景**：自动抠图并调整照片的光照和背景场景。
- **Kiger 实时变声器**：提供实时变声功能，用户可以在交流中改变声音效果。
- **Kiger 角色的 AI 翻唱和变声**：为 Kiger 角色提供 AI 翻唱和变声功能，丰富角色表现力。
- **BMI 推算**：根据用户输入的数据，计算并推算 BMI 指数。
- **体型检测和相应角色推荐**：分析用户体型并推荐最适合的 Kiger 角色。

## 项目结构
```
    ├── app.py
    ├── image_processor.py
    ├── requirements.txt
    ├── README.md
    ├── static
    │   ├── uploads # 存储上传的图片
    │   ├── database
    │   └── sounds
    │       └── hover.mp3 # 按钮悬停音效
    ├── templates
    │   ├── index.html
    │   ├── admin.html
    │   ├── result.html
    │   ├── bmi.html
    │   ├── login.html
    │   └── similarity.html
    ├── images.db
    └── yolov8n-pose.pt # YOLO姿态检测模型文件
```