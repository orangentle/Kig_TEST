# 鼠鼠工坊微信小程序

<div align="center">
  <img src="icon.png" alt="鼠鼠工坊Logo" width="120"/>
  <h3>专注于Kigurumi头壳制作的工作室小程序</h3>
</div>

## 项目简介

鼠鼠工坊微信小程序是为专注于Kigurumi头壳制作的工作室开发的订单管理与进度追踪平台。该小程序为客户提供了便捷的订单查询和制作进度跟踪功能，让客户可以随时了解自己定制头壳的制作状态。

PS：云端的envId随便写的测试用环境。

## 功能特点

- **订单查询**：通过淘宝订单号快速查询头壳制作进度
- **进度追踪**：直观的进度条展示当前制作阶段
- **个人中心**：登录后查看个人所有订单列表
- **微信登录**：支持一键微信授权登录
- **云开发功能**：使用微信云开发实现数据管理和订单处理

## 制作流程

头壳制作流程分为以下几个阶段：
1. 订单确认
2. 设计图确认
3. 模型制作
4. 打印
5. 打磨上色
6. 组装
7. 质检
8. 发货

## 技术栈

- 微信小程序原生开发
- TypeScript
- TDesign 微信小程序组件库
- 微信云开发

## 项目结构

```
miniprogram/
  ├── app.json          # 全局配置
  ├── app.ts            # 应用入口
  ├── app.wxss          # 全局样式
  ├── assets/           # 静态资源
  │   ├── icons/        # 图标资源（包含导航栏图标）
  │   └── images/       # 图片资源
  ├── pages/            # 页面文件
  │   ├── index/        # 首页（订单查询）
  │   ├── logs/         # 日志页面
  │   ├── order-detail/ # 订单详情页
  │   ├── profile/      # 个人中心
  │   └── webview/      # 网页视图（用于打开外部链接）
  ├── miniprogram_npm/  # 小程序依赖包
  └── utils/            # 工具函数

cloudfunctions/
  ├── adminAuth/        # 管理员身份验证
  ├── createOrder/      # 创建订单
  ├── getOpenId/        # 获取用户OpenID
  ├── getOrders/        # 获取订单列表
  ├── initOrderData/    # 初始化订单数据
  ├── login/            # 用户登录
  └── updateAvatar/     # 更新用户头像
```

## 原型设计

```
prototype/
  ├── admin.html            # 管理员后台原型
  ├── home.html             # 首页原型
  ├── index.html            # 原型入口
  ├── order-detail.html     # 订单详情原型
  ├── profile.html          # 已登录个人中心原型
  └── profile-unlogin.html  # 未登录个人中心原型
```

## 安装与运行

1. 克隆仓库
```bash
git clone https://github.com/your-username/ratstudio-miniprogram.git
```

2. 使用微信开发者工具打开项目

3. 在微信开发者工具中导入项目，选择项目根目录

4. 编译运行

## 项目截图

<div align="center">
  <h3>用户端界面</h3>
  <img src="用户端设计稿.png" alt="鼠鼠工坊用户端界面" width="100%"/>
  
  <h3>管理员后台</h3>
  <img src="管理员设计稿.png" alt="鼠鼠工坊管理员后台" width="100%"/>
</div>

## 设计原型

项目包含完整的设计原型，位于 `prototype/` 目录下，可通过浏览器直接打开 `prototype/index.html` 查看。

## 开发进度

- [x] 项目初始化
- [x] 首页开发
- [x] 订单详情页开发
- [x] 个人中心页面开发
- [x] 原型展示页面
- [x] WebView页面开发（用于打开淘宝客服）
- [x] 云函数开发
- [x] 订单表单设计
- [ ] 后端接口对接
- [ ] 上线发布

## 版权信息

© 2025 鼠鼠工坊 - 保留所有权利 