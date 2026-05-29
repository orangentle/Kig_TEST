# 鼠鼠工坊微信小程序

<div align="center">
  <img src="icon.png" alt="鼠鼠工坊Logo" width="120"/>
  <h3>专注于 Kigurumi 头壳制作的工作室小程序</h3>
</div>

## 项目简介

鼠鼠工坊微信小程序是为专注于 Kigurumi 头壳制作的工作室开发的订单管理与进度追踪平台。前端覆盖客户下单、订单查询、个人中心、社区活动等场景；后端通过微信云开发提供完整的订单生命周期管理（提交 → 审核 → 排单 → 制作 → 发货），并配备千级订单的管理后台（分页 / 批量 / 筛选 / 审核）。

PS：云端的 envId 为测试环境，字段、订单内容仅供参考。

## 功能特点

### 客户端
- **首页订单查询**：通过淘宝订单号实时查询云端进度（已对接 `getOrders` 云函数）
- **在线下单**：填写身材数据 / 角色信息 / 参考图 / 替换脸 / 加急等，自动锁定待审核
- **个人中心**：登录后查看个人所有订单及制作进度
- **工期计算器**：在"我的"页面入口,按"开始日期 + 工作日数"估算预计完成日(自动跳过周末与法定节假日,识别调休补班),结果仅供参考
- **作品展示 / 偶壳娃聚**：作品案例与社区活动入口
- **微信登录**：一键授权登录

### 管理后台
- **概览统计**：总数 / 待审核 / 制作中 / 加急 / 逾期 / 已完成 / 已归档 多维概览卡，可点击穿透筛选
- **筛选与搜索**：阶段 Tab、关键词模糊（订单号/客户/角色/旺旺/昵称）、日期区间、是否含归档
- **订单审核**：独立审核页（待审核 / 已通过 / 已驳回 / 全部 Tab + 计数），支持单条与批量通过/驳回，驳回带备注
- **批量操作**：选择模式 + 推进阶段 / 设为阶段 / 设加急 / 取消加急 / 分配排单号 / 解锁 / 归档 / 删除
- **新增订单**：表单与客户端字段对齐，分 6 个区块（基本 / 联系 / 身材 / 定制选项 / 排期 / 图片 / 备注），支持多图上传
- **订单详情(管理员视角)**：附加"客户信息"卡(`?admin=true` 进入时显示),展示称呼/微信昵称/旺旺/QQ/手机号,支持复制/拨打
- **危险操作**：清理测试数据（按 `queueNumber` 前缀）

## 订单生命周期

```
[客户提交]            [管理员审核]            [开始制作]                          [完成]
status=pending  →  ─┬─ approve → status=normal  →  推进阶段 → status=completed
stage=pending       │                              stage=queued/modeling/...    stage=shipped
                    └─ reject  → status=rejected
                                 stage=pending（可重新沟通）
```

## 制作流程

对外展示 **5 个阶段**（`OrderStage`）：

| 阶段 | 标签 | 进度 |
|---|---|---|
| `queued` | 已排单 | 10% |
| `modeling` | 建模 | 30% |
| `painting` | 上妆 | 55% |
| `hair` | 假毛 | 80% |
| `shipped` | 已发货 | 100% |

> **后续可选增强：**
> - 内部子状态 `subStage`（如：建模 → 3D 建模 / 打印；上妆 → 打磨 / 喷漆），对外仍展示 5 阶段
> - 客户留言 / 订单评论时间线
> - 物流单号录入与订阅消息自动推送

## 下单字段（客户端 / 后台共用）

提交后自动锁定，需联系客服解锁修改。

- **基本**：淘宝订单号、角色名、角色所属 IP
- **身材**：身高 / 体重 / 头围 / 肩宽
- **参考图**：多视角参考图（最多 3 张）
- **配饰**：是否需要配饰
- **替换脸**：是否需要 + 数量（1-3 张），每张对应一张表情参考图
- **加急**：+1200 元
- **备注**：客户特殊需求

## 数据结构

订单核心类型集中在 `miniprogram/types/order.ts`，所有页面统一引用：

- `OrderStatus`: `pending | normal | urgent | rejected | completed`
- `OrderStage`: `pending | queued | modeling | painting | hair | shipped`
- `Order` 字段：基本信息 / 时间 / 进度 / 状态 / `userInfo` / `bodyMeasurements` / `options` / 图片 / `reviewInfo`

## 技术栈

- 微信小程序原生开发（glass-easel Component）
- **TypeScript**（启用 `useCompilerPlugins: ["typescript"]`，页面层全量 .ts）
- TDesign 微信小程序组件库
- 微信云开发（云函数 + 云数据库 + 云存储）

## 项目结构

```
miniprogram/
  ├── app.json / app.ts / app.wxss
  ├── types/                     # 共享 TS 类型
  │   ├── order.ts               # Order / OrderStatus / OrderStage / STAGE_FLOW
  │   └── user.ts                # UserInfo / UserProfile
  ├── assets/                    # 静态资源（图标、图片）
  ├── pages/
  │   ├── index/                 # 首页（订单查询入口）
  │   ├── order/                 # 客户下单
  │   ├── order-detail/          # 订单详情（含进度时间轴）
  │   ├── profile/               # 个人中心
  │   ├── work-day-calc/         # 工期计算器(对接 timor.tech 节假日 API)
  │   ├── works/                 # 作品展示
  │   ├── gathering/             # 偶壳娃聚
  │   ├── admin/                 # 管理后台
  │   │   ├── admin.ts/.wxml     # 主页面（列表 + 概览 + 批量）
  │   │   ├── order-review/      # 订单审核子页
  │   │   └── works-manage/      # 作品管理子页
  │   ├── webview/               # WebView（外部链接）
  │   ├── logs/                  # 日志页
  │   └── common/                # 通用样式
  └── utils/

cloudfunctions/
  ├── adminAuth/                 # 管理员鉴权
  ├── login/                     # 登录获取 openid
  ├── getOpenId/                 # 获取 openid
  ├── updateAvatar/              # 更新头像
  ├── submitOrder/               # 客户端提交订单（含图片上传后续处理）
  ├── createOrder/               # 后台新建订单（与 Order 结构对齐）
  ├── getOrders/                 # 订单列表（分页/筛选/搜索/计数）
  ├── getOrderDetail/            # 订单详情
  ├── batchUpdateOrders/         # 批量操作（推进 / 审核 / 加急 / 归档 / 解锁 / 删除）
  ├── clearOrders/               # 清理测试数据
  └── initOrderData/             # 初始化示例数据

prototype/                       # 设计原型
```

## 原型设计

```
prototype/
  ├── admin.html / index.html / home.html
  ├── order-detail.html
  ├── profile.html / profile-unlogin.html
  └── ...
```

## 安装与运行

1. 克隆仓库
   ```bash
   git clone https://github.com/your-username/ratstudio-miniprogram.git
   ```
2. 使用微信开发者工具导入项目根目录
3. 在 `project.config.json` 中确认开启 TS 编译插件（`useCompilerPlugins: ["typescript"]`）
4. 配置云开发环境 ID（`miniprogram/app.ts` 与 `cloudfunctions/cloud-env.json`）
5. 上传部署所有云函数（首次或修改后）
6. **配置 request 合法域名**(用到了外部节假日 API):
   - 微信公众平台 → 开发管理 → 开发设置 → 服务器域名 → request 合法域名 添加 `https://timor.tech`
   - 本地调试可在开发者工具勾选"详情 → 本地设置 → 不校验合法域名"临时跳过
7. 编译运行

## 外部依赖

| 用途 | 服务 | 说明 |
|---|---|---|
| 法定节假日数据 | `https://timor.tech/api/holiday/year/{YYYY}` | 公共免费 API,按年拉取,本地 `wx.storage` 缓存 30 天;失败时降级为"仅按周末计算" |

## 项目截图

<div align="center">
  <h3>用户端界面</h3>
  <img src="用户端设计稿.png" alt="鼠鼠工坊用户端界面" width="100%"/>

  <h3>管理员后台</h3>
  <img src="管理员设计稿.png" alt="鼠鼠工坊管理员后台" width="100%"/>
</div>

## 开发进度

- [x] 项目初始化
- [x] 首页 / 订单详情 / 个人中心 / 作品 / 偶壳娃聚
- [x] 客户端下单流程（含锁定 + 订阅消息）
- [x] 云函数：登录 / 订单 CRUD / 批量操作 / 审核
- [x] 管理后台（列表 / 概览 / 筛选 / 批量 / 新增）
- [x] 订单审核子页（通过 / 驳回 / 计数）
- [x] 共享 TS 类型迁移
- [x] 工期计算器(节假日 API + 多年缓存)
- [x] 全站 UI 美化(首页装饰背景 + 工期计算器视觉重做 + 二次元俏皮文案)
- [ ] 物流单号录入 + 订阅消息自动推送
- [ ] 内部子状态 `subStage`
- [ ] 上线发布

## 文案风格

客户端面向用户的提示统一走"二次元 / 俏皮"路线,管理后台保持简洁专业。

| 场景 | 文案 |
|---|---|
| 加载中 | 鼠鼠在搬数据~ / 鼠鼠正在算... |
| 加载失败 | 咦,加载迷路了 / 信号迷路了,再试一次? |
| 复制成功 | 复制好啦~ |
| 登录成功 | 欢迎回来呀~ ✨ |
| 提交成功 | 提交成功啦~ ✨ 订单已交给鼠鼠~ |
| 开发中入口 | 这个被你看到啦~ 还在赶工中 |
| 订单不存在 | 咦,这只订单跑掉啦~ |
| 空状态 | 还没有订单呢~ 快去定制一只吧 ✨ |

## 版权信息

© 2025 鼠鼠工坊 - 保留所有权利
