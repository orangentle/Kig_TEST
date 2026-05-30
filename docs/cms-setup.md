# 微信云开发 CMS 配置清单

> 配 CMS 时照着这份抄。Demo 在个人版云环境,正式上线前需迁移到工作室主账号 — 迁移步骤见文末。

## 0. 启用 CMS

1. 微信云开发控制台 → 选目标环境 → 左侧菜单「内容管理」→ 一键开通
2. 首次开通会自动创建一个 `tcb-ext-cms-*` 的子环境用于存 CMS 自身数据,不影响业务库
3. 开通后会拿到一个 CMS 后台访问地址,客服扫码登录即可

---

## 1. 内容模型(集合)

需要为以下三个集合各建一个内容模型:

- `orders` — 订单
- `works` — 作品橱窗
- `users` — 用户

字段表见下文。

### 1.1 `orders` 订单

| 字段名称 | 字段标识 | 字段描述 | 数据类型 | 格式 | 默认值 | 必填 | 唯一 | 主展示 |
|---|---|---|---|---|---|---|---|---|
| 系统单号 | orderId | 自动生成,KIG 开头 | 单行字符串 | — | — | ✅ | ✅ |  |
| 淘宝订单号 | tbOrderId | 业务主键,客户填 | 单行字符串 | — | — | ✅ | ✅ | ✅ |
| 排单号 | queueNumber | RatStudio-YYYY-NNN | 单行字符串 | — | — |  |  |  |
| 用户 openid | _openid | 系统自动写入 | 单行字符串 | 只读 | — | ✅ |  |  |
| 客户称呼 | customerName | 通常等于微信昵称 | 单行字符串 | — | — |  |  |  |
| 角色名 | roleName | 定制角色 | 单行字符串 | — | — | ✅ |  |  |
| 角色出处 IP | ip | 原神/星铁/原创 | 单行字符串 | — | — |  |  |  |
| 业务状态 | status | 订单状态 | 枚举 | pending=待审核; normal=正常; urgent=加急; canceled=已取消; completed=已完成 | pending | ✅ |  |  |
| 制作阶段 | stage | 5 阶段流程 | 枚举 | pending=待审核; queued=已排单; modeling=建模; painting=上妆; hair=假毛; shipped=已发货 | pending | ✅ |  |  |
| 阶段中文 | progressStage | UI 显示用 | 单行字符串 | — | 待审核 |  |  |  |
| 进度百分比 | progressPercent | 0~100 | 数字 | 0-100 | 0 |  |  |  |
| 是否加急 | isUrgent | +1200 元 | 布尔 | — | false |  |  |  |
| 是否锁定 | isLocked | 锁定后客户改不了 | 布尔 | — | true |  |  |  |
| 是否归档 | isArchived | 完结后归档 | 布尔 | — | false |  |  |  |
| 锁定时间 | lockedAt | — | 时间 | — | — |  |  |  |
| 下单日期 | orderTime | YYYY-MM-DD | 单行字符串 | — | — |  |  |  |
| 截止日期 | deadline | YYYY-MM-DD | 单行字符串 | — | — |  |  |  |
| 创建时间 | createTime | 系统写入 | 时间 | — | 当前时间 | ✅ |  |  |
| 更新时间 | updateTime | 系统写入 | 时间 | — | 当前时间 |  |  |  |
| 用户快照 | userInfo | 下单时存的联系方式 | JSON 对象 | { nickName, avatarUrl, phone, taobaoName, qq } | — |  |  |  |
| 身材数据 | bodyMeasurements | — | JSON 对象 | { height, weight, headCircumference, shoulderWidth } | — |  |  |  |
| 定制选项 | options | — | JSON 对象 | { needAccessory, needReplaceFace, replaceFaceCount, isUrgent } | — |  |  |  |
| 角色参考图 | referenceImages | 最多 3 张 | 图片数组 | cloud:// | [] |  |  |  |
| 替换脸图 | replaceFaceImages | — | 图片数组 | cloud:// | [] |  |  |  |
| 备注 | remark | 客户特殊要求 | 多行文本 | — | — |  |  |  |
| 审核信息 | reviewInfo | — | JSON 对象 | { reviewTime, reviewBy, reviewRemark } | — |  |  |  |
| 取消信息 | cancelInfo | 用户主动取消时 | JSON 对象 | { cancelTime, cancelBy, cancelReason } | — |  |  |  |

### 1.2 `works` 作品橱窗

| 字段名称 | 字段标识 | 字段描述 | 数据类型 | 格式 | 默认值 | 必填 | 唯一 | 主展示 |
|---|---|---|---|---|---|---|---|---|
| 角色名 | roleName | 作品角色 | 单行字符串 | — | — | ✅ |  | ✅ |
| 出处来源 | source | 原神/自设/初音… | 单行字符串 | — | — |  |  |  |
| 分类 | category | 决定徽章颜色 | 枚举 | original=自设; game=游戏; anime=动漫 | original | ✅ |  |  |
| 封面图 | coverFileId | 主封面 | 图片 | cloud:// 单张 | — | ✅ |  |  |
| 是否上架 | isPublished | 客户端只显示上架的 | 布尔 | — | false | ✅ |  |  |
| 创建时间 | createTime | 列表倒序 | 时间 | — | 当前时间 | ✅ |  |  |

### 1.3 `users` 用户

| 字段名称 | 字段标识 | 字段描述 | 数据类型 | 格式 | 默认值 | 必填 | 唯一 | 主展示 |
|---|---|---|---|---|---|---|---|---|
| openid | _openid | 微信用户唯一标识,系统写入 | 单行字符串 | 只读 | — | ✅ | ✅ |  |
| 微信昵称 | nickName | — | 单行字符串 | — | — |  |  | ✅ |
| 头像 | avatarUrl | cloud:// 或 https:// | 图片 | — | — |  |  |  |
| 淘宝旺旺 | taobaoName | 客服联系用 | 单行字符串 | — | — |  |  |  |
| 手机号 | phone | — | 单行字符串 | ^1\d{10}$ | — |  |  |  |
| QQ | qq | — | 单行字符串 | — | — |  |  |  |
| 身材数据 | bodyMeasurements | 个人资料里填的 | JSON 对象 | { height, weight, headCircumference, shoulderWidth } | — |  |  |  |
| 是否管理员 | isAdmin | 客服/管理员开关 | 布尔 | — | false |  |  |  |
| 待办驳回通知 | pendingRejectNotices | 客服驳回后系统写入 | JSON 对象数组 | [{ tbOrderId, roleName, orderId, reason, rejectTime }] | [] |  |  |  |
| 注册时间 | createTime | — | 时间 | — | 当前时间 |  |  |  |
| 更新时间 | updateTime | — | 时间 | — | 当前时间 |  |  |  |

### 配置时小提醒

- **字段标识**必须跟代码里完全一致(包括 `_openid` 前面的下划线),改错了 CMS 就读不到数据
- **字段名称**只是 CMS UI 显示给客服看的中文,随便改不影响代码
- **唯一**只是 CMS 提交时的软校验,真正的唯一保护靠 DB 索引(`tbOrderId` 已加 `idx_tbOrderId`)
- **主展示列**整张表只勾一个,客服在列表里一眼看到的就是这列
- 嵌套对象用 JSON 类型即可,不用拆成子字段(否则配起来太累)

---

## 2. 数据库权限(数据库 → 集合 → 权限设置)

CMS 权限和数据库权限**是两套独立设置**:

- **数据库权限**: 控制小程序客户端直接读写时,谁能读/写哪些行
- **CMS 权限**: 控制谁能登录 CMS 后台,登进去就有该集合的全部读写权(走云函数级权限,自动绕过数据库权限)

### 推荐配置

| 集合 | 选这个预设 | 理由 |
|---|---|---|
| `orders` | 仅创建者可读写 | 客户只能看/改自己的订单,客服走云函数(`batchUpdateOrders`)绕过 |
| `users` | 仅创建者可读写 | 每个用户只能读写自己那一行 |
| `works` | 所有用户可读,仅管理端可写 | 客户端展示给所有人看,上架走云函数 |

> `submitOrder` 是云函数自带 admin 权限,不受影响;批量审核同理。

---

## 3. CMS 角色

| 角色 | 能干啥 | 给谁 |
|---|---|---|
| 管理员 | 改 schema、删集合、所有数据 CRUD | 你(开发者)自己 |
| 运营者 | 所有内容 CRUD,不能改 schema | 资深客服 / 主管 |
| 协作者 | 只能改指定集合的内容 | 兼职客服 / 志愿者 |

**鼠鼠工坊建议**: 你 = 管理员,客服 = 运营者。**先不开字段级权限**,反正客服本来就要看客户完整信息。

---

## 4. 个人版 → 工作室主账号迁移

正式上线前需要把 Demo 从个人版迁过去。步骤:

### 4.1 主账号侧准备

1. 主账号开通云开发,拿到新 `envId`(如 `shushugongfang-prod-xxx`)
2. 在新环境**预先创建好** `orders` / `users` / `works` 三个集合(可空,先建集合名就行)
3. 在新环境**重建唯一索引**: `orders.idx_tbOrderId`(tbOrderId 升序,唯一)

### 4.2 数据迁移

云开发控制台支持数据导出/导入:
1. 个人版环境 → 数据库 → 选集合 → 右上角「导出」→ JSON Lines
2. 主账号环境 → 同集合 → 「导入」→ 上传刚才导出的文件
3. 三个集合都过一遍

> **云存储文件**(`cloud://...` 那些图片):需要单独导出后上传到主账号的云存储,且**文件 ID 会变** — 订单里的 `referenceImages` 字段需要批量替换为新 ID。可以写个一次性脚本跑批,或者干脆**老订单保留旧 fileID 失效不查问题不大**(已发货归档单),只迁正在生产中的订单的图片。

### 4.3 代码侧切换

需要改三处 `envId`:

1. `miniprogram/app.ts` — 小程序云初始化
2. `cloudfunctions/cloud-env.json` — 云函数运行时环境
3. `cloudfunctions/submitOrder/index.js` 第 5 行附近 `cloud.init({ env: '...' })` — 这个云函数硬编码了 envId,需手改

```bash
# 全局搜一下还有没有别的硬编码 envId
grep -r "shushugongfang-d2gl1995c27f9730e" .
```

### 4.4 云函数重新部署

主账号环境是全新的,所有云函数都要重新部署一遍:

```
adminAuth / batchUpdateOrders / cancelOrder / clearOrders / createOrder /
getOpenId / getOrderDetail / getOrders / initOrderData / login /
submitOrder / updateAvatar
```

### 4.5 CMS 重建

CMS 内容模型不会跟着环境迁移,需在主账号侧按本文档第 1 节重新配一遍。

### 4.6 上线前验收清单

- [ ] 三个集合都能正常读取
- [ ] `idx_tbOrderId` 唯一索引已建
- [ ] 提交订单 → 检查云存储图片能正常显示
- [ ] 管理员审核通过/驳回 → 客户端弹窗正常
- [ ] 取消订单流程正常
- [ ] 订阅消息模板 ID(`QUEUED_TMPL_ID` / `SHIPPING_TMPL_ID`)在新公众平台后台重新申请并替换
