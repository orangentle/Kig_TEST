// miniprogram/types/gathering.ts
// 偶壳娃聚模块共享类型，所有娃聚相关页面统一从此处 import。
//
// 设计约定（与 order.ts 一致）：
// - 每个集合先定义「数据库文档」接口（落库字段，云函数读写以此为准）；
// - 再定义「视图模型」接口（在文档基础上扩展前端计算字段，如中文状态文案、
//   日历用的 month/day 拆分等，这些字段不入库，由页面在 load 时拼装）。
//
// 对应云数据库集合（建议命名）：
//   gathering_events    活动
//   gathering_signups   报名记录
//   gathering_albums    相册（按活动分）
//   gathering_photos    照片
//   gathering_products  周边商品
//   gathering_banners   轮播图（娃聚主页 / 周边商城共用，用 scene 区分）

// ============================================================
// 枚举与常量
// ============================================================

/** 活动报名状态（写入 events.status 字段） */
export type EventStatus =
  | 'open'      // 报名中
  | 'upcoming'  // 即将开始（未开放报名）
  | 'closed'    // 已结束
  | 'full';     // 已报满

/** 活动类型（写入 events.type 字段） */
export type EventType =
  | 'gathering' // 娃聚
  | 'photo'     // 摄影
  | 'workshop'  // 工作坊
  | 'market';   // 市集

/** 报名记录状态（写入 signups.status 字段） */
export type SignupStatus =
  | 'confirmed' // 已确认参加
  | 'pending'   // 待确认
  | 'cancelled';// 已取消

/** 轮播图所属场景（写入 banners.scene 字段） */
export type BannerScene =
  | 'gathering' // 娃聚主页
  | 'merch';    // 周边商城

/** 周边商品分类 id（写入 products.categoryId 字段） */
export type ProductCategoryId =
  | 'clothing'   // 娃衣
  | 'accessory'  // 配饰
  | 'prop'       // 道具
  | 'stationery' // 文具
  | 'bag'        // 包袋
  | 'other';     // 其他

/** 活动类型定义（含中文文案） */
export interface EventTypeDef {
  value: EventType;
  label: string;
}

export const EVENT_TYPES: EventTypeDef[] = [
  { value: 'gathering', label: '娃聚' },
  { value: 'photo',     label: '摄影' },
  { value: 'workshop',  label: '工作坊' },
  { value: 'market',    label: '市集' }
];

/** 活动状态文案映射 */
export const EVENT_STATUS_TEXT: Record<EventStatus, string> = {
  open:     '报名中',
  upcoming: '即将开始',
  closed:   '已结束',
  full:     '已报满'
};

/** 报名状态文案映射 */
export const SIGNUP_STATUS_TEXT: Record<SignupStatus, string> = {
  confirmed: '已确认',
  pending:   '待确认',
  cancelled: '已取消'
};

/** 商品分类定义（icon 为 t-icon 名称） */
export interface ProductCategoryDef {
  id: ProductCategoryId | 'all';
  name: string;
  icon: string;
}

export const PRODUCT_CATEGORIES: ProductCategoryDef[] = [
  { id: 'all',        name: '全部', icon: 'view-module' },
  { id: 'clothing',   name: '娃衣', icon: 'relativity' },
  { id: 'accessory',  name: '配饰', icon: 'gift' },
  { id: 'prop',       name: '道具', icon: 'tools' },
  { id: 'stationery', name: '文具', icon: 'edit' },
  { id: 'bag',        name: '包袋', icon: 'wallet' },
  { id: 'other',      name: '其他', icon: 'ellipsis' }
];

// ============================================================
// 活动 events
// ============================================================

/** 活动议程单项 */
export interface AgendaItem {
  time: string;    // 如 '10:00-11:30'
  content: string; // 可含换行
}

/** 活动数据库文档（gathering_events 集合） */
export interface GatheringEvent {
  _id?: string;
  _openid?: string;          // 创建者（管理员）
  eventId: string;           // 业务 id，如 'event_001'
  name: string;              // 活动名称
  coverFileId?: string;      // 封面云存储 fileID
  date: string;              // 开始日期 YYYY-MM-DD
  endDate?: string;          // 结束日期（多日活动）
  startTime?: string;        // 开始时间，或 '待定'
  endTime?: string;          // 结束时间，或 '待定'
  location: string;          // 地点
  host?: string;             // 主办方
  type: EventType;
  status: EventStatus;
  capacity?: number;         // 名额上限，0 / 不填表示不限
  signupCount?: number;      // 当前报名数
  fee?: number;              // 报名费用（元），0 为免费
  description?: string;      // 活动介绍
  agenda?: AgendaItem[];     // 议程
  isPublished?: boolean;     // 是否上架展示
  createTime?: Date | number | string;
  updateTime?: Date | string;
}

/**
 * 活动视图模型：在文档基础上扩展前端展示用计算字段。
 * 这些字段不落库，由页面 load 时根据文档与当前时间拼装。
 */
export interface GatheringEventView extends GatheringEvent {
  coverUrl?: string;     // 由 coverFileId 解析出的临时链接
  statusText?: string;   // EVENT_STATUS_TEXT[status]
  typeText?: string;     // EVENT_TYPES 中的 label
  // 日历 / 卡片展示用的日期拆分
  month?: string;        // '05'
  day?: string;          // '02'
  endDay?: string;       // '03'
  dateText?: string;     // '2026年5月2日-3日（暂定）'
  monthText?: string;    // '5月'
  dayNum?: number;       // 2
  weekday?: string;      // '周六'
  remainingSpots?: number | string; // 剩余名额，不限时为 '不限'
  canSignup?: boolean;   // 是否可报名（status==='open' 且未满）
  hasSignup?: boolean;   // 当前用户是否已报名
  fullText?: string;     // 报满提示
}

// ============================================================
// 报名记录 signups
// ============================================================

/** 报名记录数据库文档（gathering_signups 集合） */
export interface GatheringSignup {
  _id?: string;
  _openid?: string;        // 报名用户
  signupId: string;        // 业务 id
  eventId: string;         // 关联活动 eventId
  eventName: string;       // 活动名称快照
  eventDate: string;       // 活动日期快照
  contactName?: string;    // 联系人
  contactPhone?: string;   // 联系电话
  qq?: string;
  remark?: string;         // 备注
  status: SignupStatus;
  signupTime?: Date | number | string;
  updateTime?: Date | string;
}

/** 报名记录视图模型 */
export interface GatheringSignupView extends GatheringSignup {
  statusText?: string;     // SIGNUP_STATUS_TEXT[status]
}

// ============================================================
// 相册 albums / 照片 photos
// ============================================================

/** 相册数据库文档（gathering_albums 集合，按活动分） */
export interface GatheringAlbum {
  _id?: string;
  albumId: string;         // 业务 id，通常与 eventId 对应
  eventId?: string;        // 关联活动
  name: string;            // 相册名（活动名）
  coverFileId?: string;    // 封面云存储 fileID
  photoCount?: number;     // 照片数（冗余计数）
  date?: string;           // 活动日期 YYYY-MM-DD
  isPublished?: boolean;
  createTime?: Date | number | string;
}

/** 相册视图模型 */
export interface GatheringAlbumView extends GatheringAlbum {
  coverUrl?: string;       // 由 coverFileId 解析
}

/** 照片数据库文档（gathering_photos 集合） */
export interface GatheringPhoto {
  _id?: string;
  _openid?: string;        // 上传者
  photoId: string;         // 业务 id，如 'p001'
  albumId: string;         // 所属相册
  fileId: string;          // 原图云存储 fileID
  thumbnailFileId?: string;// 缩略图 fileID
  userName?: string;       // 上传者昵称快照
  userAvatar?: string;     // 上传者头像快照
  description?: string;    // 照片描述
  likes?: number;          // 点赞数
  likedBy?: string[];      // 点赞用户 openid 列表（用于判断 isLiked）
  comments?: number;       // 评论数
  isAudited?: boolean;     // 是否审核通过（UGC 内容审核）
  uploadTime?: Date | number | string;
}

/** 照片视图模型：扩展瀑布流 / 详情展示字段 */
export interface GatheringPhotoView extends GatheringPhoto {
  url?: string;            // 由 fileId 解析的临时链接
  thumbnailUrl?: string;   // 由 thumbnailFileId 解析
  uploadTimeText?: string; // 格式化时间，如 '2025-10-01 15:30'
  isLiked?: boolean;       // 当前用户是否已点赞
  originalIndex?: number;  // 瀑布流分列前的原始索引（用于预览定位）
}

// ============================================================
// 周边商品 products / 轮播图 banners
// ============================================================

/** 周边商品数据库文档（gathering_products 集合） */
export interface GatheringProduct {
  _id?: string;
  productId: string;       // 业务 id，如 'prod_001'
  name: string;
  desc?: string;           // 简介
  mainFileId?: string;     // 主图 fileID
  detailFileIds?: string[];// 详情图 fileID 列表
  price: number;           // 现价（元）
  originalPrice?: number;  // 原价（划线价）
  sales?: number;          // 销量
  tag?: string;            // 角标，如 '热卖' / '新品'
  categoryId: ProductCategoryId;
  specs?: string[];        // 规格 / 款式可选项
  content?: string;        // 图文详情（可含换行）
  isOnSale?: boolean;      // 是否上架
  createTime?: Date | number | string;
  updateTime?: Date | string;
}

/** 周边商品视图模型 */
export interface GatheringProductView extends GatheringProduct {
  imageUrl?: string;       // 由 mainFileId 解析
  images?: string[];       // 由 mainFileId + detailFileIds 解析
}

/** 轮播图数据库文档（gathering_banners 集合，娃聚主页与周边商城共用） */
export interface GatheringBanner {
  _id?: string;
  bannerId: string;        // 业务 id
  scene: BannerScene;      // 所属场景
  fileId: string;          // 图片 fileID
  title?: string;
  desc?: string;
  linkUrl?: string;        // 点击跳转（可选）
  sort?: number;           // 排序权重，越小越靠前
  isPublished?: boolean;
  createTime?: Date | number | string;
}

/** 轮播图视图模型 */
export interface GatheringBannerView extends GatheringBanner {
  imageUrl?: string;       // 由 fileId 解析
}
