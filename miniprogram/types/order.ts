// miniprogram/types/order.ts
// 订单相关共享类型，所有页面统一从此处 import

/** 订单业务状态（写入数据库 `status` 字段） */
export type OrderStatus =
  | 'pending'    // 已提交，等待管理员审核
  | 'normal'     // 审核通过，正常生产中
  | 'urgent'     // 加急（与 isUrgent 等价的状态标记）
  | 'rejected'   // 已驳回
  | 'completed'; // 已完成（最终态）

/** 订单制作阶段（写入数据库 `stage` 字段） */
export type OrderStage =
  | 'pending'  // 待审核 / 已驳回
  | 'queued'   // 已排单
  | 'modeling' // 建模
  | 'painting' // 上妆
  | 'hair'     // 假毛
  | 'shipped'; // 已发货

/** 身材数据 */
export interface BodyMeasurements {
  height?: number;            // 身高(cm)
  weight?: number;            // 体重(kg)
  headCircumference?: number; // 头围(cm)
  shoulderWidth?: number;     // 肩宽(cm)
}

/** 定制选项 */
export interface OrderOptions {
  needAccessory?: boolean;
  needHeadwear?: boolean;
  needReplaceFace?: boolean;
  replaceFaceCount?: number;
  isUrgent?: boolean;
}

/** 用户信息（订单内嵌一份快照） */
export interface OrderUserInfo {
  nickName?: string;
  avatarUrl?: string;
  taobaoName?: string;
  qq?: string;
  phone?: string;
}

/** 审核信息 */
export interface ReviewInfo {
  reviewTime?: Date | string;
  reviewBy?: string;
  reviewRemark?: string;
}

/** 完整订单（与云数据库 `orders` 集合对齐） */
export interface Order {
  _id?: string;
  _openid?: string;
  // 基本
  orderId: string;            // 系统单号 KIG/KG…
  tbOrderId: string;          // 淘宝订单号（业务主键）
  queueNumber?: string;       // 排单号 RatStudio-YYYY-NNN
  customerName?: string;
  roleName: string;
  ip?: string;                // 角色出处 IP
  sourceWork?: string;        // 旧字段：来源作品（仅读取兼容）
  // 时间
  orderTime?: string;
  deadline?: string;
  createTime?: Date | number | string;
  updateTime?: Date | string;
  // 进度
  stage: OrderStage;
  progressStage: string;      // 中文：已排单 / 建模 / ...
  progressPercent: number;
  // 状态
  status: OrderStatus;
  isUrgent?: boolean;
  isArchived?: boolean;
  isLocked?: boolean;
  // 详情
  userInfo?: OrderUserInfo;
  bodyMeasurements?: BodyMeasurements;
  options?: OrderOptions;
  referenceImages?: string[];
  replaceFaceImages?: string[];
  remark?: string;
  reviewInfo?: ReviewInfo;
}

/** 阶段定义（推进流程） */
export interface StageDef {
  value: OrderStage;
  label: string;
  percent: number;
}

export const STAGE_FLOW: StageDef[] = [
  { value: 'queued',   label: '已排单', percent: 10 },
  { value: 'modeling', label: '建模',   percent: 30 },
  { value: 'painting', label: '上妆',   percent: 55 },
  { value: 'hair',     label: '假毛',   percent: 80 },
  { value: 'shipped',  label: '已发货', percent: 100 }
];
