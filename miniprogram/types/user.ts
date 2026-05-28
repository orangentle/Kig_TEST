// miniprogram/types/user.ts
import type { BodyMeasurements } from './order';

/** 微信 / 自定义昵称-头像登录返回的基本信息 */
export interface UserInfo {
  avatarUrl: string;
  nickName: string;
  city?: string;
  country?: string;
  gender?: number;
  language?: string;
  province?: string;
}

/** 用户档案（users 集合） */
export interface UserProfile {
  _id?: string;
  _openid?: string;
  avatarUrl: string;
  nickName: string;
  userId?: string;
  taobaoName?: string;
  qq?: string;
  phone?: string;
  email?: string;
  bodyMeasurements?: BodyMeasurements;
  isAdmin?: boolean;
  createTime?: number | Date;
}
