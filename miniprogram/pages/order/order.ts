// order.ts
import Message from 'tdesign-miniprogram/message/index';
import type { BodyMeasurements } from '../../types/order';

const app = getApp<IAppOption>();

interface OrderFormData {
  tbOrderId: string;
  roleName: string;
  ip: string;
  height?: number;
  weight?: number;
  headCircumference?: number;
  shoulderWidth?: number;
  needAccessory: boolean;
  needReplaceFace: boolean;
  replaceFaceCount: number;
  isUrgent: boolean;
  remark: string;
}

Page({
  data: {
    formData: {
      tbOrderId: '',
      roleName: '',
      ip: '',
      height: 0,
      weight: 0,
      headCircumference: 0,
      shoulderWidth: 0,
      needAccessory: false,
      needReplaceFace: false,
      replaceFaceCount: 1,
      isUrgent: false,
      remark: ''
    } as OrderFormData,
    useProfileBodyData: true,
    profileBodyData: null as BodyMeasurements | null,
    referenceImages: [] as any[],     // 角色多视角图，最多 3 张
    replaceFaceImages: [] as any[],   // 替换脸图片，数量与 replaceFaceCount 对应
    faceCountOptions: [1, 2, 3],
    faceCountIndex: 0,
    isSubmitting: false,
    hasLogin: false,
    tbOrderIdHint: ''   // 订单号校验提示
  },

  onLoad() {
    this.checkLoginAndLoadData();
  },

  async checkLoginAndLoadData() {
    try {
      const { result } = await wx.cloud.callFunction({ name: 'login' }) as any;

      if (result && result.openid) {
        this.setData({ hasLogin: true });

        const db = wx.cloud.database();
        const userResult = await db.collection('users').where({
          _openid: result.openid
        }).get();

        if (userResult.data && userResult.data.length > 0) {
          const user = userResult.data[0] as any;
          if (user.bodyMeasurements) {
            this.setData({ profileBodyData: user.bodyMeasurements });
          }
        }
      } else {
        this.setData({ hasLogin: false });
        wx.showModal({
          title: '请先登录',
          content: '您需要先登录才能下单',
          showCancel: false,
          success: () => {
            wx.switchTab({ url: '/pages/profile/profile' });
          }
        });
      }
    } catch (error) {
      console.error('检查登录状态失败', error);
    }
  },

  onRoleNameChange(e: any) {
    this.setData({ 'formData.roleName': (e?.detail?.value ?? '') as string });
  },

  onTbOrderIdChange(e: any) {
    this.setData({
      'formData.tbOrderId': (e?.detail?.value ?? '') as string,
      tbOrderIdHint: ''
    });
  },

  async onTbOrderIdBlur() {
    const tbOrderId = (this.data.formData.tbOrderId || '').trim();
    if (!tbOrderId) return;
    try {
      const { result } = await wx.cloud.callFunction({
        name: 'getOrders',
        data: { tbOrderId }
      }) as any;
      const list = (result && result.data) || [];
      const conflict = list.find((o: any) => o.tbOrderId === tbOrderId);
      if (conflict) {
        this.setData({ tbOrderIdHint: '这个单号已经有小伙伴用过啦~ 确认一下是不是填错了？如需改动请联系客服喔 ♡' });
      } else {
        this.setData({ tbOrderIdHint: '' });
      }
    } catch (_) {
      // 网络异常静默，提交时云函数会再次拦截
    }
  },

  onIpChange(e: any) {
    this.setData({ 'formData.ip': (e?.detail?.value ?? '') as string });
  },

  onUseProfileBodyDataChange(e: any) {
    this.setData({ useProfileBodyData: e.detail.value });
  },

  onHeightChange(e: any) {
    this.setData({ 'formData.height': parseFloat(e.detail.value) || 0 });
  },

  onWeightChange(e: any) {
    this.setData({ 'formData.weight': parseFloat(e.detail.value) || 0 });
  },

  onHeadCircumferenceChange(e: any) {
    this.setData({ 'formData.headCircumference': parseFloat(e.detail.value) || 0 });
  },

  onShoulderWidthChange(e: any) {
    this.setData({ 'formData.shoulderWidth': parseFloat(e.detail.value) || 0 });
  },

  onAccessoryChange(e: any) {
    this.setData({ 'formData.needAccessory': e.detail.value });
  },

  onReplaceFaceChange(e: any) {
    const value = e.detail.value;
    this.setData({ 'formData.needReplaceFace': value });
    if (!value) {
      this.setData({ replaceFaceImages: [], 'formData.replaceFaceCount': 1, faceCountIndex: 0 });
    }
  },

  onFaceCountChange(e: any) {
    const idx = parseInt(e.detail.value);
    const count = this.data.faceCountOptions[idx];
    const trimmed = this.data.replaceFaceImages.slice(0, count);
    this.setData({
      faceCountIndex: idx,
      'formData.replaceFaceCount': count,
      replaceFaceImages: trimmed
    });
  },

  onUrgentChange(e: any) {
    const value = e.detail.value;
    if (value) {
      wx.showModal({
        title: '确认加急',
        content: '鼠鼠们会加班加点为你赶工，将额外收取 1200 元加急费。确认开启吗？',
        confirmText: '确认',
        cancelText: '取消',
        success: (res) => {
          if (res.confirm) {
            this.setData({ 'formData.isUrgent': true });
          }
        }
      });
    } else {
      this.setData({ 'formData.isUrgent': false });
    }
  },

  onRemarkChange(e: any) {
    this.setData({ 'formData.remark': e.detail.value });
  },

  onUploadAdd(e: any) {
    const { files } = e.detail;
    const next = [...this.data.referenceImages, ...files].slice(0, 3);
    this.setData({ referenceImages: next });
  },

  onUploadRemove(e: any) {
    const { index } = e.detail;
    const newImages = [...this.data.referenceImages];
    newImages.splice(index, 1);
    this.setData({ referenceImages: newImages });
  },

  onFaceUploadAdd(e: any) {
    const { files } = e.detail;
    const limit = this.data.formData.replaceFaceCount;
    const next = [...this.data.replaceFaceImages, ...files].slice(0, limit);
    this.setData({ replaceFaceImages: next });
  },

  onFaceUploadRemove(e: any) {
    const { index } = e.detail;
    const arr = [...this.data.replaceFaceImages];
    arr.splice(index, 1);
    this.setData({ replaceFaceImages: arr });
  },

  goToProfile() {
    wx.switchTab({ url: '/pages/profile/profile' });
  },

  validateForm(): boolean {
    const { formData, useProfileBodyData, profileBodyData, referenceImages, replaceFaceImages } = this.data;

    if (!formData.tbOrderId.trim()) {
      Message.warning({ context: this, offset: [20, 32], content: '鼠鼠找不到淘宝订单号呜呜~ 先填一下嘛 (｡>﹏<｡)' });
      return false;
    }
    if (!formData.roleName.trim()) {
      Message.warning({ context: this, offset: [20, 32], content: '小可爱还没有名字哦~ 给ta取一个吧 ♡' });
      return false;
    }
    if (!formData.ip.trim()) {
      Message.warning({ context: this, offset: [20, 32], content: '角色来自哪个作品呀? 鼠鼠想知道~ (◍•ᴗ•◍)' });
      return false;
    }

    if (useProfileBodyData) {
      if (!profileBodyData || !profileBodyData.height) {
        Message.warning({ context: this, offset: [20, 32], content: '个人资料里还没身材数据呢~ 先去补一下嘛 ♡' });
        return false;
      }
    } else {
      if (!formData.height || !formData.headCircumference) {
        Message.warning({ context: this, offset: [20, 32], content: '身高和头围是必须的喔~ 不然鼠鼠没法量 (｡•́︿•̀｡)' });
        return false;
      }
    }

    if (referenceImages.length === 0) {
      Message.warning({ context: this, offset: [20, 32], content: '至少要一张参考图嘛~ 鼠鼠才知道要捏成什么样子呀 (˃ ⌑ ˂ഃ )' });
      return false;
    }

    if (formData.needReplaceFace && replaceFaceImages.length < formData.replaceFaceCount) {
      Message.warning({
        context: this, offset: [20, 32],
        content: `还差 ${formData.replaceFaceCount - replaceFaceImages.length} 张替换脸图哦~ 一脸一图才不会认错呀 ♡`
      });
      return false;
    }

    return true;
  },

  async uploadFiles(files: any[], prefix: string): Promise<string[]> {
    const urls: string[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const filePath = file.url || file.path;
      if (filePath.startsWith('cloud://')) {
        urls.push(filePath);
        continue;
      }
      const timestamp = Date.now();
      const cloudPath = `orders/${prefix}/${timestamp}_${i}.${filePath.split('.').pop() || 'jpg'}`;
      const r = await wx.cloud.uploadFile({ cloudPath, filePath });
      urls.push(r.fileID);
    }
    return urls;
  },

  async onSubmit() {
    if (this.data.isSubmitting) return;

    if (!this.data.hasLogin) {
      wx.showModal({
        title: '请先登录',
        content: '您需要先登录才能下单',
        showCancel: false,
        success: () => { wx.switchTab({ url: '/pages/profile/profile' }); }
      });
      return;
    }

    if (!this.validateForm()) return;

    const roleNameTrim = (this.data.formData.roleName || '').trim();
    const ipTrim = (this.data.formData.ip || '').trim();
    const tbOrderIdTrim = (this.data.formData.tbOrderId || '').trim();

    this.setData({ isSubmitting: true });

    // 请求订阅消息授权（已排单 / 发货 两个节点）
    const QUEUED_TMPL_ID   = 'hNOQznZnaw3VR7Bcnmnv0HJBO1F8P_kxIGpIACi3VTk';
    const SHIPPING_TMPL_ID = 'rJMkotC0cffPQfvmFi_kwhfFcCo5fQcmFfa3ai_xkFk';
    try {
      await new Promise((resolve) => {
        wx.requestSubscribeMessage({
          tmplIds: [QUEUED_TMPL_ID, SHIPPING_TMPL_ID],
          complete: () => resolve(null)
        });
      });
    } catch (_) { /* 忽略授权失败 */ }

    try {
      wx.showLoading({ title: '提交中...' });

      const [referenceImageUrls, replaceFaceImageUrls] = await Promise.all([
        this.uploadFiles(this.data.referenceImages, 'reference'),
        this.data.formData.needReplaceFace
          ? this.uploadFiles(this.data.replaceFaceImages, 'replace-face')
          : Promise.resolve([] as string[])
      ]);

      let bodyData: BodyMeasurements;
      if (this.data.useProfileBodyData && this.data.profileBodyData) {
        bodyData = this.data.profileBodyData;
      } else {
        bodyData = {
          height: this.data.formData.height,
          weight: this.data.formData.weight,
          headCircumference: this.data.formData.headCircumference,
          shoulderWidth: this.data.formData.shoulderWidth
        };
      }

      const { result } = await wx.cloud.callFunction({
        name: 'submitOrder',
        data: {
          tbOrderId: tbOrderIdTrim,
          roleName: roleNameTrim,
          ip: ipTrim,
          bodyMeasurements: bodyData,
          referenceImages: referenceImageUrls,
          replaceFaceImages: replaceFaceImageUrls,
          options: {
            needAccessory: this.data.formData.needAccessory,
            needReplaceFace: this.data.formData.needReplaceFace,
            replaceFaceCount: this.data.formData.needReplaceFace ? this.data.formData.replaceFaceCount : 0,
            isUrgent: this.data.formData.isUrgent
          },
          remark: this.data.formData.remark
        }
      }) as any;

      wx.hideLoading();

      if (result && result.success) {
        wx.showModal({
          title: '提交成功啦~ ✨',
          content: '订单已交给鼠鼠~ 已锁定等待管理员审核哦,需要改动的话联系客服解锁就好啦~',
          showCancel: false,
          success: () => { wx.switchTab({ url: '/pages/profile/profile' }); }
        });
      } else {
        throw new Error(result?.error || '提交失败');
      }
    } catch (error: any) {
      wx.hideLoading();
      console.error('提交订单失败', error);
      Message.error({ context: this, offset: [20, 32], content: error.message || '提交失败惹~ 鼠鼠再试一次嘛 (｡•́︿•̀｡)' });
    } finally {
      this.setData({ isSubmitting: false });
    }
  }
});
