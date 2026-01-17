// order.ts
import Message from 'tdesign-miniprogram/message/index';

// 获取应用实例
const app = getApp<IAppOption>();

// 表单数据接口
interface OrderFormData {
  roleName: string;
  sourceWork: string;
  // 身材数据
  height?: number;
  weight?: number;
  headCircumference?: number;
  neckCircumference?: number;
  shoulderWidth?: number;
  // 定制选项
  needReplaceFace: boolean;
  needHeadwear: boolean;
  needAntiGravity: boolean;
  needCornsilkPerm: boolean;
  isUrgent: boolean;
  // 备注
  remark: string;
}

// 身材数据接口
interface BodyMeasurements {
  height?: number;
  weight?: number;
  headCircumference?: number;
  neckCircumference?: number;
  shoulderWidth?: number;
}

Page({
  data: {
    formData: {
      roleName: '',
      sourceWork: '',
      height: 0,
      weight: 0,
      headCircumference: 0,
      neckCircumference: 0,
      shoulderWidth: 0,
      needReplaceFace: false,
      needHeadwear: false,
      needAntiGravity: false,
      needCornsilkPerm: false,
      isUrgent: false,
      remark: ''
    } as OrderFormData,
    useProfileBodyData: true,  // 默认使用个人资料身材数据
    profileBodyData: null as BodyMeasurements | null,
    referenceImages: [] as any[],  // 参考图片列表
    isSubmitting: false,
    hasLogin: false
  },

  onLoad() {
    // 检查登录状态并加载用户身材数据
    this.checkLoginAndLoadData();
  },

  // 检查登录并加载数据
  async checkLoginAndLoadData() {
    try {
      // 获取云开发用户信息
      const { result } = await wx.cloud.callFunction({
        name: 'login',
      }) as any;
      
      if (result && result.openid) {
        this.setData({ hasLogin: true });
        
        // 查询用户身材数据
        const db = wx.cloud.database();
        const userResult = await db.collection('users').where({
          _openid: result.openid
        }).get();
        
        if (userResult.data && userResult.data.length > 0) {
          const user = userResult.data[0] as any;
          if (user.bodyMeasurements) {
            this.setData({
              profileBodyData: user.bodyMeasurements
            });
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

  // 角色名称变更
  onRoleNameChange(e: any) {
    this.setData({ 'formData.roleName': e.detail.value });
  },

  // 来源作品变更
  onSourceWorkChange(e: any) {
    this.setData({ 'formData.sourceWork': e.detail.value });
  },

  // 是否使用个人资料身材数据
  onUseProfileBodyDataChange(e: any) {
    this.setData({ useProfileBodyData: e.detail.value });
  },

  // 身材数据变更
  onHeightChange(e: any) {
    this.setData({ 'formData.height': parseFloat(e.detail.value) || 0 });
  },

  onWeightChange(e: any) {
    this.setData({ 'formData.weight': parseFloat(e.detail.value) || 0 });
  },

  onHeadCircumferenceChange(e: any) {
    this.setData({ 'formData.headCircumference': parseFloat(e.detail.value) || 0 });
  },

  onNeckCircumferenceChange(e: any) {
    this.setData({ 'formData.neckCircumference': parseFloat(e.detail.value) || 0 });
  },

  onShoulderWidthChange(e: any) {
    this.setData({ 'formData.shoulderWidth': parseFloat(e.detail.value) || 0 });
  },

  // 定制选项变更
  onReplaceFaceChange(e: any) {
    this.setData({ 'formData.needReplaceFace': e.detail.value });
  },

  onHeadwearChange(e: any) {
    this.setData({ 'formData.needHeadwear': e.detail.value });
  },

  onAntiGravityChange(e: any) {
    this.setData({ 'formData.needAntiGravity': e.detail.value });
  },

  onCornsilkPermChange(e: any) {
    this.setData({ 'formData.needCornsilkPerm': e.detail.value });
  },

  // 加急选项变更
  onUrgentChange(e: any) {
    const value = e.detail.value;
    if (value) {
      wx.showModal({
        title: '确认加急',
        content: '加急服务将额外收取1000元费用，订单将优先制作。确认开启加急服务吗？',
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

  // 备注变更
  onRemarkChange(e: any) {
    this.setData({ 'formData.remark': e.detail.value });
  },

  // 图片上传
  onUploadAdd(e: any) {
    const { files } = e.detail;
    this.setData({
      referenceImages: [...this.data.referenceImages, ...files]
    });
  },

  // 图片删除
  onUploadRemove(e: any) {
    const { index } = e.detail;
    const newImages = [...this.data.referenceImages];
    newImages.splice(index, 1);
    this.setData({ referenceImages: newImages });
  },

  // 去完善个人资料
  goToProfile() {
    wx.switchTab({ url: '/pages/profile/profile' });
  },

  // 表单验证
  validateForm(): boolean {
    const { formData, useProfileBodyData, profileBodyData, referenceImages } = this.data;
    
    if (!formData.roleName.trim()) {
      Message.error({ context: this, offset: [20, 32], content: '请输入角色名称' });
      return false;
    }
    
    if (!formData.sourceWork.trim()) {
      Message.error({ context: this, offset: [20, 32], content: '请输入来源作品' });
      return false;
    }
    
    // 检查身材数据
    if (useProfileBodyData) {
      if (!profileBodyData || !profileBodyData.height) {
        Message.error({ context: this, offset: [20, 32], content: '请先完善个人资料中的身材数据' });
        return false;
      }
    } else {
      if (!formData.height || !formData.headCircumference) {
        Message.error({ context: this, offset: [20, 32], content: '请填写完整的身材数据' });
        return false;
      }
    }
    
    if (referenceImages.length === 0) {
      Message.error({ context: this, offset: [20, 32], content: '请上传至少一张表情参考图' });
      return false;
    }
    
    return true;
  },

  // 上传图片到云存储
  async uploadImages(): Promise<string[]> {
    const uploadedUrls: string[] = [];
    
    for (let i = 0; i < this.data.referenceImages.length; i++) {
      const file = this.data.referenceImages[i];
      const filePath = file.url || file.path;
      
      // 如果已经是云存储地址，直接使用
      if (filePath.startsWith('cloud://')) {
        uploadedUrls.push(filePath);
        continue;
      }
      
      try {
        const timestamp = Date.now();
        const cloudPath = `orders/reference/${timestamp}_${i}.${filePath.split('.').pop() || 'jpg'}`;
        
        const uploadResult = await wx.cloud.uploadFile({
          cloudPath,
          filePath
        });
        
        uploadedUrls.push(uploadResult.fileID);
      } catch (error) {
        console.error('上传图片失败', error);
        throw error;
      }
    }
    
    return uploadedUrls;
  },

  // 提交订单
  async onSubmit() {
    if (!this.data.hasLogin) {
      wx.showModal({
        title: '请先登录',
        content: '您需要先登录才能下单',
        showCancel: false,
        success: () => {
          wx.switchTab({ url: '/pages/profile/profile' });
        }
      });
      return;
    }
    
    if (!this.validateForm()) {
      return;
    }
    
    this.setData({ isSubmitting: true });
    
    try {
      wx.showLoading({ title: '提交中...' });
      
      // 上传参考图片
      const imageUrls = await this.uploadImages();
      
      // 准备身材数据
      let bodyData: BodyMeasurements;
      if (this.data.useProfileBodyData && this.data.profileBodyData) {
        bodyData = this.data.profileBodyData;
      } else {
        bodyData = {
          height: this.data.formData.height,
          weight: this.data.formData.weight,
          headCircumference: this.data.formData.headCircumference,
          neckCircumference: this.data.formData.neckCircumference,
          shoulderWidth: this.data.formData.shoulderWidth
        };
      }
      
      // 调用云函数提交订单
      const { result } = await wx.cloud.callFunction({
        name: 'submitOrder',
        data: {
          roleName: this.data.formData.roleName,
          sourceWork: this.data.formData.sourceWork,
          bodyMeasurements: bodyData,
          referenceImages: imageUrls,
          options: {
            needReplaceFace: this.data.formData.needReplaceFace,
            needHeadwear: this.data.formData.needHeadwear,
            needAntiGravity: this.data.formData.needAntiGravity,
            needCornsilkPerm: this.data.formData.needCornsilkPerm,
            isUrgent: this.data.formData.isUrgent
          },
          remark: this.data.formData.remark
        }
      }) as any;
      
      wx.hideLoading();
      
      if (result && result.success) {
        wx.showModal({
          title: '提交成功',
          content: '您的订单已提交，请等待管理员审核。审核通过后，我们会通知您进行淘宝下单。',
          showCancel: false,
          success: () => {
            wx.switchTab({ url: '/pages/profile/profile' });
          }
        });
      } else {
        throw new Error(result?.error || '提交失败');
      }
    } catch (error: any) {
      wx.hideLoading();
      console.error('提交订单失败', error);
      Message.error({ context: this, offset: [20, 32], content: error.message || '提交失败，请重试' });
    } finally {
      this.setData({ isSubmitting: false });
    }
  }
});
