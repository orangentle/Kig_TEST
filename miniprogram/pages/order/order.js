// order.js
var Message = require('tdesign-miniprogram/message/index');
var app = getApp();

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
    },
    useProfileBodyData: true,
    profileBodyData: null,
    referenceImages: [],
    isSubmitting: false,
    hasLogin: false
  },

  onLoad: function() {
    this.checkLoginAndLoadData();
  },

  checkLoginAndLoadData: function() {
    var that = this;
    wx.cloud.callFunction({
      name: 'login'
    }).then(function(res) {
      var result = res.result;
      if (result && result.openid) {
        that.setData({ hasLogin: true });
        
        var db = wx.cloud.database();
        db.collection('users').where({
          _openid: result.openid
        }).get().then(function(userResult) {
          if (userResult.data && userResult.data.length > 0) {
            var user = userResult.data[0];
            if (user.bodyMeasurements) {
              that.setData({
                profileBodyData: user.bodyMeasurements
              });
            }
          }
        });
      } else {
        that.setData({ hasLogin: false });
        wx.showModal({
          title: '请先登录',
          content: '您需要先登录才能下单',
          showCancel: false,
          success: function() {
            wx.switchTab({ url: '/pages/profile/profile' });
          }
        });
      }
    }).catch(function(error) {
      console.error('检查登录状态失败', error);
    });
  },

  onRoleNameChange: function(e) {
    this.setData({ 'formData.roleName': e.detail.value });
  },

  onSourceWorkChange: function(e) {
    this.setData({ 'formData.sourceWork': e.detail.value });
  },

  onUseProfileBodyDataChange: function(e) {
    this.setData({ useProfileBodyData: e.detail.value });
  },

  onHeightChange: function(e) {
    this.setData({ 'formData.height': parseFloat(e.detail.value) || 0 });
  },

  onWeightChange: function(e) {
    this.setData({ 'formData.weight': parseFloat(e.detail.value) || 0 });
  },

  onHeadCircumferenceChange: function(e) {
    this.setData({ 'formData.headCircumference': parseFloat(e.detail.value) || 0 });
  },

  onNeckCircumferenceChange: function(e) {
    this.setData({ 'formData.neckCircumference': parseFloat(e.detail.value) || 0 });
  },

  onShoulderWidthChange: function(e) {
    this.setData({ 'formData.shoulderWidth': parseFloat(e.detail.value) || 0 });
  },

  onReplaceFaceChange: function(e) {
    this.setData({ 'formData.needReplaceFace': e.detail.value });
  },

  onHeadwearChange: function(e) {
    this.setData({ 'formData.needHeadwear': e.detail.value });
  },

  onAntiGravityChange: function(e) {
    this.setData({ 'formData.needAntiGravity': e.detail.value });
  },

  onCornsilkPermChange: function(e) {
    this.setData({ 'formData.needCornsilkPerm': e.detail.value });
  },

  onUrgentChange: function(e) {
    var value = e.detail.value;
    if (value) {
      var that = this;
      wx.showModal({
        title: '确认加急',
        content: '加急服务将额外收取1000元费用，订单将优先制作。确认开启加急服务吗？',
        confirmText: '确认',
        cancelText: '取消',
        success: function(res) {
          if (res.confirm) {
            that.setData({ 'formData.isUrgent': true });
          }
        }
      });
    } else {
      this.setData({ 'formData.isUrgent': false });
    }
  },

  onRemarkChange: function(e) {
    this.setData({ 'formData.remark': e.detail.value });
  },

  onUploadAdd: function(e) {
    var files = e.detail.files;
    this.setData({
      referenceImages: this.data.referenceImages.concat(files)
    });
  },

  onUploadRemove: function(e) {
    var index = e.detail.index;
    var newImages = this.data.referenceImages.slice();
    newImages.splice(index, 1);
    this.setData({ referenceImages: newImages });
  },

  goToProfile: function() {
    wx.switchTab({ url: '/pages/profile/profile' });
  },

  validateForm: function() {
    var formData = this.data.formData;
    var useProfileBodyData = this.data.useProfileBodyData;
    var profileBodyData = this.data.profileBodyData;
    var referenceImages = this.data.referenceImages;
    
    if (!formData.roleName.trim()) {
      Message.error({ context: this, offset: [20, 32], content: '请输入角色名称' });
      return false;
    }
    
    if (!formData.sourceWork.trim()) {
      Message.error({ context: this, offset: [20, 32], content: '请输入来源作品' });
      return false;
    }
    
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

  uploadImages: function() {
    var that = this;
    var uploadedUrls = [];
    var promises = [];
    
    for (var i = 0; i < this.data.referenceImages.length; i++) {
      (function(index) {
        var file = that.data.referenceImages[index];
        var filePath = file.url || file.path;
        
        if (filePath.startsWith('cloud://')) {
          uploadedUrls[index] = filePath;
        } else {
          var timestamp = Date.now();
          var cloudPath = 'orders/reference/' + timestamp + '_' + index + '.' + (filePath.split('.').pop() || 'jpg');
          
          promises.push(
            wx.cloud.uploadFile({
              cloudPath: cloudPath,
              filePath: filePath
            }).then(function(uploadResult) {
              uploadedUrls[index] = uploadResult.fileID;
            })
          );
        }
      })(i);
    }
    
    return Promise.all(promises).then(function() {
      return uploadedUrls.filter(function(url) { return url; });
    });
  },

  onSubmit: function() {
    var that = this;
    
    // 防止重复提交
    if (this.data.isSubmitting) {
      return;
    }
    
    if (!this.data.hasLogin) {
      wx.showModal({
        title: '请先登录',
        content: '您需要先登录才能下单',
        showCancel: false,
        success: function() {
          wx.switchTab({ url: '/pages/profile/profile' });
        }
      });
      return;
    }
    
    if (!this.validateForm()) {
      return;
    }
    
    this.setData({ isSubmitting: true });
    
    wx.showLoading({ title: '提交中...' });
    
    this.uploadImages().then(function(imageUrls) {
      var bodyData;
      if (that.data.useProfileBodyData && that.data.profileBodyData) {
        bodyData = that.data.profileBodyData;
      } else {
        bodyData = {
          height: that.data.formData.height,
          weight: that.data.formData.weight,
          headCircumference: that.data.formData.headCircumference,
          neckCircumference: that.data.formData.neckCircumference,
          shoulderWidth: that.data.formData.shoulderWidth
        };
      }
      
      return wx.cloud.callFunction({
        name: 'submitOrder',
        data: {
          roleName: that.data.formData.roleName,
          sourceWork: that.data.formData.sourceWork,
          bodyMeasurements: bodyData,
          referenceImages: imageUrls,
          options: {
            needReplaceFace: that.data.formData.needReplaceFace,
            needHeadwear: that.data.formData.needHeadwear,
            needAntiGravity: that.data.formData.needAntiGravity,
            needCornsilkPerm: that.data.formData.needCornsilkPerm,
            isUrgent: that.data.formData.isUrgent
          },
          remark: that.data.formData.remark
        }
      });
    }).then(function(res) {
      wx.hideLoading();
      var result = res.result;
      
      if (result && result.success) {
        wx.showModal({
          title: '提交成功',
          content: '您的订单已提交，请等待管理员审核。审核通过后，我们会通知您进行淘宝下单。',
          showCancel: false,
          success: function() {
            wx.switchTab({ url: '/pages/profile/profile' });
          }
        });
      } else {
        throw new Error(result && result.error || '提交失败');
      }
    }).catch(function(error) {
      wx.hideLoading();
      console.error('提交订单失败', error);
      Message.error({ context: that, offset: [20, 32], content: error.message || '提交失败，请重试' });
    }).finally(function() {
      that.setData({ isSubmitting: false });
    });
  }
});
