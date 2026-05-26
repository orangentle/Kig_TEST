// index.ts
// 获取应用实例
const app = getApp<IAppOption>()
const defaultAvatarUrl = 'https://mmbiz.qpic.cn/mmbiz/icTdbqWNOwNRna42FI242Lcia07jQodd2FJGIYQfG0LAJGFxM4FbnQP6yfMxBgJ0F3YRqJCJ1aPAK2dQagdusBZg/0'

Component({
  data: {
    userInfo: {
      avatarUrl: defaultAvatarUrl,
      nickName: '',
    },
    hasUserInfo: false,
    logoUrl: '',
  },

  lifetimes: {
    attached() {
      // 确保云环境已初始化
      if (wx.cloud) {
        console.log('云环境已初始化');
      } else {
        console.error('云环境未初始化，请检查app.ts中的初始化代码');
      }
      
      // 设置图片路径
      this.setData({
        logoUrl: app.globalData.logoUrl,
      });
    }
  },

  methods: {
    // 跳转到下单页面
    goToOrder() {
      wx.navigateTo({
        url: '/pages/order/order'
      });
    },

    // 跳转到作品页面
    goToWorks() {
      wx.switchTab({
        url: '/pages/works/works'
      });
    },

    // 跳转到我的页面
    goToProfile() {
      wx.switchTab({
        url: '/pages/profile/profile'
      });
    },

    // 订单查询：通过淘宝订单号查询进度
    queryOrder() {
      const validOrderIds = ['TB123456789', 'TB987654321', 'TB456789123', 'TB789123456'];
      wx.showModal({
        title: '订单查询',
        editable: true,
        placeholderText: '请输入淘宝订单号 (如 TB123456789)',
        confirmText: '查询',
        confirmColor: '#ff8800',
        success: (res) => {
          if (!res.confirm) return;
          const tbOrderId = (res.content || '').trim();
          if (!tbOrderId) {
            wx.showToast({ title: '请输入订单号', icon: 'none' });
            return;
          }
          if (!validOrderIds.includes(tbOrderId)) {
            wx.showModal({
              title: '未找到订单',
              content: `未查询到订单号 ${tbOrderId}\n\n可使用以下测试订单号：\n${validOrderIds.join('\n')}`,
              showCancel: false,
              confirmText: '我知道了',
              confirmColor: '#ff8800'
            });
            return;
          }
          wx.navigateTo({
            url: `/pages/order-detail/order-detail?id=${tbOrderId}`
          });
        }
      });
    },

    // 跳转到偶壳娃聚页面
    goToGathering() {
      wx.navigateTo({
        url: '/pages/gathering/gathering'
      });
    },

    // 显示开发中提示
    showDeveloping() {
      wx.showToast({
        title: '功能开发中，敬请期待',
        icon: 'none',
        duration: 2000
      });
    }
  },
})
