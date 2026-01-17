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
