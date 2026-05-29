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

    // 订单查询弹窗
    showQueryPopup: false,
    queryValue: '',
    queryError: '',
    isQuerying: false,
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
      this.setData({
        showQueryPopup: true,
        queryValue: '',
        queryError: ''
      });
    },

    onQueryPopupClose() {
      this.setData({ showQueryPopup: false });
    },

    onQueryInput(e: any) {
      this.setData({
        queryValue: e.detail.value,
        queryError: ''
      });
    },

    onQueryClear() {
      this.setData({ queryValue: '', queryError: '' });
    },

    onQuerySubmit() {
      const tbOrderId = (this.data.queryValue || '').trim();
      if (!tbOrderId) {
        this.setData({ queryError: '订单号不能空着哦~' });
        return;
      }
      if (this.data.isQuerying) return;
      this.setData({ isQuerying: true, queryError: '' });
      wx.cloud.callFunction({
        name: 'getOrders',
        data: { tbOrderId },
        success: (res: any) => {
          const list = (res.result && res.result.data) || [];
          const found = list.find((o: any) => o.tbOrderId === tbOrderId);
          if (!found) {
            this.setData({
              isQuerying: false,
              queryError: `咦,没找到订单 ${tbOrderId}`
            });
            return;
          }
          this.setData({ isQuerying: false, showQueryPopup: false });
          wx.navigateTo({
            url: `/pages/order-detail/order-detail?id=${tbOrderId}`
          });
        },
        fail: () => {
          this.setData({ isQuerying: false, queryError: '信号迷路了,再试一次?' });
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
        title: '这个被你看到啦~ 还在赶工中',
        icon: 'none',
        duration: 2000
      });
    }
  },
})
