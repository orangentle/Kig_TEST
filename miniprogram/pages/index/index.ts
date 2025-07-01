// index.ts
// 获取应用实例
const app = getApp<IAppOption>()
const defaultAvatarUrl = 'https://mmbiz.qpic.cn/mmbiz/icTdbqWNOwNRna42FI242Lcia07jQodd2FJGIYQfG0LAJGFxM4FbnQP6yfMxBgJ0F3YRqJCJ1aPAK2dQagdusBZg/0'

interface OrderInfo {
  orderId: string;
  roleName: string;
}

Component({
  data: {
    motto: 'Hello World',
    userInfo: {
      avatarUrl: defaultAvatarUrl,
      nickName: '',
    },
    hasUserInfo: false,
    canIUseGetUserProfile: wx.canIUse('getUserProfile'),
    canIUseNicknameComp: wx.canIUse('input.type.nickname'),
    searchValue: '',
    recentSearches: [] as OrderInfo[],
    logoUrl: '',
    emptyImageUrl: ''
  },

  lifetimes: {
    attached() {
      // 获取最近查询记录
      this.loadRecentSearches();
      
      // 设置图片路径
      this.setData({
        logoUrl: app.globalData.logoUrl,
        emptyImageUrl: app.globalData.emptyImageUrl
      });
    }
  },

  methods: {
    // 加载最近查询记录
    loadRecentSearches() {
      // 模拟数据，实际应从本地存储或服务器获取
      const mockData: OrderInfo[] = [
        { orderId: 'TB123456789', roleName: '狐狸头壳' },
        { orderId: 'TB987654321', roleName: '猫咪头壳' }
      ];
      
      this.setData({
        recentSearches: mockData
      });
    },

    // 搜索框内容变化
    onSearchChange(e: any) {
      this.setData({
        searchValue: e.detail.value
      });
    },

    // 提交搜索
    onSearch() {
      const { searchValue } = this.data;
      if (!searchValue.trim()) {
        wx.showToast({
          title: '请输入订单号',
          icon: 'none'
        });
        return;
      }

      this.searchOrder(searchValue);
    },

    // 点击搜索按钮
    onSearchButtonClick() {
      const { searchValue } = this.data;
      if (!searchValue.trim()) {
        wx.showToast({
          title: '请输入订单号',
          icon: 'none'
        });
        return;
      }

      this.searchOrder(searchValue);
    },

    // 搜索订单
    searchOrder(orderId: string) {
      wx.showLoading({
        title: '查询中...'
      });

      // 模拟API请求
      setTimeout(() => {
        wx.hideLoading();
        
        console.log('跳转到订单详情页，订单ID:', orderId);
        
        // 导航到订单详情页
        wx.navigateTo({
          url: `/pages/order-detail/order-detail?id=${orderId}`,
          success: (res) => {
            console.log('导航成功');
          },
          fail: (err) => {
            console.error('导航失败', err);
          }
        });
      }, 1000);
    },

    // 点击订单项
    onOrderClick(e: any) {
      const orderId = e.currentTarget.dataset.orderId;
      console.log('点击订单项，订单ID:', orderId);
      
      wx.navigateTo({
        url: `/pages/order-detail/order-detail?id=${orderId}`,
        success: (res) => {
          console.log('导航成功');
        },
        fail: (err) => {
          console.error('导航失败', err);
        }
      });
    },

    // 事件处理函数
    bindViewTap() {
      wx.navigateTo({
        url: '../logs/logs',
      })
    },
    onChooseAvatar(e: any) {
      const { avatarUrl } = e.detail
      const { nickName } = this.data.userInfo
      this.setData({
        "userInfo.avatarUrl": avatarUrl,
        hasUserInfo: nickName && avatarUrl && avatarUrl !== defaultAvatarUrl,
      })
    },
    onInputChange(e: any) {
      const nickName = e.detail.value
      const { avatarUrl } = this.data.userInfo
      this.setData({
        "userInfo.nickName": nickName,
        hasUserInfo: nickName && avatarUrl && avatarUrl !== defaultAvatarUrl,
      })
    },
    getUserProfile() {
      // 推荐使用wx.getUserProfile获取用户信息，开发者每次通过该接口获取用户个人信息均需用户确认，开发者妥善保管用户快速填写的头像昵称，避免重复弹窗
      wx.getUserProfile({
        desc: '展示用户信息', // 声明获取用户个人信息后的用途，后续会展示在弹窗中，请谨慎填写
        success: (res) => {
          console.log(res)
          this.setData({
            userInfo: res.userInfo,
            hasUserInfo: true
          })
        }
      })
    },
    // 查看全部搜索记录
    viewAllSearches() {
      wx.showToast({
        title: '查看全部记录功能开发中',
        icon: 'none'
      });
    }
  },
})
