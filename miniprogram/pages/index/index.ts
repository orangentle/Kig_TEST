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
      // 确保云环境已初始化
      if (wx.cloud) {
        console.log('云环境已初始化');
      } else {
        console.error('云环境未初始化，请检查app.ts中的初始化代码');
      }
      
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
      // 从本地存储获取最近查询记录
      const recentSearches = wx.getStorageSync('recentSearches') || [];
      
      this.setData({
        recentSearches
      });
    },

    // 保存最近查询记录
    saveRecentSearch(orderInfo: OrderInfo) {
      // 获取现有记录
      let recentSearches = wx.getStorageSync('recentSearches') || [];
      
      // 检查是否已存在相同订单号
      const existingIndex = recentSearches.findIndex((item: OrderInfo) => item.orderId === orderInfo.orderId);
      
      if (existingIndex !== -1) {
        // 如果存在，删除旧记录
        recentSearches.splice(existingIndex, 1);
      }
      
      // 添加到记录开头
      recentSearches.unshift(orderInfo);
      
      // 限制记录数量为5条
      if (recentSearches.length > 5) {
        recentSearches = recentSearches.slice(0, 5);
      }
      
      // 保存到本地存储
      wx.setStorageSync('recentSearches', recentSearches);
      
      // 更新数据
      this.setData({
        recentSearches
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

      this.searchOrder(searchValue.trim());
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

      this.searchOrder(searchValue.trim());
    },

    // 搜索订单
    async searchOrder(orderId: string) {
      wx.showLoading({
        title: '查询中...'
      });

      try {
        // 使用云数据库查询订单
        const db = wx.cloud.database();
        const orderResult = await db.collection('orders').where({
          orderId: orderId
        }).get();
        
        // 隐藏加载提示
        wx.hideLoading();
        
        if (orderResult.data && orderResult.data.length > 0) {
          // 订单存在，保存到最近查询记录
          const orderInfo = {
            orderId: orderId,
            roleName: orderResult.data[0].roleName || '未知角色'
          };
          
          this.saveRecentSearch(orderInfo);
          
          // 导航到订单详情页
          wx.navigateTo({
            url: `/pages/order-detail/order-detail?id=${orderId}`,
            success: (res) => {
              console.log('导航成功');
            },
            fail: (err) => {
              console.error('导航失败', err);
              // 导航失败时显示提示
              wx.showToast({
                title: '页面跳转失败',
                icon: 'none'
              });
            }
          });
        } else {
          // 订单不存在，显示提示窗口
          wx.showModal({
            title: '未找到订单',
            content: `没有找到订单号为 ${orderId} 的订单信息，请确认订单号是否正确。`,
            showCancel: false,
            confirmText: '确定'
          });
          
          // 清空搜索框
          this.setData({
            searchValue: ''
          });
        }
      } catch (error) {
        console.error('查询订单失败', error);
        wx.hideLoading();
        
        // 查询失败，显示错误提示
        wx.showModal({
          title: '查询失败',
          content: '订单查询失败，请稍后重试。',
          showCancel: false,
          confirmText: '确定'
        });
      }
    },

    // 点击订单项
    onOrderClick(e: any) {
      const orderId = e.currentTarget.dataset.orderId;
      console.log('点击订单项，订单ID:', orderId);
      
      // 设置搜索框内容
      this.setData({
        searchValue: orderId
      });
      
      // 搜索订单
      this.searchOrder(orderId);
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
