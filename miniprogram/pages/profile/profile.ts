const app = getApp<IAppOption>();

interface OrderInfo {
  orderId: string;
  roleName: string;
  orderTime: string;
  status: 'processing' | 'completed';
}

Component({
  data: {
    hasLogin: false,
    userInfo: {
      avatarUrl: '',
      nickName: ''
    },
    userId: '123456',
    orders: [] as OrderInfo[],
    emptyImageUrl: ''
  },

  lifetimes: {
    attached() {
      // 检查登录状态
      const hasLogin = app.globalData.hasLogin;
      const userInfo = app.globalData.userInfo;
      
      this.setData({
        hasLogin,
        userInfo: userInfo || this.data.userInfo,
        emptyImageUrl: app.globalData.emptyImageUrl
      });
      
      if (hasLogin) {
        this.loadOrders();
      }
    }
  },

  methods: {
    // 获取用户信息
    onGetUserInfo(e: any) {
      if (e.detail.userInfo) {
        const userInfo = e.detail.userInfo;
        
        // 更新全局数据
        app.globalData.userInfo = userInfo;
        app.globalData.hasLogin = true;
        
        // 存储用户信息
        wx.setStorageSync('userInfo', userInfo);
        
        this.setData({
          hasLogin: true,
          userInfo
        });
        
        // 加载订单数据
        this.loadOrders();
        
        wx.showToast({
          title: '登录成功',
          icon: 'success'
        });
      } else {
        wx.showToast({
          title: '登录失败，请重试',
          icon: 'none'
        });
      }
    },
    
    // 加载订单数据
    loadOrders() {
      // 模拟数据，实际应从服务器获取
      const mockData: OrderInfo[] = [
        { 
          orderId: 'TB123456789', 
          roleName: '狐狸头壳', 
          orderTime: '2025-10-15', 
          status: 'processing' 
        },
        { 
          orderId: 'TB987654321', 
          roleName: '猫咪头壳', 
          orderTime: '2025-09-01', 
          status: 'completed' 
        },
        { 
          orderId: 'TB456789123', 
          roleName: '兔子头壳', 
          orderTime: '2025-08-15', 
          status: 'completed' 
        }
      ];
      
      this.setData({
        orders: mockData
      });
    },
    
    // 点击订单项
    onOrderClick(e: any) {
      const orderId = e.currentTarget.dataset.orderId;
      console.log('从个人中心点击订单项，订单ID:', orderId);
      
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
    
    // 联系客服
    contactService() {
      wx.showModal({
        title: '联系客服',
        content: '即将打开淘宝店铺客服页面',
        success: (res) => {
          if (res.confirm) {
            // 使用淘宝短链接
            const taobaoUrl = 'https://m.tb.cn/h.hf1womHplfjsH5V';
            wx.setStorageSync('webviewUrl', taobaoUrl);
            
            wx.navigateTo({
              url: '/pages/webview/webview',
              success: () => {
                console.log('成功打开淘宝网页');
              },
              fail: (err) => {
                console.error('打开淘宝网页失败', err);
                wx.showToast({
                  title: '打开失败，请稍后重试',
                  icon: 'none'
                });
              }
            });
          }
        }
      });
    }
  }
}) 