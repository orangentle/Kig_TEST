const app = getApp<IAppOption>();

interface OrderInfo {
  orderId: string;
  roleName: string;
  orderTime: string;
  status: 'processing' | 'completed';
}

interface OrderStats {
  processing: number;
  completed: number;
  total: number;
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
    orderStats: {
      processing: 0,
      completed: 0,
      total: 0
    } as OrderStats,
    isAdmin: false,  // 是否为管理员
    adminClickCount: 0,  // 点击用户ID的次数，用于触发管理员登录
    adminPassword: '123456'  // 管理员密码，实际应用中应该从服务器获取或更安全的方式存储
  },

  lifetimes: {
    attached() {
      // 检查登录状态
      const hasLogin = app.globalData.hasLogin;
      const userInfo = app.globalData.userInfo;
      
      this.setData({
        hasLogin,
        userInfo: userInfo || this.data.userInfo
      });
      
      if (hasLogin) {
        this.loadOrders();
        // 检查是否是管理员
        this.checkAdminStatus();
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
      
      // 计算订单统计数据
      const stats = {
        processing: mockData.filter(order => order.status === 'processing').length,
        completed: mockData.filter(order => order.status === 'completed').length,
        total: mockData.length
      };
      
      this.setData({
        orders: mockData,
        orderStats: stats
      });
    },
    
    // 点击订单项
    onOrderClick(e: any) {
      const orderId = e.currentTarget.dataset.orderId;
      
      wx.navigateTo({
        url: `/pages/order-detail/order-detail?id=${orderId}`
      });
    },
    
    // 查看全部订单
    viewAllOrders() {
      wx.showToast({
        title: '查看全部订单功能开发中',
        icon: 'none'
      });
    },
    
    // 联系客服
    contactService() {
      wx.showModal({
        title: '联系客服',
        content: '即将打开客服会话',
        success: (res) => {
          if (res.confirm) {
            wx.showToast({
              title: '客服功能开发中',
              icon: 'none'
            });
          }
        }
      });
    },

    // 检查管理员状态
    checkAdminStatus() {
      // 从存储中获取管理员状态
      const isAdmin = wx.getStorageSync('isAdmin') || false;
      this.setData({ isAdmin });
    },

    // 用户ID点击触发器 - 隐藏的管理员入口
    onAdminLoginTrigger() {
      let { adminClickCount } = this.data;
      adminClickCount++;
      
      this.setData({ adminClickCount });
      
      // 如果点击达到5次，弹出管理员登录对话框
      if (adminClickCount >= 5) {
        this.showAdminLoginDialog();
        // 重置点击计数
        this.setData({ adminClickCount: 0 });
      }
    },

    // 显示管理员登录对话框
    showAdminLoginDialog() {
      wx.showModal({
        title: '管理员登录',
        content: '请输入管理员密码',
        editable: true,
        placeholderText: '请输入密码',
        success: (res) => {
          if (res.confirm && res.content) {
            this.verifyAdminPassword(res.content);
          }
        }
      });
    },

    // 验证管理员密码
    verifyAdminPassword(password: string) {
      // 实际应用中应该使用更安全的验证方式，比如与服务器交互
      if (password === this.data.adminPassword) {
        // 设置为管理员
        this.setData({ isAdmin: true });
        wx.setStorageSync('isAdmin', true);
        
        wx.showToast({
          title: '管理员登录成功',
          icon: 'success'
        });
      } else {
        wx.showToast({
          title: '密码错误',
          icon: 'error'
        });
      }
    },

    // 进入管理后台
    enterAdminPanel() {
      wx.navigateTo({
        url: '/pages/admin/admin'
      });
    }
  }
}) 