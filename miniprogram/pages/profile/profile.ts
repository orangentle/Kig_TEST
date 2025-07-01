// 获取应用实例
const globalApp = getApp<IAppOption>();

// 用户信息接口
interface UserInfo {
  avatarUrl: string;
  nickName: string;
  city?: string;
  country?: string;
  gender?: number;
  language?: string;
  province?: string;
}

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

// 用户资料接口
interface UserProfile {
  _id?: string;
  _openid?: string;
  avatarUrl: string;
  nickName: string;
  userId: string;
  phone?: string;
  email?: string;
  isAdmin?: boolean;
  createTime?: number;
}

Component({
  data: {
    hasLogin: false,
    userInfo: {
      avatarUrl: '',
      nickName: ''
    },
    userId: '',
    orders: [] as OrderInfo[],
    orderStats: {
      processing: 0,
      completed: 0,
      total: 0
    } as OrderStats,
    isAdmin: false,  // 是否为管理员
    adminClickCount: 0,  // 点击用户ID的次数，用于触发管理员登录
    adminPassword: '123456',  // 管理员密码，实际应用中应该从服务器获取或更安全的方式存储
    isLoading: false, // 是否正在加载
    showEditPopup: false, // 是否显示编辑个人资料弹窗
    tempUserInfo: {
      avatarUrl: '',
      nickName: ''
    }, // 临时存储编辑中的用户信息
    tempAvatarPath: '' // 临时存储选择的头像路径
  },

  lifetimes: {
    attached() {
      // 初始化云开发
      if (!wx.cloud) {
        console.error('请使用 2.2.3 或以上的基础库以使用云能力');
      } else {
        wx.cloud.init({
          env: this.data.cloudEnv,
          traceUser: true
        });
      }
      
      // 检查登录状态
      this.checkLoginStatus();
    }
  },

  methods: {
    // 检查登录状态
    async checkLoginStatus() {
      try {
        this.setData({ isLoading: true });
        
        // 获取云开发用户信息
        const { result } = await wx.cloud.callFunction({
          name: 'login',
        }) as any;
        
        // 如果已经授权，获取用户信息
        if (result && result.openid) {
          // 查询数据库中的用户信息
          const db = wx.cloud.database();
          const userResult = await db.collection('users').where({
            _openid: result.openid
          }).get();
          
          if (userResult.data && userResult.data.length > 0) {
            const userProfile = userResult.data[0] as UserProfile;
            
            // 设置全局数据
            globalApp.globalData.userInfo = {
              avatarUrl: userProfile.avatarUrl,
              nickName: userProfile.nickName,
              city: '',
              country: '',
              gender: 0,
              language: 'zh_CN',
              province: ''
            };
            globalApp.globalData.hasLogin = true;
            
            // 设置页面数据
            this.setData({
              hasLogin: true,
              userInfo: {
                avatarUrl: userProfile.avatarUrl,
                nickName: userProfile.nickName
              },
              userId: userProfile.userId || result.openid.slice(-8),
              isAdmin: userProfile.isAdmin || false
            });
            
            // 加载订单数据
            this.loadOrders();
          } else {
            console.log('用户未注册，等待用户授权');
            this.setData({ hasLogin: false });
          }
        }
      } catch (error) {
        console.error('检查登录状态失败', error);
      } finally {
        this.setData({ isLoading: false });
      }
    },
    
    // 获取用户信息并注册
    async onGetUserInfo(e: any) {
      if (e.detail.userInfo) {
        try {
          this.setData({ isLoading: true });
          
          const userInfo = e.detail.userInfo;
          
          // 调用云函数登录
          const { result } = await wx.cloud.callFunction({
            name: 'login',
          }) as any;
          
          if (result && result.openid) {
            // 生成用户ID
            const userId = result.openid.slice(-8);
            
            // 创建用户资料
            const userProfile: UserProfile = {
              avatarUrl: userInfo.avatarUrl,
              nickName: userInfo.nickName,
              userId: userId,
              createTime: Date.now()
            };
            
            // 存储到云数据库
            const db = wx.cloud.database();
            
            // 检查用户是否已存在
            const userCheck = await db.collection('users').where({
              _openid: result.openid
            }).get();
            
            if (userCheck.data.length === 0) {
              // 新用户，添加到数据库
              await db.collection('users').add({
                data: userProfile
              });
            } else {
              // 更新用户信息
              const docId = userCheck.data[0]._id as string;
              await db.collection('users').doc(docId).update({
                data: {
                  avatarUrl: userInfo.avatarUrl,
                  nickName: userInfo.nickName,
                  updateTime: Date.now()
                }
              });
            }
            
            // 更新全局数据
            globalApp.globalData.userInfo = userInfo;
            globalApp.globalData.hasLogin = true;
            
            // 存储用户信息
            wx.setStorageSync('userInfo', userInfo);
            
            this.setData({
              hasLogin: true,
              userInfo,
              userId
            });
            
            // 加载订单数据
            this.loadOrders();
            
            wx.showToast({
              title: '登录成功',
              icon: 'success'
            });
          }
        } catch (error) {
          console.error('登录失败', error);
          wx.showToast({
            title: '登录失败，请重试',
            icon: 'none'
          });
        } finally {
          this.setData({ isLoading: false });
        }
      } else {
        wx.showToast({
          title: '登录失败，请重试',
          icon: 'none'
        });
      }
    },
    
    // 加载订单数据
    async loadOrders() {
      try {
        this.setData({ isLoading: true });
        
        // 从云数据库加载订单
        const db = wx.cloud.database();
        const wxContext = await wx.cloud.callFunction({
          name: 'getOpenId'
        }) as any;
        
        const openid = wxContext.result.openid;
        
        const orderResult = await db.collection('orders')
          .where({
            _openid: openid
          })
          .orderBy('createTime', 'desc')
          .limit(5)
          .get();
        
        if (orderResult.data && orderResult.data.length > 0) {
          // 转换数据格式
          const orders = orderResult.data.map((order: any) => {
            return {
              orderId: order.orderId,
              roleName: order.roleName,
              orderTime: this.formatDate(order.createTime),
              status: order.status
            } as OrderInfo;
          });
          
          // 计算订单统计数据
          const stats = {
            processing: orders.filter(order => order.status === 'processing').length,
            completed: orders.filter(order => order.status === 'completed').length,
            total: orders.length
          };
          
          this.setData({
            orders,
            orderStats: stats
          });
        } else {
          // 如果没有订单数据，使用模拟数据（实际应用中可以显示空状态）
          this.loadMockOrders();
        }
      } catch (error) {
        console.error('加载订单失败', error);
        // 加载失败时使用模拟数据
        this.loadMockOrders();
      } finally {
        this.setData({ isLoading: false });
      }
    },
    
    // 加载模拟订单数据（用于开发测试）
    loadMockOrders() {
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
    
    // 格式化日期
    formatDate(timestamp: number) {
      const date = new Date(timestamp);
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
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
    async checkAdminStatus() {
      try {
        // 从云数据库获取管理员状态
        const db = wx.cloud.database();
        const wxContext = await wx.cloud.callFunction({
          name: 'getOpenId'
        }) as any;
        
        const openid = wxContext.result.openid;
        
        const userResult = await db.collection('users').where({
          _openid: openid
        }).get();
        
        if (userResult.data && userResult.data.length > 0) {
          const isAdmin = userResult.data[0].isAdmin || false;
          this.setData({ isAdmin });
        }
      } catch (error) {
        console.error('检查管理员状态失败', error);
        // 如果获取失败，尝试从本地存储获取
        const isAdmin = wx.getStorageSync('isAdmin') || false;
        this.setData({ isAdmin });
      }
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
    async verifyAdminPassword(password: string) {
      try {
        // 调用云函数验证管理员密码
        const { result } = await wx.cloud.callFunction({
          name: 'adminAuth',
          data: {
            password
          }
        }) as any;
        
        if (result && result.success) {
          // 设置为管理员
          this.setData({ isAdmin: true });
          
          // 更新云数据库中的用户信息
          const db = wx.cloud.database();
          const wxContext = await wx.cloud.callFunction({
            name: 'getOpenId'
          }) as any;
          
          const openid = wxContext.result.openid;
          
          const userResult = await db.collection('users').where({
            _openid: openid
          }).get();
          
          if (userResult.data && userResult.data.length > 0) {
            const docId = userResult.data[0]._id as string;
            await db.collection('users').doc(docId).update({
              data: {
                isAdmin: true
              }
            });
          }
          
          // 本地存储作为备份
          wx.setStorageSync('isAdmin', true);
          
          wx.showToast({
            title: '管理员登录成功',
            icon: 'success'
          });
        } else {
          // 如果云函数不可用，回退到本地验证
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
        }
      } catch (error) {
        console.error('验证管理员密码失败', error);
        // 如果云函数调用失败，回退到本地验证
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
      }
    },

    // 进入管理后台
    enterAdminPanel() {
      wx.navigateTo({
        url: '/pages/admin/admin'
      });
    },

    // 显示编辑资料弹窗
    showEditProfileModal() {
      console.log('编辑资料按钮被点击');
      // 设置临时用户信息
      this.setData({
        showEditPopup: true,
        tempUserInfo: { 
          avatarUrl: this.data.userInfo.avatarUrl,
          nickName: this.data.userInfo.nickName
        },
        tempAvatarPath: ''
      });
      console.log('弹窗状态设置完成', this.data.showEditPopup, this.data.tempUserInfo);
    },

    // 关闭编辑资料弹窗
    onCloseEditPopup() {
      this.setData({ showEditPopup: false });
    },

    // 选择头像
    chooseAvatar() {
      if (!this.data.hasLogin) return;

      wx.chooseImage({
        count: 1,
        sizeType: ['compressed'],
        sourceType: ['album', 'camera'],
        success: (res) => {
          const tempFilePaths = res.tempFilePaths;
          
          this.setData({
            tempAvatarPath: tempFilePaths[0],
            'tempUserInfo.avatarUrl': tempFilePaths[0]
          });

          // 如果是直接点击头像（不是在编辑资料弹窗中），则直接上传头像并保存
          if (!this.data.showEditPopup) {
            this.uploadAvatarAndSaveProfile();
          }
        }
      });
    },

    // 昵称输入变更
    onNicknameChange(e: any) {
      this.setData({
        'tempUserInfo.nickName': e.detail.value
      });
    },

    // 保存用户资料
    async saveUserProfile() {
      try {
        this.setData({ isLoading: true });
        
        // 如果有新头像，先上传头像
        if (this.data.tempAvatarPath) {
          await this.uploadAvatarAndSaveProfile();
        } else {
          // 直接保存资料
          await this.updateUserProfile({
            nickName: this.data.tempUserInfo.nickName,
            avatarUrl: this.data.userInfo.avatarUrl
          });
        }
        
        this.setData({ showEditPopup: false });

      } catch (error) {
        console.error('保存用户资料失败', error);
        wx.showToast({
          title: '保存资料失败，请重试',
          icon: 'none'
        });
      } finally {
        this.setData({ isLoading: false });
      }
    },

    // 上传头像并保存资料
    async uploadAvatarAndSaveProfile() {
      if (!this.data.tempAvatarPath) return;

      try {
        this.setData({ isLoading: true });
        wx.showLoading({ title: '正在上传头像...' });
        
        // 获取用户openid
        const { result } = await wx.cloud.callFunction({
          name: 'getOpenId'
        }) as any;
        
        const openid = result.openid;
        
        // 生成文件名：使用用户ID + 时间戳
        const timestamp = Date.now();
        const fileExtension = this.data.tempAvatarPath.match(/\.([^.]+)$/)?.[1] || 'png';
        const cloudPath = `images/avatars/${this.data.userId}_${timestamp}.${fileExtension}`;
        
        // 上传图片到云存储
        const uploadResult = await wx.cloud.uploadFile({
          cloudPath,
          filePath: this.data.tempAvatarPath
        });
        
        if (uploadResult.fileID) {
          // 更新用户资料
          await this.updateUserProfile({
            nickName: this.data.tempUserInfo?.nickName || this.data.userInfo.nickName,
            avatarUrl: uploadResult.fileID
          });
        } else {
          throw new Error('头像上传失败');
        }
      } catch (error) {
        console.error('上传头像失败', error);
        wx.showToast({
          title: '头像上传失败，请重试',
          icon: 'none'
        });
      } finally {
        wx.hideLoading();
        this.setData({ isLoading: false, tempAvatarPath: '' });
      }
    },

    // 更新用户资料到数据库
    async updateUserProfile(userInfo: { nickName: string, avatarUrl: string }) {
      try {
        // 获取用户openid
        const wxContext = await wx.cloud.callFunction({
          name: 'getOpenId'
        }) as any;
        
        const openid = wxContext.result.openid;
        
        // 更新云数据库
        const db = wx.cloud.database();
        const userResult = await db.collection('users').where({
          _openid: openid
        }).get();
        
        if (userResult.data && userResult.data.length > 0) {
          const docId = userResult.data[0]._id as string;
          
          await db.collection('users').doc(docId).update({
            data: {
              avatarUrl: userInfo.avatarUrl,
              nickName: userInfo.nickName,
              updateTime: Date.now()
            }
          });
          
          // 更新本地和全局数据
          const updatedUserInfo = {
            ...this.data.userInfo,
            avatarUrl: userInfo.avatarUrl,
            nickName: userInfo.nickName
          };
          
          globalApp.globalData.userInfo = {
            ...globalApp.globalData.userInfo as WechatMiniprogram.UserInfo,
            avatarUrl: userInfo.avatarUrl,
            nickName: userInfo.nickName
          };
          
          wx.setStorageSync('userInfo', updatedUserInfo);
          
          this.setData({
            userInfo: updatedUserInfo
          });
          
          wx.showToast({
            title: '资料更新成功',
            icon: 'success'
          });
        }
      } catch (error) {
        console.error('更新用户资料失败', error);
        throw error;
      }
    }
  }
}) 