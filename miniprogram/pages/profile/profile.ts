// 获取应用实例
import type { OrderStatus } from '../../types/order';
import type { UserProfile } from '../../types/user';
import { validateName, validateContactAndBody, toastIfFail } from '../../utils/validator';

const globalApp = getApp<IAppOption>();

interface OrderInfo {
  orderId: string;
  tbOrderId: string;
  roleName: string;
  orderTime: string;
  status: OrderStatus;
}

interface OrderStats {
  pending: number;
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
    userId: '',
    taobaoName: '',
    orders: [] as OrderInfo[],
    orderStats: {
      pending: 0,
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
      nickName: '',
      taobaoName: '',
      qq: '',
      phone: ''
    }, // 临时存储编辑中的用户信息
    tempBodyMeasurements: {
      height: 0,
      weight: 0,
      headCircumference: 0,
      shoulderWidth: 0
    }, // 临时存储编辑中的身材数据
    tempAvatarPath: '', // 临时存储选择的头像路径
    loginForm: {
      avatarUrl: '',     // chooseAvatar 返回的临时路径
      nickName: ''       // nickname 输入框的值
    }
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

  pageLifetimes: {
    // 页面显示时重新加载订单数据
    show() {
      if (this.data.hasLogin) {
        this.loadOrders();
        (this as any).checkRejectNotices();
      }
      if (wx.getStorageSync('openBodyForm')) {
        wx.removeStorageSync('openBodyForm');
        if (this.data.hasLogin) {
          setTimeout(() => { (this as any).showEditProfileModal(); }, 200);
        } else {
          wx.showToast({ title: '请先登录', icon: 'none' });
        }
      }
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
              taobaoName: userProfile.taobaoName || '',
              isAdmin: userProfile.isAdmin || false
            });
            
            // 加载订单数据
            this.loadOrders();
          } else {
            this.setData({ hasLogin: false });
          }
        }
      } catch (error) {
        console.error('检查登录状态失败', error);
      } finally {
        this.setData({ isLoading: false });
      }
    },
    
    // 获取用户信息并注册（已废弃：微信 2022/10 后 getUserInfo 不再返回真实信息，改用 onLoginSubmit）
    async onGetUserInfo(_e: any) {
      this.onLoginSubmit();
    },
    
    // 登录：选择头像（微信原生能力，返回临时文件路径）
    onLoginChooseAvatar(e: any) {
      const { avatarUrl } = e.detail || {};
      if (avatarUrl) {
        this.setData({ 'loginForm.avatarUrl': avatarUrl });
      }
    },

    // 登录：昵称输入（type="nickname" 获焦时微信会展示用户昵称建议）
    onLoginNicknameInput(e: any) {
      this.setData({ 'loginForm.nickName': (e.detail.value || '').trim() });
    },

    // 登录：提交
    async onLoginSubmit() {
      const { avatarUrl, nickName } = this.data.loginForm;
      if (!avatarUrl) {
        wx.showToast({ title: '头像选一个吧~', icon: 'none' });
        return;
      }
      if (!nickName) {
        wx.showToast({ title: '昵称还没填哦~', icon: 'none' });
        return;
      }

      this.setData({ isLoading: true });
      wx.showLoading({ title: '正在带你进窝~' });

      try {
        const { result } = await wx.cloud.callFunction({ name: 'login' }) as any;
        if (!result || !result.openid) throw new Error('获取 openid 失败');
        const openid = result.openid;
        const userId = openid.slice(-8);

        const ext = avatarUrl.match(/\.([^.?]+)(\?|$)/)?.[1] || 'png';
        const cloudPath = `images/avatars/${userId}_${Date.now()}.${ext}`;
        const up = await wx.cloud.uploadFile({ cloudPath, filePath: avatarUrl });
        const cloudAvatar = up.fileID;

        const db = wx.cloud.database();
        const userCheck = await db.collection('users').where({ _openid: openid }).get();

        let isAdmin = false;
        if (userCheck.data.length === 0) {
          await db.collection('users').add({
            data: {
              avatarUrl: cloudAvatar,
              nickName,
              userId,
              createTime: Date.now()
            }
          });
        } else {
          const doc = userCheck.data[0] as UserProfile;
          isAdmin = doc.isAdmin || false;
          await db.collection('users').doc(doc._id as string).update({
            data: {
              avatarUrl: cloudAvatar,
              nickName,
              updateTime: Date.now()
            }
          });
        }

        const userInfo = { avatarUrl: cloudAvatar, nickName };
        globalApp.globalData.userInfo = {
          ...userInfo,
          city: '', country: '', gender: 0, language: 'zh_CN', province: ''
        } as any;
        globalApp.globalData.hasLogin = true;
        wx.setStorageSync('userInfo', userInfo);

        this.setData({
          hasLogin: true,
          userInfo,
          userId,
          isAdmin,
          loginForm: { avatarUrl: '', nickName: '' }
        });

        this.loadOrders();
        wx.showToast({ title: '欢迎回来呀~ ✨', icon: 'success' });
      } catch (err) {
        console.error('登录失败', err);
        wx.showToast({ title: '登录出小差啦,再试试?', icon: 'none' });
      } finally {
        wx.hideLoading();
        this.setData({ isLoading: false });
      }
    },

    // 检查并展示订单驳回通知(从云端 users.pendingRejectNotices 拉取)
    async checkRejectNotices() {
      try {
        const db = wx.cloud.database();
        const _ = db.command;
        const { result: loginRes } = await wx.cloud.callFunction({ name: 'login' }) as any;
        const openid = loginRes && loginRes.openid;
        if (!openid) return;
        const r = await db.collection('users')
          .where({ _openid: openid })
          .field({ pendingRejectNotices: true })
          .limit(1)
          .get();
        const notices = (r.data[0] && (r.data[0] as any).pendingRejectNotices) || [];
        if (!notices.length) return;

        // 拼接展示内容
        const lines = notices.slice(0, 5).map((n: any, i: number) =>
          `${i + 1}. 「${n.roleName || '订单'}」(单号 ${n.tbOrderId || '-'})\n   原因: ${n.reason || '信息有误'}`
        );
        const extra = notices.length > 5 ? `\n\n…还有 ${notices.length - 5} 条` : '';
        const content = `鼠鼠帮你看了下,有 ${notices.length} 单需要重新填一下喔~\n\n${lines.join('\n')}${extra}`;

        wx.showModal({
          title: '订单需要重新提交 ♡',
          content,
          confirmText: '去重新下单',
          cancelText: '知道啦',
          success: (res) => {
            // 无论选哪个都清掉通知,避免反复弹
            db.collection('users')
              .where({ _openid: openid })
              .update({ data: { pendingRejectNotices: _.set([]) } })
              .catch((e: any) => console.warn('清理驳回通知失败', e));
            if (res.confirm) {
              wx.switchTab({ url: '/pages/order/order' });
            }
          }
        });
      } catch (err) {
        console.warn('checkRejectNotices 失败', err);
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

        // 云数据库单次查询最多 20 条，循环分页取全部订单
        const PAGE_SIZE = 20;
        const allData: any[] = [];
        let skip = 0;
        while (true) {
          const page = await db.collection('orders')
            .where({ _openid: openid })
            .orderBy('createTime', 'desc')
            .skip(skip)
            .limit(PAGE_SIZE)
            .get();
          allData.push(...page.data);
          if (page.data.length < PAGE_SIZE) break;
          skip += PAGE_SIZE;
        }
        const orderResult = { data: allData };

        if (orderResult.data && orderResult.data.length > 0) {
          // 转换数据格式
          const orders = orderResult.data.map((order: any) => {
            return {
              orderId: order.orderId,
              tbOrderId: order.tbOrderId || '',
              roleName: order.roleName,
              orderTime: this.formatDate(order.createTime),
              status: order.status
            } as OrderInfo;
          });
          
          // 计算订单统计数据
          const stats = {
            pending: orders.filter(order => order.status === 'pending').length,
            processing: orders.filter(order => order.status === 'normal' || order.status === 'urgent').length,
            completed: orders.filter(order => order.status === 'completed').length,
            total: orders.filter(order => order.status !== 'canceled').length
          };
          
          this.setData({
            orders,
            orderStats: stats
          });
        } else {
          // 如果没有订单数据，显示空状态
          this.setData({
            orders: [],
            orderStats: { pending: 0, processing: 0, completed: 0, total: 0 }
          });
        }
      } catch (error) {
        console.error('加载订单失败', error);
        // 加载失败时显示空状态
        this.setData({
          orders: [],
          orderStats: { pending: 0, processing: 0, completed: 0, total: 0 }
        });
      } finally {
        this.setData({ isLoading: false });
      }
    },
    
    // 格式化日期
    formatDate(timestamp: number | string | Date | undefined | null) {
      if (!timestamp) return '';
      const date = new Date(timestamp as any);
      if (isNaN(date.getTime())) return '';
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
    },
    
    // 点击订单项
    onOrderClick(e: any) {
      const tbOrderId = e.currentTarget.dataset.tbOrderId;
      if (!tbOrderId) {
        wx.showToast({ title: '订单缺少淘宝单号', icon: 'none' });
        return;
      }
      wx.navigateTo({
        url: `/pages/order-detail/order-detail?id=${tbOrderId}`
      });
    },

    // 打开工期计算器
    onOpenWorkDayCalc() {
      wx.navigateTo({ url: '/pages/work-day-calc/work-day-calc' });
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

    // 隐藏的管理员入口：长按头像触发密码弹窗（无可见提示）
    onAdminLongPress() {
      this.showAdminLoginDialog();
    },

    // 用户ID点击触发器 - 隐藏的管理员入口（保留兼容旧逻辑，未在页面绑定）
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
      // 检查是否为管理员
      if (!this.data.isAdmin) {
        wx.showModal({
          title: '权限不足',
          content: '只有管理员才能进入管理后台',
          showCancel: false,
          confirmText: '我知道了'
        });
        return;
      }
      
      wx.navigateTo({
        url: '/pages/admin/admin'
      });
    },

    // 显示编辑资料弹窗
    async showEditProfileModal() {
      try {
        // 从数据库获取完整的用户信息
        const { result } = await wx.cloud.callFunction({
          name: 'getOpenId'
        }) as any;
        
        const db = wx.cloud.database();
        const userResult = await db.collection('users').where({
          _openid: result.openid
        }).get();
        
        let userProfile: UserProfile | null = null;
        if (userResult.data && userResult.data.length > 0) {
          userProfile = userResult.data[0] as UserProfile;
        }
        
        // 设置临时用户信息
        this.setData({
          showEditPopup: true,
          tempUserInfo: { 
            avatarUrl: this.data.userInfo.avatarUrl,
            nickName: this.data.userInfo.nickName,
            taobaoName: userProfile?.taobaoName || '',
            qq: userProfile?.qq || '',
            phone: userProfile?.phone || ''
          },
          tempBodyMeasurements: {
            height: userProfile?.bodyMeasurements?.height || 0,
            weight: userProfile?.bodyMeasurements?.weight || 0,
            headCircumference: userProfile?.bodyMeasurements?.headCircumference || 0,
            shoulderWidth: userProfile?.bodyMeasurements?.shoulderWidth || 0
          },
          tempAvatarPath: ''
        });
      } catch (error) {
        console.error('加载用户信息失败', error);
        // 如果加载失败，使用默认值
        this.setData({
          showEditPopup: true,
          tempUserInfo: { 
            avatarUrl: this.data.userInfo.avatarUrl,
            nickName: this.data.userInfo.nickName,
            taobaoName: '',
            qq: '',
            phone: ''
          },
          tempBodyMeasurements: {
            height: 0,
            weight: 0,
            headCircumference: 0,
            shoulderWidth: 0
          },
          tempAvatarPath: ''
        });
      }
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

    // 淘宝名称输入变更
    onTaobaoNameChange(e: any) {
      this.setData({
        'tempUserInfo.taobaoName': e.detail.value
      });
    },

    // QQ账号输入变更
    onQQChange(e: any) {
      this.setData({
        'tempUserInfo.qq': e.detail.value
      });
    },

    // 手机号输入变更
    onPhoneChange(e: any) {
      this.setData({
        'tempUserInfo.phone': e.detail.value
      });
    },

    // 身高输入变更
    onHeightChange(e: any) {
      this.setData({
        'tempBodyMeasurements.height': parseFloat(e.detail.value) || 0
      });
    },

    // 体重输入变更
    onWeightChange(e: any) {
      this.setData({
        'tempBodyMeasurements.weight': parseFloat(e.detail.value) || 0
      });
    },

    // 头围输入变更
    onHeadCircumferenceChange(e: any) {
      this.setData({
        'tempBodyMeasurements.headCircumference': parseFloat(e.detail.value) || 0
      });
    },

    // 肩宽输入变更
    onShoulderWidthChange(e: any) {
      this.setData({
        'tempBodyMeasurements.shoulderWidth': parseFloat(e.detail.value) || 0
      });
    },

    // 保存用户资料
    async saveUserProfile() {
      try {
        // 鼠鼠校验：昵称 + 联系方式 + 身材数据
        const nick = this.data.tempUserInfo.nickName;
        const nickRes = validateName(nick, '昵称', true, 20);
        if (!toastIfFail(nickRes)) return;

        const contactRes = validateContactAndBody({
          qq: this.data.tempUserInfo.qq,
          phone: this.data.tempUserInfo.phone,
          taobaoName: this.data.tempUserInfo.taobaoName,
          height: this.data.tempBodyMeasurements.height,
          weight: this.data.tempBodyMeasurements.weight,
          headCircumference: this.data.tempBodyMeasurements.headCircumference,
          shoulderWidth: this.data.tempBodyMeasurements.shoulderWidth,
        });
        if (!toastIfFail(contactRes)) return;

        this.setData({ isLoading: true });
        
        // 如果有新头像，先上传头像
        if (this.data.tempAvatarPath) {
          await this.uploadAvatarAndSaveProfile();
        } else {
          // 直接保存资料
          await this.updateUserProfile({
            nickName: this.data.tempUserInfo.nickName,
            avatarUrl: this.data.userInfo.avatarUrl,
            taobaoName: this.data.tempUserInfo.taobaoName,
            qq: this.data.tempUserInfo.qq,
            phone: this.data.tempUserInfo.phone
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
            avatarUrl: uploadResult.fileID,
            taobaoName: this.data.tempUserInfo?.taobaoName,
            qq: this.data.tempUserInfo?.qq,
            phone: this.data.tempUserInfo?.phone
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
    async updateUserProfile(userInfo: { nickName: string, avatarUrl: string, taobaoName?: string, qq?: string, phone?: string }) {
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
          
          // 准备更新数据
          const updateData: any = {
            avatarUrl: userInfo.avatarUrl,
            nickName: userInfo.nickName,
            taobaoName: userInfo.taobaoName || '',
            qq: userInfo.qq || '',
            phone: userInfo.phone || '',
            bodyMeasurements: this.data.tempBodyMeasurements,
            updateTime: Date.now()
          };
          
          await db.collection('users').doc(docId).update({
            data: updateData
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
            userInfo: updatedUserInfo,
            taobaoName: userInfo.taobaoName || ''
          });
          
          wx.showToast({
            title: '资料保存好啦~ ✨',
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