// order-detail.ts
// 获取应用实例
const appInstance = getApp<IAppOption>();

interface OrderDetail {
  orderId: string;
  queueId: string;
  roleName: string;
  expectedCompletionDate: string;
  currentStepIndex: number;
  previewImage?: string;
}

interface Step {
  title: string;
  content: string;
}

Page({
  data: {
    orderId: '',
    order: {
      orderId: '',
      queueId: '',
      roleName: '',
      expectedCompletionDate: '',
      currentStepIndex: 3,
      previewImage: ''
    } as OrderDetail,
    currentStepIndex: 3, // 默认当前步骤为打印阶段
    progressPercent: 50, // 进度百分比
    steps: [
      { title: '订单确认', content: '确认订单信息和需求' },
      { title: '设计图确认', content: '确认头壳设计图纸' },
      { title: '模型制作', content: '根据设计图制作3D模型' },
      { title: '打印', content: '3D打印头壳部件' },
      { title: '打磨上色', content: '对打印件进行打磨和上色' },
      { title: '组装', content: '组装头壳各部件' },
      { title: '质检', content: '对成品进行质量检查' },
      { title: '发货', content: '包装并发货' }
    ] as Step[],
    isAdmin: false
  },

  onLoad: function(options) {
    // 从页面参数获取订单ID
    if (options) {
      if (options.id) {
        const orderId = options.id;
        this.setData({ orderId });
        this.loadOrderDetail(orderId);
      } else {
        // 如果没有获取到ID，使用默认ID
        this.loadOrderDetail('TB123456789');
      }
      
      // 检查是否是管理员模式
      if (options.admin === 'true') {
        this.setData({ isAdmin: true });
      }
    }
  },

  // 加载订单详情
  loadOrderDetail(orderId: string) {
    wx.showLoading({
      title: '加载中...'
    });
    
    // 模拟API请求
    setTimeout(() => {
      // 模拟数据
      let mockData: OrderDetail;
      let progressPercent = 0;
      
      // 根据订单ID返回不同的模拟数据
      if (orderId === 'TB987654321') {
        mockData = {
          orderId: 'TB987654321',
          queueId: 'RatStudio-2025-002',
          roleName: '猫咪头壳',
          expectedCompletionDate: '2025-11-15',
          currentStepIndex: 2, // 模型制作
          previewImage: ''
        };
        progressPercent = 30;
      } else if (orderId === 'TB456789123') {
        mockData = {
          orderId: 'TB456789123',
          queueId: 'RatStudio-2025-003',
          roleName: '兔子头壳',
          expectedCompletionDate: '2025-10-20',
          currentStepIndex: 6, // 质检
          previewImage: ''
        };
        progressPercent = 90;
      } else if (orderId === 'TB789123456') {
        mockData = {
          orderId: 'TB789123456',
          queueId: 'RatStudio-2025-004',
          roleName: '熊猫头壳',
          expectedCompletionDate: '2025-02-10',
          currentStepIndex: 1, // 设计图确认
          previewImage: ''
        };
        progressPercent = 20;
      } else {
        // 默认数据或TB123456789
        mockData = {
          orderId: orderId || 'TB123456789',
          queueId: 'RatStudio-2025-001',
          roleName: '狐狸头壳',
          expectedCompletionDate: '2025-12-30',
          currentStepIndex: 3, // 打印阶段
          previewImage: ''
        };
        progressPercent = 50;
      }
      
      this.setData({
        order: mockData,
        currentStepIndex: mockData.currentStepIndex,
        progressPercent: progressPercent
      });
      
      wx.hideLoading();
    }, 1000);
  },
  
  // 返回上一页
  onBack() {
    wx.navigateBack();
  },
  
  // 联系客服
  onContactService() {
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
  }
}) 