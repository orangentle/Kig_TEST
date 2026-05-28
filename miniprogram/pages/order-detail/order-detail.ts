// order-detail.ts
// 获取应用实例
const appInstance = getApp<IAppOption>();

interface OrderDetail {
  tbOrderId: string;  // 淘宝订单号
  queueNumber: string; // 排单号
  orderId: string;  // 系统订单号
  customerName: string; // 客户名称
  roleName: string; // 角色名称
  orderTime: string; // 下单时间
  deadline: string; // 预期完成时间
  progressPercent: number; // 制作进度
  progressStage: string; // 进度阶段描述
  stage: string; // 制作阶段
  status: string; // 订单状态
  previewImage?: string; // 预期成品展示图
}

interface Step {
  title: string;
  content: string;
}

Page({
  data: {
    tbOrderId: '',
    order: {
      tbOrderId: '',
      queueNumber: '',
      orderId: '',
      customerName: '',
      roleName: '',
      orderTime: '',
      deadline: '',
      progressPercent: 0,
      progressStage: '',
      stage: '',
      status: '',
      previewImage: ''
    } as OrderDetail,
    currentStepIndex: 0, // 当前步骤索引
    progressPercent: 0, // 进度百分比
    steps: [
      { title: '已排单', content: '订单确认，进入制作排队' },
      { title: '建模', content: '3D 建模与打印' },
      { title: '上妆', content: '打磨、喷漆与细节处理' },
      { title: '假毛', content: '毛发种植与造型' },
      { title: '已发货', content: '质检后包装并发出' }
    ] as Step[],
    isAdmin: false
  },

  onLoad: function(options) {
    // 从页面参数获取订单ID
    if (options) {
      if (options.id) {
        const tbOrderId = options.id;
        this.setData({ tbOrderId });
        this.loadOrderDetail(tbOrderId);
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
  loadOrderDetail(tbOrderId: string) {
    wx.showLoading({
      title: '加载中...'
    });
    
    // 使用getOrders云函数获取订单数据
    wx.cloud.callFunction({
      name: 'getOrders',
      data: { tbOrderId },
      success: (res: any) => {
        const orderList = res.result && res.result.data;
        
        if (orderList && orderList.length > 0) {
          // 如果成功获取到数据，使用真实数据
          const orderData = orderList.find((order: any) => order.tbOrderId === tbOrderId) || orderList[0];
          
          // 将制作阶段转换为步骤索引
          const stageToIndex = {
            'queued': 0,
            'modeling': 1,
            'painting': 2,
            'hair': 3,
            'shipped': 4
          };
          
          const currentStepIndex = stageToIndex[orderData.stage] || 0;
          
          this.setData({
            order: orderData,
            currentStepIndex: currentStepIndex,
            progressPercent: orderData.progressPercent || 0
          });
        } else {
          // 如果没有获取到数据，使用模拟数据
          this.loadMockOrderDetail(tbOrderId);
        }
        
        wx.hideLoading();
      },
      fail: (err) => {
        console.error('获取订单详情失败', err);
        // 失败时使用模拟数据
        this.loadMockOrderDetail(tbOrderId);
        wx.hideLoading();
      }
    });
  },
  
  // 加载模拟订单详情数据
  loadMockOrderDetail(tbOrderId: string) {
    // 模拟数据
    let mockData: OrderDetail;
    let progressPercent = 0;
    
    // 根据订单ID返回不同的模拟数据
    if (tbOrderId === 'TB987654321') {
      mockData = {
        tbOrderId: 'TB987654321',
        queueNumber: 'RatStudio-2025-002',
        orderId: 'KG20250002',
        customerName: '李小红',
        roleName: '猫咪头壳',
        orderTime: '2025-11-05',
        deadline: '2025-11-15',
        progressPercent: 30,
        progressStage: '建模',
        stage: 'modeling',
        status: 'normal',
        previewImage: ''
      };
      progressPercent = 30;
    } else if (tbOrderId === 'TB456789123') {
      mockData = {
        tbOrderId: 'TB456789123',
        queueNumber: 'RatStudio-2025-003',
        orderId: 'KG20250003',
        customerName: '张小华',
        roleName: '兔子头壳',
        orderTime: '2025-09-20',
        deadline: '2025-10-20',
        progressPercent: 80,
        progressStage: '假毛',
        stage: 'hair',
        status: 'soon',
        previewImage: ''
      };
      progressPercent = 80;
    } else if (tbOrderId === 'TB789123456') {
      mockData = {
        tbOrderId: 'TB789123456',
        queueNumber: 'RatStudio-2025-004',
        orderId: 'KG20250004',
        customerName: '赵小刚',
        roleName: '熊猫头壳',
        orderTime: '2025-11-20',
        deadline: '2025-02-10',
        progressPercent: 10,
        progressStage: '已排单',
        stage: 'queued',
        status: 'normal',
        previewImage: ''
      };
      progressPercent = 10;
    } else if (tbOrderId === 'TB123456789') {
      mockData = {
        tbOrderId: 'TB123456789',
        queueNumber: 'RatStudio-2025-001',
        orderId: 'KG20250001',
        customerName: '王小明',
        roleName: '狐狸头壳',
        orderTime: '2025-10-15',
        deadline: '2025-12-30',
        progressPercent: 55,
        progressStage: '上妆',
        stage: 'painting',
        status: 'urgent',
        previewImage: ''
      };
      progressPercent = 55;
    } else {
      // 默认数据
      mockData = {
        tbOrderId: tbOrderId || 'unknown',
        queueNumber: 'RatStudio-2025-000',
        orderId: 'KG20250000',
        customerName: '未知客户',
        roleName: '未知角色',
        orderTime: '未知',
        deadline: '未定',
        progressPercent: 10,
        progressStage: '已排单',
        stage: 'queued',
        status: 'normal',
        previewImage: ''
      };
      progressPercent = 10;
    }
    
    this.setData({
      order: mockData,
      currentStepIndex: this.getStepIndexFromStage(mockData.stage),
      progressPercent: progressPercent
    });
  },
  
  // 根据阶段获取步骤索引
  getStepIndexFromStage(stage: string): number {
    const stageToIndex = {
      'queued': 0,
      'modeling': 1,
      'painting': 2,
      'hair': 3,
      'shipped': 4
    };

    return stageToIndex[stage] || 0;
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