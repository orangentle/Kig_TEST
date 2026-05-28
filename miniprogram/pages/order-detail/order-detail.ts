// order-detail.ts

interface OrderDetail {
  tbOrderId: string;
  queueNumber: string;
  orderId: string;
  customerName: string;
  roleName: string;
  orderTime: string;
  deadline: string;
  progressPercent: number;
  progressStage: string;
  stage: string;
  status: string;
  previewImage?: string;
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
    currentStepIndex: 0,
    progressPercent: 0,
    steps: [
      { title: '已排单', content: '订单确认，进入制作排队' },
      { title: '建模', content: '3D 建模与打印' },
      { title: '上妆', content: '打磨、喷漆与细节处理' },
      { title: '假毛', content: '毛发种植与造型' },
      { title: '已发货', content: '质检后包装并发出' }
    ] as Step[],
    isAdmin: false,
    notFound: false,
    loadError: false
  },

  onLoad(options) {
    if (!options || !options.id) {
      this.setData({ notFound: true });
      return;
    }
    const tbOrderId = options.id;
    this.setData({
      tbOrderId,
      isAdmin: options.admin === 'true'
    });
    this.loadOrderDetail(tbOrderId);
  },

  loadOrderDetail(tbOrderId: string) {
    wx.showLoading({ title: '加载中...' });

    wx.cloud.callFunction({
      name: 'getOrders',
      data: { tbOrderId },
      success: (res: any) => {
        wx.hideLoading();
        const orderList = res.result && res.result.data;
        if (!orderList || orderList.length === 0) {
          this.setData({ notFound: true });
          return;
        }
        const orderData = orderList.find((o: any) => o.tbOrderId === tbOrderId) || orderList[0];
        this.setData({
          order: orderData,
          currentStepIndex: this.getStepIndexFromStage(orderData.stage),
          progressPercent: orderData.progressPercent || 0,
          notFound: false,
          loadError: false
        });
      },
      fail: (err) => {
        wx.hideLoading();
        console.error('获取订单详情失败', err);
        this.setData({ loadError: true });
        wx.showToast({ title: '加载失败，请重试', icon: 'none' });
      }
    });
  },

  getStepIndexFromStage(stage: string): number {
    const map: Record<string, number> = {
      queued: 0, modeling: 1, painting: 2, hair: 3, shipped: 4
    };
    return map[stage] || 0;
  },

  onBack() {
    wx.navigateBack();
  },

  onContactService() {
    wx.showModal({
      title: '联系客服',
      content: '即将打开客服会话',
      success: (res) => {
        if (res.confirm) {
          wx.showToast({ title: '客服功能开发中', icon: 'none' });
        }
      }
    });
  }
})
