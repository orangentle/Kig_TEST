// order-detail.ts
import type { Order } from '../../types/order';

type OrderDetail = Order;

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
      status: ''
    } as unknown as OrderDetail,
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
    wx.showLoading({ title: '鼠鼠在搬数据~' });

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
        // 兼容老订单没有 orderTime 字段: 用 createTime 兜底
        if (!orderData.orderTime && orderData.createTime) {
          const d = new Date(orderData.createTime);
          if (!isNaN(d.getTime())) {
            orderData.orderTime = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
          }
        }
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
        wx.showToast({ title: '咦,加载迷路了,再试一次?', icon: 'none' });
      }
    });
  },

  getStepIndexFromStage(stage: string): number {
    if (stage === 'pending') return -1;
    const map: Record<string, number> = {
      queued: 0, modeling: 1, painting: 2, hair: 3, shipped: 4
    };
    return map[stage] || 0;
  },

  onBack() {
    wx.navigateBack();
  },

  onPreviewImage(e: any) {
    const { urls, current } = e.currentTarget.dataset;
    if (!urls || !urls.length) return;
    wx.previewImage({ urls, current });
  },

  onAdminCopy(e: any) {
    const text = e.currentTarget.dataset.text;
    if (!text) return;
    wx.setClipboardData({
      data: String(text),
      success: () => wx.showToast({ title: '复制好啦~', icon: 'success' })
    });
  },

  onAdminCall(e: any) {
    const text = e.currentTarget.dataset.text;
    if (!text) return;
    wx.makePhoneCall({
      phoneNumber: String(text),
      fail: () => wx.showToast({ title: '拨打失败 (´;ω;`)', icon: 'none' })
    });
  }
})
