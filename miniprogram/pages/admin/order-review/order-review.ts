// order-review.ts
import Message from 'tdesign-miniprogram/message/index';

Page({
  data: {
    orders: [] as any[],
    currentTab: 0,
    showDetailPopup: false,
    currentOrder: null as any,
    isRefreshing: false,
    hasMore: true,
    pageSize: 10,
    lastId: '',
    tempTaobaoOrderId: ''
  },

  onLoad() {
    this.loadOrders();
  },

  onShow() {
    // 每次显示页面时刷新数据
    this.refreshOrders();
  },

  // Tab切换
  onTabChange(e: any) {
    const tab = e.detail.value;
    this.setData({ 
      currentTab: tab,
      orders: [],
      lastId: '',
      hasMore: true
    });
    this.loadOrders();
  },

  // 刷新订单
  async onRefresh() {
    this.setData({ isRefreshing: true });
    await this.refreshOrders();
    this.setData({ isRefreshing: false });
  },

  refreshOrders() {
    this.setData({
      orders: [],
      lastId: '',
      hasMore: true
    });
    return this.loadOrders();
  },

  // 加载更多
  loadMore() {
    if (this.data.hasMore) {
      this.loadOrders();
    }
  },

  // 加载订单列表
  async loadOrders() {
    try {
      const db = wx.cloud.database();
      const _ = db.command;
      
      // 根据Tab筛选状态
      let statusFilter: any = {};
      switch (this.data.currentTab) {
        case 0: // 待审核
          statusFilter = { status: 'pending' };
          break;
        case 1: // 已通过
          statusFilter = { status: _.in(['approved', 'processing', 'completed']) };
          break;
        case 2: // 已拒绝
          statusFilter = { status: 'rejected' };
          break;
        default: // 全部
          statusFilter = {};
      }
      
      let query = db.collection('orders').where(statusFilter).orderBy('createTime', 'desc');
      
      const result = await query.limit(this.data.pageSize).get();
      
      if (result.data && result.data.length > 0) {
        // 格式化时间
        const orders = result.data.map((order: any) => {
          return {
            ...order,
            createTimeStr: this.formatDate(order.createTime)
          };
        });
        
        this.setData({
          orders: [...this.data.orders, ...orders],
          lastId: result.data[result.data.length - 1]._id,
          hasMore: result.data.length === this.data.pageSize
        });
      } else {
        this.setData({ hasMore: false });
      }
    } catch (error) {
      console.error('加载订单失败', error);
      Message.error({ context: this, offset: [20, 32], content: '加载订单失败' });
    }
  },

  // 格式化日期
  formatDate(date: any) {
    if (!date) return '';
    const d = new Date(date);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  },

  // 查看订单详情
  viewOrderDetail(e: any) {
    const order = e.currentTarget.dataset.order;
    this.setData({
      currentOrder: order,
      showDetailPopup: true,
      tempTaobaoOrderId: order.taobaoOrderId || ''
    });
  },

  // 关闭详情弹窗
  onCloseDetailPopup() {
    this.setData({ showDetailPopup: false });
  },

  // 阻止冒泡
  stopPropagation() {},

  // 通过订单
  approveOrder(e: any) {
    const order = e.currentTarget.dataset.order;
    this.doApproveOrder(order);
  },

  approveCurrentOrder() {
    this.doApproveOrder(this.data.currentOrder);
  },

  async doApproveOrder(order: any) {
    wx.showModal({
      title: '确认通过',
      content: `确定要通过订单 ${order.orderId} 吗？`,
      success: async (res) => {
        if (res.confirm) {
          try {
            wx.showLoading({ title: '处理中...' });
            
            const db = wx.cloud.database();
            await db.collection('orders').doc(order._id).update({
              data: {
                status: 'approved',
                'reviewInfo.reviewTime': db.serverDate(),
                updateTime: db.serverDate()
              }
            });
            
            wx.hideLoading();
            Message.success({ context: this, offset: [20, 32], content: '订单已通过' });
            
            this.setData({ showDetailPopup: false });
            this.refreshOrders();
          } catch (error) {
            wx.hideLoading();
            console.error('审核失败', error);
            Message.error({ context: this, offset: [20, 32], content: '操作失败' });
          }
        }
      }
    });
  },

  // 拒绝订单
  rejectOrder(e: any) {
    const order = e.currentTarget.dataset.order;
    this.doRejectOrder(order);
  },

  rejectCurrentOrder() {
    this.doRejectOrder(this.data.currentOrder);
  },

  async doRejectOrder(order: any) {
    wx.showModal({
      title: '确认拒绝',
      content: `确定要拒绝订单 ${order.orderId} 吗？`,
      editable: true,
      placeholderText: '请输入拒绝原因（可选）',
      success: async (res) => {
        if (res.confirm) {
          try {
            wx.showLoading({ title: '处理中...' });
            
            const db = wx.cloud.database();
            await db.collection('orders').doc(order._id).update({
              data: {
                status: 'rejected',
                'reviewInfo.reviewTime': db.serverDate(),
                'reviewInfo.reviewRemark': res.content || '',
                updateTime: db.serverDate()
              }
            });
            
            wx.hideLoading();
            Message.success({ context: this, offset: [20, 32], content: '订单已拒绝' });
            
            this.setData({ showDetailPopup: false });
            this.refreshOrders();
          } catch (error) {
            wx.hideLoading();
            console.error('审核失败', error);
            Message.error({ context: this, offset: [20, 32], content: '操作失败' });
          }
        }
      }
    });
  },

  // 淘宝订单号变更
  onTaobaoOrderIdChange(e: any) {
    this.setData({ tempTaobaoOrderId: e.detail.value });
  },

  // 保存淘宝订单号
  async saveTaobaoOrderId() {
    const taobaoOrderId = this.data.tempTaobaoOrderId.trim();
    
    if (!taobaoOrderId) {
      Message.error({ context: this, offset: [20, 32], content: '请输入淘宝订单号' });
      return;
    }
    
    try {
      wx.showLoading({ title: '保存中...' });
      
      const db = wx.cloud.database();
      await db.collection('orders').doc(this.data.currentOrder._id).update({
        data: {
          taobaoOrderId: taobaoOrderId,
          orderId: taobaoOrderId, // 同时更新订单号为淘宝订单号
          status: 'processing', // 更新状态为制作中
          updateTime: db.serverDate()
        }
      });
      
      wx.hideLoading();
      Message.success({ context: this, offset: [20, 32], content: '淘宝订单号已保存' });
      
      // 更新当前订单数据
      this.setData({
        'currentOrder.taobaoOrderId': taobaoOrderId,
        'currentOrder.status': 'processing'
      });
      
      this.refreshOrders();
    } catch (error) {
      wx.hideLoading();
      console.error('保存失败', error);
      Message.error({ context: this, offset: [20, 32], content: '保存失败' });
    }
  },

  // 预览图片
  previewImage(e: any) {
    const { url, urls } = e.currentTarget.dataset;
    wx.previewImage({
      current: url,
      urls: urls
    });
  }
});
