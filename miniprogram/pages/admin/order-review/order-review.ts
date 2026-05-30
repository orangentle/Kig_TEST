// order-review.ts
import Message from 'tdesign-miniprogram/message/index';

const TABS = [
  { key: 'pending',  label: '待审核' },
  { key: 'approved', label: '已通过' },
  { key: 'all',      label: '全部' }
];

Page({
  data: {
    tabs: TABS,
    currentTab: 'pending',
    orders: [] as any[],
    counts: { pending: 0, approved: 0, all: 0 } as Record<string, number>,
    searchValue: '',
    showDetailPopup: false,
    currentOrder: null as any,
    isRefreshing: false,
    hasMore: true,
    pageSize: 20,
    page: 1
  },

  onLoad() {
    this.loadCounts();
    this.loadOrders();
  },

  onShow() {
    this.refreshOrders();
    this.loadCounts();
  },

  onTabTap(e: any) {
    const key = e.currentTarget.dataset.key;
    if (key === this.data.currentTab) return;
    this.setData({ currentTab: key, orders: [], page: 1, hasMore: true });
    this.loadOrders();
  },

  onSearchChange(e: any) {
    this.setData({ searchValue: e.detail.value });
  },
  onSearchSubmit() { this.refreshOrders(); },
  onSearchClear() {
    this.setData({ searchValue: '' });
    this.refreshOrders();
  },

  async onRefresh() {
    this.setData({ isRefreshing: true });
    await Promise.all([this.refreshOrders(), this.loadCounts()]);
    this.setData({ isRefreshing: false });
  },

  refreshOrders() {
    this.setData({ orders: [], page: 1, hasMore: true });
    return this.loadOrders();
  },

  loadMore() {
    if (this.data.hasMore && this.data.orders.length > 0) {
      this.setData({ page: this.data.page + 1 });
      this.loadOrders(true);
    }
  },

  buildWhere() {
    const db = wx.cloud.database();
    const _ = db.command;
    let where: any = {};
    switch (this.data.currentTab) {
      case 'pending':  where.status = 'pending'; break;
      case 'approved': where.stage = _.in(['queued', 'modeling', 'painting', 'hair', 'shipped']); break;
    }
    const kw = (this.data.searchValue || '').trim();
    if (kw) {
      where = _.and([
        where,
        _.or([
          { tbOrderId: db.RegExp({ regexp: kw, options: 'i' }) },
          { roleName: db.RegExp({ regexp: kw, options: 'i' }) },
          { 'userInfo.nickName': db.RegExp({ regexp: kw, options: 'i' }) },
          { 'userInfo.taobaoName': db.RegExp({ regexp: kw, options: 'i' }) },
          { customerName: db.RegExp({ regexp: kw, options: 'i' }) }
        ])
      ]);
    }
    return where;
  },

  async loadCounts() {
    try {
      const db = wx.cloud.database();
      const _ = db.command;
      const col = db.collection('orders');
      const [pending, approved, all] = await Promise.all([
        col.where({ status: 'pending' }).count(),
        col.where({ stage: _.in(['queued', 'modeling', 'painting', 'hair', 'shipped']) }).count(),
        col.count()
      ]);
      this.setData({
        counts: {
          pending: pending.total || 0,
          approved: approved.total || 0,
          all: all.total || 0
        }
      });
    } catch (err) {
      console.warn('计数加载失败', err);
    }
  },

  async loadOrders(append: boolean = false) {
    try {
      const db = wx.cloud.database();
      const where = this.buildWhere();
      const result = await db.collection('orders')
        .where(where)
        .orderBy('createTime', 'desc')
        .skip((this.data.page - 1) * this.data.pageSize)
        .limit(this.data.pageSize)
        .get();

      const list = (result.data || []).map((order: any) => ({
        ...order,
        userInfo: order.userInfo || {},
        bodyMeasurements: order.bodyMeasurements || {},
        options: order.options || {},
        createTimeStr: this.formatDate(order.createTime)
      }));

      this.setData({
        orders: append ? [...this.data.orders, ...list] : list,
        hasMore: list.length === this.data.pageSize
      });
    } catch (error) {
      console.error('加载订单失败', error);
      Message.error({ context: this, offset: [20, 32], content: '加载订单失败' });
    }
  },

  formatDate(date: any) {
    if (!date) return '';
    const d = new Date(date);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  },

  viewOrderDetail(e: any) {
    const order = e.currentTarget.dataset.order;
    this.setData({ currentOrder: order, showDetailPopup: true });
  },

  onCloseDetailPopup() {
    this.setData({ showDetailPopup: false });
  },

  stopPropagation() {},

  approveOrder(e: any) { this.doApproveOrder(e.currentTarget.dataset.order); },
  approveCurrentOrder() { this.doApproveOrder(this.data.currentOrder); },

  async doApproveOrder(order: any) {
    wx.showModal({
      title: '确认通过',
      content: `通过订单「${order.tbOrderId || order.orderId}」？通过后将进入"已排单"阶段。`,
      confirmColor: '#ff8800',
      success: async (res) => {
        if (!res.confirm) return;
        try {
          wx.showLoading({ title: '处理中...' });
          const db = wx.cloud.database();
          await db.collection('orders').doc(order._id).update({
            data: {
              status: 'normal',
              stage: 'queued',
              progressStage: '已排单',
              progressPercent: 10,
              'reviewInfo.reviewTime': db.serverDate(),
              'reviewInfo.reviewRemark': '审核通过',
              updateTime: db.serverDate()
            }
          });
          wx.hideLoading();
          Message.success({ context: this, offset: [20, 32], content: '订单已通过' });
          this.setData({ showDetailPopup: false });
          this.refreshOrders();
          this.loadCounts();
        } catch (error) {
          wx.hideLoading();
          console.error('审核失败', error);
          Message.error({ context: this, offset: [20, 32], content: '操作失败' });
        }
      }
    });
  },

  rejectOrder(e: any) { this.doRejectOrder(e.currentTarget.dataset.order); },
  rejectCurrentOrder() { this.doRejectOrder(this.data.currentOrder); },

  async doRejectOrder(order: any) {
    wx.showModal({
      title: '驳回订单',
      content: `驳回订单「${order.tbOrderId || order.orderId}」会直接删除这条记录,并通知客户重新提交。请填写驳回原因:`,
      editable: true,
      placeholderText: '如:身材数据缺失、参考图模糊、淘宝单号填错…',
      confirmColor: '#cf1322',
      success: async (res) => {
        if (!res.confirm) return;
        const remark = (res.content || '').trim() || '信息有误,请按提示重新下单';
        try {
          wx.showLoading({ title: '处理中...' });
          const { result } = await wx.cloud.callFunction({
            name: 'batchUpdateOrders',
            data: {
              orderIds: [order._id],
              action: 'review-reject',
              payload: { remark }
            }
          }) as any;
          wx.hideLoading();
          if (result && result.success) {
            Message.success({ context: this, offset: [20, 32], content: '已驳回并删除,客户会收到通知' });
            this.setData({ showDetailPopup: false });
            this.refreshOrders();
            this.loadCounts();
          } else {
            throw new Error((result && result.error) || '操作失败');
          }
        } catch (error: any) {
          wx.hideLoading();
          console.error('驳回失败', error);
          Message.error({ context: this, offset: [20, 32], content: error.message || '操作失败' });
        }
      }
    });
  },

  previewImage(e: any) {
    const { url, urls } = e.currentTarget.dataset;
    wx.previewImage({ current: url, urls });
  }
});
