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

  buildQueryParams() {
    const params: any = {
      page: this.data.page,
      pageSize: this.data.pageSize
    };
    switch (this.data.currentTab) {
      case 'pending':  params.status = 'pending'; break;
      case 'approved': params.stage = ['queued', 'modeling', 'painting', 'hair', 'shipped']; break;
    }
    const kw = (this.data.searchValue || '').trim();
    if (kw) params.keyword = kw;
    return params;
  },

  async loadCounts() {
    try {
      const [pending, approved, all] = await Promise.all([
        wx.cloud.callFunction({ name: 'getOrders', data: { status: 'pending', countOnly: true } }),
        wx.cloud.callFunction({ name: 'getOrders', data: { stage: ['queued', 'modeling', 'painting', 'hair', 'shipped'], countOnly: true } }),
        wx.cloud.callFunction({ name: 'getOrders', data: { countOnly: true } })
      ]) as any[];
      this.setData({
        counts: {
          pending: (pending.result && pending.result.total) || 0,
          approved: (approved.result && approved.result.total) || 0,
          all: (all.result && all.result.total) || 0
        }
      });
    } catch (err) {
      console.warn('计数加载失败', err);
    }
  },

  async loadOrders(append: boolean = false) {
    try {
      const { result } = await wx.cloud.callFunction({
        name: 'getOrders',
        data: this.buildQueryParams()
      }) as any;
      if (!result || !result.success) {
        throw new Error((result && result.error) || '加载失败');
      }

      const list = (result.data || []).map((order: any) => ({
        ...order,
        userInfo: order.userInfo || {},
        bodyMeasurements: order.bodyMeasurements || {},
        options: order.options || {},
        createTimeStr: this.formatDate(order.createTime)
      }));

      this.setData({
        orders: append ? [...this.data.orders, ...list] : list,
        hasMore: !!result.hasMore
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
          const { result } = await wx.cloud.callFunction({
            name: 'batchUpdateOrders',
            data: { orderIds: [order._id], action: 'review-approve' }
          }) as any;
          wx.hideLoading();
          if (result && result.success) {
            Message.success({ context: this, offset: [20, 32], content: '订单已通过' });
            this.setData({ showDetailPopup: false });
            this.refreshOrders();
            this.loadCounts();
          } else {
            throw new Error((result && result.error) || '操作失败');
          }
        } catch (error: any) {
          wx.hideLoading();
          console.error('审核失败', error);
          Message.error({ context: this, offset: [20, 32], content: error.message || '操作失败' });
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
