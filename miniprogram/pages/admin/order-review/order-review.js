// order-review.js
var Message = require('tdesign-miniprogram/message/index').default || require('tdesign-miniprogram/message/index');

var TABS = [
  { key: 'pending',  label: '待审核' },
  { key: 'approved', label: '已通过' },
  { key: 'rejected', label: '已驳回' },
  { key: 'all',      label: '全部' }
];

Page({
  data: {
    tabs: TABS,
    currentTab: 'pending',
    orders: [],
    counts: { pending: 0, approved: 0, rejected: 0, all: 0 },
    searchValue: '',
    showDetailPopup: false,
    currentOrder: null,
    isRefreshing: false,
    hasMore: true,
    pageSize: 20,
    page: 1
  },

  onLoad: function() {
    this.loadCounts();
    this.loadOrders();
  },

  onShow: function() {
    this.refreshOrders();
    this.loadCounts();
  },

  onTabTap: function(e) {
    var key = e.currentTarget.dataset.key;
    if (key === this.data.currentTab) return;
    this.setData({ currentTab: key, orders: [], page: 1, hasMore: true });
    this.loadOrders();
  },

  onSearchChange: function(e) { this.setData({ searchValue: e.detail.value }); },
  onSearchSubmit: function() { this.refreshOrders(); },
  onSearchClear: function() {
    this.setData({ searchValue: '' });
    this.refreshOrders();
  },

  onRefresh: function() {
    var that = this;
    this.setData({ isRefreshing: true });
    Promise.all([this.refreshOrders(), this.loadCounts()]).then(function() {
      that.setData({ isRefreshing: false });
    });
  },

  refreshOrders: function() {
    this.setData({ orders: [], page: 1, hasMore: true });
    return this.loadOrders();
  },

  loadMore: function() {
    if (this.data.hasMore && this.data.orders.length > 0) {
      this.setData({ page: this.data.page + 1 });
      this.loadOrders(true);
    }
  },

  buildWhere: function() {
    var db = wx.cloud.database();
    var _ = db.command;
    var where = {};
    switch (this.data.currentTab) {
      case 'pending':  where.status = 'pending'; break;
      case 'approved': where.stage = _.in(['queued', 'modeling', 'painting', 'hair', 'shipped']); break;
      case 'rejected': where.status = 'rejected'; break;
    }
    var kw = (this.data.searchValue || '').trim();
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

  loadCounts: function() {
    var that = this;
    var db = wx.cloud.database();
    var _ = db.command;
    var col = db.collection('orders');
    return Promise.all([
      col.where({ status: 'pending' }).count(),
      col.where({ stage: _.in(['queued', 'modeling', 'painting', 'hair', 'shipped']) }).count(),
      col.where({ status: 'rejected' }).count(),
      col.count()
    ]).then(function(rs) {
      that.setData({
        counts: {
          pending: rs[0].total || 0,
          approved: rs[1].total || 0,
          rejected: rs[2].total || 0,
          all: rs[3].total || 0
        }
      });
    }).catch(function(err) { console.warn('计数加载失败', err); });
  },

  loadOrders: function(append) {
    var that = this;
    var db = wx.cloud.database();
    var where = this.buildWhere();
    return db.collection('orders')
      .where(where)
      .orderBy('createTime', 'desc')
      .skip((this.data.page - 1) * this.data.pageSize)
      .limit(this.data.pageSize)
      .get()
      .then(function(result) {
        var list = (result.data || []).map(function(order) {
          return Object.assign({}, order, {
            userInfo: order.userInfo || {},
            bodyMeasurements: order.bodyMeasurements || {},
            options: order.options || {},
            createTimeStr: that.formatDate(order.createTime)
          });
        });
        that.setData({
          orders: append ? that.data.orders.concat(list) : list,
          hasMore: list.length === that.data.pageSize
        });
      }).catch(function(error) {
        console.error('加载订单失败', error);
        Message.error({ context: that, offset: [20, 32], content: '加载订单失败' });
      });
  },

  formatDate: function(date) {
    if (!date) return '';
    var d = new Date(date);
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0')
      + ' ' + String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  },

  viewOrderDetail: function(e) {
    this.setData({ currentOrder: e.currentTarget.dataset.order, showDetailPopup: true });
  },

  onCloseDetailPopup: function() { this.setData({ showDetailPopup: false }); },

  stopPropagation: function() {},

  approveOrder: function(e) { this.doApproveOrder(e.currentTarget.dataset.order); },
  approveCurrentOrder: function() { this.doApproveOrder(this.data.currentOrder); },

  doApproveOrder: function(order) {
    var that = this;
    wx.showModal({
      title: '确认通过',
      content: '通过订单「' + (order.tbOrderId || order.orderId) + '」？通过后将进入"已排单"阶段。',
      confirmColor: '#ff8800',
      success: function(res) {
        if (!res.confirm) return;
        wx.showLoading({ title: '处理中...' });
        var db = wx.cloud.database();
        db.collection('orders').doc(order._id).update({
          data: {
            status: 'normal',
            stage: 'queued',
            progressStage: '已排单',
            progressPercent: 10,
            'reviewInfo.reviewTime': db.serverDate(),
            'reviewInfo.reviewRemark': '审核通过',
            updateTime: db.serverDate()
          }
        }).then(function() {
          wx.hideLoading();
          Message.success({ context: that, offset: [20, 32], content: '订单已通过' });
          that.setData({ showDetailPopup: false });
          that.refreshOrders();
          that.loadCounts();
        }).catch(function(error) {
          wx.hideLoading();
          console.error('审核失败', error);
          Message.error({ context: that, offset: [20, 32], content: '操作失败' });
        });
      }
    });
  },

  rejectOrder: function(e) { this.doRejectOrder(e.currentTarget.dataset.order); },
  rejectCurrentOrder: function() { this.doRejectOrder(this.data.currentOrder); },

  doRejectOrder: function(order) {
    var that = this;
    wx.showModal({
      title: '驳回订单',
      content: '驳回订单「' + (order.tbOrderId || order.orderId) + '」，请填写驳回原因：',
      editable: true,
      placeholderText: '如：身材数据缺失、参考图模糊…',
      confirmColor: '#cf1322',
      success: function(res) {
        if (!res.confirm) return;
        var remark = (res.content || '').trim() || '信息有误，请联系客服';
        wx.showLoading({ title: '处理中...' });
        var db = wx.cloud.database();
        db.collection('orders').doc(order._id).update({
          data: {
            status: 'rejected',
            stage: 'pending',
            progressStage: '已驳回',
            progressPercent: 0,
            'reviewInfo.reviewTime': db.serverDate(),
            'reviewInfo.reviewRemark': remark,
            updateTime: db.serverDate()
          }
        }).then(function() {
          wx.hideLoading();
          Message.success({ context: that, offset: [20, 32], content: '订单已驳回' });
          that.setData({ showDetailPopup: false });
          that.refreshOrders();
          that.loadCounts();
        }).catch(function(error) {
          wx.hideLoading();
          console.error('驳回失败', error);
          Message.error({ context: that, offset: [20, 32], content: '操作失败' });
        });
      }
    });
  },

  previewImage: function(e) {
    wx.previewImage({ current: e.currentTarget.dataset.url, urls: e.currentTarget.dataset.urls });
  }
});
