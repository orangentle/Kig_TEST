// order-review.js
var Message = require('tdesign-miniprogram/message/index');

Page({
  data: {
    orders: [],
    currentTab: 0,
    showDetailPopup: false,
    currentOrder: null,
    isRefreshing: false,
    hasMore: true,
    pageSize: 10,
    lastId: '',
    tempTaobaoOrderId: ''
  },

  onLoad: function() {
    this.loadOrders();
  },

  onShow: function() {
    this.refreshOrders();
  },

  onTabChange: function(e) {
    var tab = e.detail.value;
    this.setData({ 
      currentTab: tab,
      orders: [],
      lastId: '',
      hasMore: true
    });
    this.loadOrders();
  },

  onRefresh: function() {
    var that = this;
    this.setData({ isRefreshing: true });
    this.refreshOrders().then(function() {
      that.setData({ isRefreshing: false });
    });
  },

  refreshOrders: function() {
    this.setData({
      orders: [],
      lastId: '',
      hasMore: true
    });
    return this.loadOrders();
  },

  loadMore: function() {
    if (this.data.hasMore) {
      this.loadOrders();
    }
  },

  loadOrders: function() {
    var that = this;
    var db = wx.cloud.database();
    var _ = db.command;
    
    var statusFilter = {};
    switch (this.data.currentTab) {
      case 0:
        statusFilter = { status: 'pending' };
        break;
      case 1:
        statusFilter = { status: _.in(['approved', 'processing', 'completed']) };
        break;
      case 2:
        statusFilter = { status: 'rejected' };
        break;
      default:
        statusFilter = {};
    }
    
    var query = db.collection('orders').where(statusFilter).orderBy('createTime', 'desc');
    
    return query.limit(this.data.pageSize).get().then(function(result) {
      if (result.data && result.data.length > 0) {
        var orders = result.data.map(function(order) {
          return Object.assign({}, order, {
            createTimeStr: that.formatDate(order.createTime)
          });
        });
        
        that.setData({
          orders: that.data.orders.concat(orders),
          lastId: result.data[result.data.length - 1]._id,
          hasMore: result.data.length === that.data.pageSize
        });
      } else {
        that.setData({ hasMore: false });
      }
    }).catch(function(error) {
      console.error('加载订单失败', error);
      Message.error({ context: that, offset: [20, 32], content: '加载订单失败' });
    });
  },

  formatDate: function(date) {
    if (!date) return '';
    var d = new Date(date);
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0') + ' ' + String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  },

  viewOrderDetail: function(e) {
    var order = e.currentTarget.dataset.order;
    this.setData({
      currentOrder: order,
      showDetailPopup: true,
      tempTaobaoOrderId: order.taobaoOrderId || ''
    });
  },

  onCloseDetailPopup: function() {
    this.setData({ showDetailPopup: false });
  },

  stopPropagation: function() {},

  approveOrder: function(e) {
    var order = e.currentTarget.dataset.order;
    this.doApproveOrder(order);
  },

  approveCurrentOrder: function() {
    this.doApproveOrder(this.data.currentOrder);
  },

  doApproveOrder: function(order) {
    var that = this;
    wx.showModal({
      title: '确认通过',
      content: '确定要通过订单 ' + order.orderId + ' 吗？',
      success: function(res) {
        if (res.confirm) {
          wx.showLoading({ title: '处理中...' });
          
          var db = wx.cloud.database();
          db.collection('orders').doc(order._id).update({
            data: {
              status: 'approved',
              'reviewInfo.reviewTime': db.serverDate(),
              updateTime: db.serverDate()
            }
          }).then(function() {
            wx.hideLoading();
            Message.success({ context: that, offset: [20, 32], content: '订单已通过' });
            that.setData({ showDetailPopup: false });
            that.refreshOrders();
          }).catch(function(error) {
            wx.hideLoading();
            console.error('审核失败', error);
            Message.error({ context: that, offset: [20, 32], content: '操作失败' });
          });
        }
      }
    });
  },

  rejectOrder: function(e) {
    var order = e.currentTarget.dataset.order;
    this.doRejectOrder(order);
  },

  rejectCurrentOrder: function() {
    this.doRejectOrder(this.data.currentOrder);
  },

  doRejectOrder: function(order) {
    var that = this;
    wx.showModal({
      title: '确认拒绝',
      content: '确定要拒绝订单 ' + order.orderId + ' 吗？',
      editable: true,
      placeholderText: '请输入拒绝原因（可选）',
      success: function(res) {
        if (res.confirm) {
          wx.showLoading({ title: '处理中...' });
          
          var db = wx.cloud.database();
          db.collection('orders').doc(order._id).update({
            data: {
              status: 'rejected',
              'reviewInfo.reviewTime': db.serverDate(),
              'reviewInfo.reviewRemark': res.content || '',
              updateTime: db.serverDate()
            }
          }).then(function() {
            wx.hideLoading();
            Message.success({ context: that, offset: [20, 32], content: '订单已拒绝' });
            that.setData({ showDetailPopup: false });
            that.refreshOrders();
          }).catch(function(error) {
            wx.hideLoading();
            console.error('审核失败', error);
            Message.error({ context: that, offset: [20, 32], content: '操作失败' });
          });
        }
      }
    });
  },

  onTaobaoOrderIdChange: function(e) {
    this.setData({ tempTaobaoOrderId: e.detail.value });
  },

  saveTaobaoOrderId: function() {
    var that = this;
    var taobaoOrderId = this.data.tempTaobaoOrderId.trim();
    
    if (!taobaoOrderId) {
      Message.error({ context: this, offset: [20, 32], content: '请输入淘宝订单号' });
      return;
    }
    
    wx.showLoading({ title: '保存中...' });
    
    var db = wx.cloud.database();
    db.collection('orders').doc(this.data.currentOrder._id).update({
      data: {
        taobaoOrderId: taobaoOrderId,
        orderId: taobaoOrderId,
        status: 'processing',
        updateTime: db.serverDate()
      }
    }).then(function() {
      wx.hideLoading();
      Message.success({ context: that, offset: [20, 32], content: '淘宝订单号已保存' });
      that.setData({
        'currentOrder.taobaoOrderId': taobaoOrderId,
        'currentOrder.status': 'processing'
      });
      that.refreshOrders();
    }).catch(function(error) {
      wx.hideLoading();
      console.error('保存失败', error);
      Message.error({ context: that, offset: [20, 32], content: '保存失败' });
    });
  },

  previewImage: function(e) {
    var url = e.currentTarget.dataset.url;
    var urls = e.currentTarget.dataset.urls;
    wx.previewImage({
      current: url,
      urls: urls
    });
  }
});
