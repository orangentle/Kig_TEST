// pages/gathering/signup/signup.ts

interface EventDetail {
  id: string;
  name: string;
  coverUrl: string;
  date: string;
  time: string;
  location: string;
  capacity: number;
  signupCount: number;
  fee: number;
  status: 'open' | 'upcoming' | 'closed' | 'full';
  statusText: string;
  canSignup: boolean;
  hasSignup: boolean;
  fullText?: string;
  description: string;
}

interface SignupRecord {
  id: string;
  eventId: string;
  eventName: string;
  signupTime: string;
  status: 'confirmed' | 'pending' | 'cancelled';
  statusText: string;
}

Page({
  data: {
    showNotice: true,
    events: [] as EventDetail[],
    mySignups: [] as SignupRecord[],
    statusOptions: [
      { label: '全部活动', value: 'all' },
      { label: '可报名', value: 'open' },
      { label: '即将开始', value: 'upcoming' },
      { label: '已结束', value: 'closed' }
    ],
    currentStatus: { label: '全部活动', value: 'all' },
    showDetail: false,
    currentEvent: {} as EventDetail
  },

  async onLoad() {
    await this.loadEvents();
    this.loadMySignups();
  },

  onPullDownRefresh() {
    this.loadEvents();
    this.loadMySignups();
    wx.stopPullDownRefresh();
  },

  goBack() {
    wx.navigateBack();
  },

  closeNotice() {
    this.setData({ showNotice: false });
  },

  async loadEvents() {
    const app = getApp<IAppOption>();
    const defaultCover = app.globalData.brandLogoUrl || (app.globalData.brandLogoReady ? await app.globalData.brandLogoReady : '');
    const mockEvents: EventDetail[] = [
      {
        id: 'event_001',
        name: 'OKR0.0启动聚',
        coverUrl: defaultCover,
        date: '2025-10-01',
        time: '09:00-17:00',
        location: '沈阳乌托邦聚会别墅V4和V5',
        capacity: 100,
        signupCount: 100,
        fee: 0,
        status: 'closed',
        statusText: '已结束',
        canSignup: false,
        hasSignup: true,
        description: '偶壳OKR首次启动聚会！感谢所有参与的娃友们，让我们一起见证了OKR社区的诞生。活动包含娃娃摄影、手作交流、周边交换等精彩环节，大家度过了难忘的一天！'
      },
      {
        id: 'event_003',
        name: 'OKR1.0聚会',
        coverUrl: defaultCover,
        date: '2026-05-02 至 2026-05-03（暂定）',
        time: '待定',
        location: '沈阳棋盘山绿地铂瑞酒店',
        capacity: 0,
        signupCount: 0,
        fee: 0,
        status: 'upcoming',
        statusText: '即将开始',
        canSignup: false,
        hasSignup: false,
        fullText: '敬请期待',
        description: 'OKR1.0正式聚会！地点定在风景优美的沈阳棋盘山绿地铂瑞酒店，人数不限。具体时间和活动安排敬请期待，欢迎关注后续通知！'
      }
    ];
    
    this.setData({ events: mockEvents });
  },

  loadMySignups() {
    // 模拟加载我的报名记录
    const mockSignups: SignupRecord[] = [
      {
        id: '1',
        eventId: 'event_001',
        eventName: 'OKR0.0启动聚',
        signupTime: '2025-09-15 10:30',
        status: 'confirmed',
        statusText: '已确认'
      }
    ];
    
    this.setData({ mySignups: mockSignups });
  },

  onStatusChange(e: WechatMiniprogram.PickerChange) {
    const index = Number(e.detail.value);
    const option = this.data.statusOptions[index];
    this.setData({ currentStatus: option });
    // TODO: 根据筛选条件重新加载数据
  },

  showEventDetail(e: WechatMiniprogram.TouchEvent) {
    const event = e.currentTarget.dataset.event;
    this.setData({
      currentEvent: event,
      showDetail: true
    });
  },

  closeDetail() {
    this.setData({ showDetail: false });
  },

  onDetailClose(e: WechatMiniprogram.CustomEvent) {
    if (!e.detail.visible) {
      this.setData({ showDetail: false });
    }
  },

  doSignup(e: WechatMiniprogram.TouchEvent) {
    const event = e.currentTarget.dataset.event;
    
    wx.showModal({
      title: '确认报名',
      content: `确定要报名「${event.name}」吗？${event.fee > 0 ? `费用：¥${event.fee}` : '免费活动'}`,
      confirmText: '确认报名',
      confirmColor: '#ff8800',
      success: (res) => {
        if (res.confirm) {
          wx.showLoading({ title: '报名中...' });
          
          // TODO: 调用云函数进行报名
          setTimeout(() => {
            wx.hideLoading();
            wx.showToast({
              title: '报名成功',
              icon: 'success'
            });
            this.setData({ showDetail: false });
            this.loadEvents();
            this.loadMySignups();
          }, 1000);
        }
      }
    });
  },

  showSignupDetail(e: WechatMiniprogram.TouchEvent) {
    const item = e.currentTarget.dataset.item;
    wx.showModal({
      title: item.eventName,
      content: `报名时间：${item.signupTime}\n状态：${item.statusText}`,
      showCancel: item.status !== 'cancelled',
      cancelText: '取消报名',
      cancelColor: '#ff4d4f',
      confirmText: '我知道了',
      success: (res) => {
        if (!res.confirm && item.status !== 'cancelled') {
          this.cancelSignup(item);
        }
      }
    });
  },

  cancelSignup(_item: SignupRecord) {
    wx.showModal({
      title: '确认取消',
      content: '确定要取消报名吗？',
      confirmText: '确认取消',
      confirmColor: '#ff4d4f',
      success: (res) => {
        if (res.confirm) {
          wx.showLoading({ title: '取消中...' });
          // TODO: 调用云函数取消报名
          setTimeout(() => {
            wx.hideLoading();
            wx.showToast({
              title: '已取消',
              icon: 'success'
            });
            this.loadMySignups();
          }, 500);
        }
      }
    });
  },

  goToMySignups() {
    // TODO: 跳转到完整的报名记录页面
    wx.showToast({
      title: '功能开发中',
      icon: 'none'
    });
  }
});
