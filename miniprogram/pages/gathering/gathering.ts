// gathering/gathering.ts - 偶壳娃聚主页
import { getPhotoUrl, getEventCoverUrl } from '../../utils/cloudStorage';

interface EventItem {
  _id: string;
  title: string;
  coverUrl: string;
  month: string;
  day: string;
  endDay?: string;
  location: string;
  remainingSpots: number | string;
  status: string;
  statusText: string;
}

interface PhotoItem {
  url: string;
  eventName: string;
}

interface SignupItem {
  _id: string;
  eventTitle: string;
  eventDate: string;
  status: string;
  statusText: string;
}

Page({
  data: {
    activeEvents: 2,
    albumCount: 128,
    brandLogoUrl: '',
    upcomingEvents: [] as EventItem[],
    recentPhotos: [] as PhotoItem[],
    mySignups: [] as SignupItem[],
  },

  onLoad() {
    this.loadBrandLogo();
    this.loadData();
  },

  onShow() {
    // 每次显示时刷新数据
    this.loadData();
  },

  onPullDownRefresh() {
    this.loadData().then(() => {
      wx.stopPullDownRefresh();
    });
  },

  async loadBrandLogo() {
    const app = getApp<IAppOption>();
    const url = app.globalData.brandLogoUrl || (app.globalData.brandLogoReady ? await app.globalData.brandLogoReady : '');
    if (url) {
      this.setData({ brandLogoUrl: url });
    }
  },

  async loadData() {
    // 加载近期活动
    this.loadUpcomingEvents();
    // 加载精彩照片
    this.loadRecentPhotos();
    // 加载我的报名
    this.loadMySignups();
  },

  loadUpcomingEvents() {
    // TODO: 从云数据库加载
    const events: EventItem[] = [
      {
        _id: 'event_003',
        title: 'OKR1.0聚会',
        coverUrl: getEventCoverUrl('event_003'),
        month: '05',
        day: '02',
        endDay: '03',
        location: '沈阳棋盘山绿地铂瑞酒店',
        remainingSpots: '不限',
        status: 'upcoming',
        statusText: '敬请期待'
      }
    ];
    
    this.setData({
      upcomingEvents: events,
      activeEvents: events.filter(e => e.status === 'open').length
    });
  },

  loadRecentPhotos() {
    // TODO: 从云存储加载最新照片，暂用品牌 logo 占位
    const defaultPhoto = this.data.brandLogoUrl;
    const photos: PhotoItem[] = [
      { url: defaultPhoto, eventName: 'OKR0.0启动聚' },
      { url: defaultPhoto, eventName: 'OKR0.0启动聚' },
      { url: defaultPhoto, eventName: 'OKR0.0启动聚' },
      { url: defaultPhoto, eventName: 'OKR0.0启动聚' },
      { url: defaultPhoto, eventName: 'OKR0.0启动聚' },
    ];

    this.setData({
      recentPhotos: photos,
      albumCount: photos.length
    });
  },

  loadMySignups() {
    // TODO: 根据用户openid从云数据库加载
    const signups: SignupItem[] = [
      {
        _id: 'event_001',
        eventTitle: 'OKR0.0启动聚',
        eventDate: '2025-10-01',
        status: 'confirmed',
        statusText: '已参加'
      }
    ];
    
    this.setData({ mySignups: signups });
  },

  // 跳转到娃聚报名
  goToSignup() {
    wx.navigateTo({
      url: '/pages/gathering/signup/signup'
    });
  },

  // 跳转到活动安排
  goToSchedule() {
    wx.navigateTo({
      url: '/pages/gathering/schedule/schedule'
    });
  },

  // 跳转到鼠鼠周边
  goToMerch() {
    wx.navigateTo({
      url: '/pages/gathering/merch/merch'
    });
  },

  // 跳转到聚会相册
  goToAlbum() {
    wx.navigateTo({
      url: '/pages/gathering/album/album'
    });
  },

  // 跳转到活动详情
  goToEventDetail(e: WechatMiniprogram.TouchEvent) {
    const id = e.currentTarget.dataset.id;
    wx.navigateTo({
      url: `/pages/gathering/signup/signup?eventId=${id}`
    });
  },

  // 预览照片
  previewPhoto(e: WechatMiniprogram.TouchEvent) {
    const index = e.currentTarget.dataset.index;
    const urls = this.data.recentPhotos.map(p => p.url);
    wx.previewImage({
      current: urls[index],
      urls: urls
    });
  }
});
