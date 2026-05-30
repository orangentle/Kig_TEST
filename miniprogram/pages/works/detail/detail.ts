const CATEGORY_LABEL: Record<string, string> = {
  original: '自设',
  game: '游戏',
  anime: '动漫'
};

Page({
  data: {
    id: '',
    work: {
      roleName: '',
      source: '',
      category: '',
      categoryLabel: '',
      coverFileId: '',
      price: null as number | null,
      date: '',
      createTime: 0
    },
    isLoading: false,
    panDistance: 0,
    coverMode: 'aspectFill' as 'aspectFill' | 'widthFix'
  },

  onLoad(options: any) {
    const id = options?.id || '';
    this.setData({ id });
    if (id) {
      this.fetchDetail(id);
    } else {
      wx.showToast({ title: '未找到作品ID', icon: 'none' });
    }
  },

  async fetchDetail(id: string) {
    this.setData({ isLoading: true });
    try {
      const db = wx.cloud.database();
      const res = await db.collection('works').doc(id).get();
      const item = res.data;
      if (!item) throw new Error('作品不存在');

      const fileId = item.coverFileId || item.imageFileId || '';
      let imageUrl = '';

      if (fileId) {
        try {
          const urlRes = await wx.cloud.getTempFileURL({ fileList: [fileId] });
          if (urlRes.fileList && urlRes.fileList.length > 0) {
            imageUrl = urlRes.fileList[0].tempFileURL || '';
          }
        } catch (error) {
          console.error('获取图片URL失败', error);
        }
      }

      // 先量好图片尺寸，确定渲染模式与浮动距离，再一次性 setData，避免渲染过程中
      // 切换 mode 触发重排导致动画卡顿。
      let coverMode: 'aspectFill' | 'widthFix' = 'aspectFill';
      let panDistance = 0;
      if (imageUrl) {
        try {
          const info = await new Promise<WechatMiniprogram.GetImageInfoSuccessCallbackResult>((resolve, reject) => {
            wx.getImageInfo({ src: imageUrl, success: resolve, fail: reject });
          });
          const w = info.width;
          const h = info.height;
          if (w && h) {
            const containerRpx = 720;
            const displayedRpx = 750 * (h / w);
            const overflow = displayedRpx - containerRpx;
            if (overflow > 24) {
              coverMode = 'widthFix';
              panDistance = -Math.round(overflow);
            }
          }
        } catch (e) {
          console.warn('封面尺寸探测失败', e);
        }
      }

      this.setData({
        coverMode,
        panDistance,
        work: {
          roleName: item.roleName || item.title || '未命名角色',
          source: item.source || item.description || '作品',
          category: item.category || 'original',
          categoryLabel: CATEGORY_LABEL[item.category || 'original'] || '未分类',
          coverFileId: imageUrl,
          price: item.price,
          date: this.formatDate(item.createTime || Date.now()),
          createTime: item.createTime || Date.now()
        }
      });
    } catch (error) {
      console.error('加载作品详情失败', error);
      wx.showToast({ title: '加载迷路了 (´·ω·`)', icon: 'none' });
    } finally {
      this.setData({ isLoading: false });
    }
  },

  goToOrder() {
    wx.switchTab({
      url: '/pages/order/order',
      fail: () => {
        wx.navigateTo({ url: '/pages/order/order' });
      }
    });
  },

  previewCover() {
    const url = this.data.work?.coverFileId;
    if (!url) return;
    wx.previewImage({ current: url, urls: [url] });
  },

  formatDate(ts: number) {
    const d = new Date(ts);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  }
});
