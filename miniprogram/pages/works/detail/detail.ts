Page({
  data: {
    id: '',
    work: {
      roleName: '',
      source: '',
      category: '',
      coverFileId: '',
      price: null as number | null,
      date: '',
      createTime: 0
    },
    isLoading: false
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
      
      // 获取云存储图片的临时链接
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
      
      this.setData({
        work: {
          roleName: item.roleName || item.title || '未命名角色',
          source: item.source || item.description || '作品',
          category: item.category || 'original',
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

  formatDate(ts: number) {
    const d = new Date(ts);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  }
});
