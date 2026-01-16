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
      this.setData({
        work: {
          roleName: item.roleName || item.title || '未命名角色',
          source: item.source || item.description || '作品',
          category: item.category || 'original',
          coverFileId: item.coverFileId || item.imageFileId || '',
          price: item.price,
          date: this.formatDate(item.createTime || Date.now()),
          createTime: item.createTime || Date.now()
        }
      });
    } catch (error) {
      console.error('加载作品详情失败', error);
      wx.showToast({ title: '加载失败', icon: 'none' });
    } finally {
      this.setData({ isLoading: false });
    }
  },

  formatDate(ts: number) {
    const d = new Date(ts);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  }
});
