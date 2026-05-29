interface WorkItem {
  _id?: string;
  roleName: string;
  source: string;
  price?: number;
  category: 'original' | 'game' | 'anime';
  coverFileId: string;
  coverUrl?: string;
  isPublished: boolean;
  createTime: number;
  displayDate?: string;
  categoryLabel?: string;
}

const CATEGORY_LABEL: Record<string, string> = {
  original: '自设',
  game: '游戏',
  anime: '动漫'
};

Page({
  data: {
    form: {
      roleName: '',
      source: '',
      price: '',
      categoryIndex: 0,
      isPublished: true,
      coverFileId: '',
      coverPreview: ''
    },
    categories: [
      { label: '自设角色', value: 'original' },
      { label: '游戏角色', value: 'game' },
      { label: '动漫角色', value: 'anime' }
    ],
    works: [] as WorkItem[],
    isLoading: false,
    isSubmitting: false,
    isUploading: false
  },

  onLoad() {
    this.fetchWorks();
  },

  async fetchWorks() {
    this.setData({ isLoading: true });
    try {
      const db = wx.cloud.database();
      const res = await db.collection('works').orderBy('createTime', 'desc').get();
      const rawWorks = (res.data || []).map((item: any) => {
        const category = item.category || 'original';
        return {
          _id: item._id,
          roleName: item.roleName || item.title || '未命名角色',
          source: item.source || item.description || '作品',
          price: item.price,
          category,
          categoryLabel: CATEGORY_LABEL[category] || '作品',
          coverFileId: item.coverFileId || item.imageFileId || '',
          coverUrl: '',
          isPublished: item.isPublished !== false,
          createTime: item.createTime || Date.now(),
          displayDate: this.formatDate(item.createTime || Date.now())
        } as WorkItem;
      });

      // 把 cloud:// fileID 解析成临时 URL,否则 image 标签显示不了
      const fileIDs = rawWorks
        .map(w => w.coverFileId)
        .filter(id => !!id && id.startsWith('cloud://'));
      const urlMap: Record<string, string> = {};
      if (fileIDs.length > 0) {
        try {
          const urlRes = await wx.cloud.getTempFileURL({ fileList: fileIDs });
          urlRes.fileList.forEach((f: any) => {
            if (f.tempFileURL) urlMap[f.fileID] = f.tempFileURL;
          });
        } catch (e) {
          console.warn('封面临时链接获取失败', e);
        }
      }
      const works = rawWorks.map(w => ({
        ...w,
        coverUrl: urlMap[w.coverFileId] || (w.coverFileId && !w.coverFileId.startsWith('cloud://') ? w.coverFileId : '')
      }));

      this.setData({ works });
    } catch (error) {
      console.error('加载作品失败', error);
      wx.showToast({ title: '加载作品失败', icon: 'none' });
    } finally {
      this.setData({ isLoading: false });
    }
  },

  onInputChange(e: any) {
    const field = e.currentTarget.dataset.field;
    const value = e.detail.value;
    this.setData({ [`form.${field}`]: value });
  },

  onCategoryChange(e: any) {
    const index = Number(e.detail.value || 0);
    this.setData({ 'form.categoryIndex': index });
  },

  onPublishSwitch(e: any) {
    this.setData({ 'form.isPublished': !!e.detail.value });
  },

  async onChooseImage() {
    if (this.data.isUploading) return;
    try {
      const filePath = await new Promise<string>((resolve, reject) => {
        wx.chooseMedia({
          count: 1,
          mediaType: ['image'],
          sizeType: ['compressed'],
          success: (res) => resolve(res.tempFiles[0].tempFilePath),
          fail: reject
        });
      });
      // 立刻显示本地预览,再后台上传
      this.setData({ 'form.coverPreview': filePath, isUploading: true });
      const cloudPath = `images/works/${Date.now()}-${Math.floor(Math.random() * 1000)}.jpg`;
      const uploadRes = await wx.cloud.uploadFile({ cloudPath, filePath });
      this.setData({
        'form.coverFileId': uploadRes.fileID,
        isUploading: false
      });
      wx.showToast({ title: '上传成功', icon: 'success' });
    } catch (error) {
      this.setData({ isUploading: false });
      console.error('上传图片失败', error);
      wx.showToast({ title: '上传失败', icon: 'none' });
    }
  },

  onRemoveCover() {
    this.setData({ 'form.coverFileId': '', 'form.coverPreview': '' });
  },

  async onSubmit() {
    const { form, categories } = this.data as any;
    const categoryOption = categories[form.categoryIndex] || categories[0];
    if (!form.roleName) {
      wx.showToast({ title: '请输入角色名称', icon: 'none' });
      return;
    }
    if (!form.source) {
      wx.showToast({ title: '请输入来源', icon: 'none' });
      return;
    }
    if (!form.coverFileId) {
      wx.showToast({ title: '请上传封面图', icon: 'none' });
      return;
    }
    if (this.data.isUploading) {
      wx.showToast({ title: '封面还在上传中', icon: 'none' });
      return;
    }

    let priceNumber: number | null = null;
    if (form.price !== '' && form.price !== undefined) {
      priceNumber = Number(form.price);
      if (Number.isNaN(priceNumber) || priceNumber < 0) {
        wx.showToast({ title: '请输入有效价格', icon: 'none' });
        return;
      }
    }

    this.setData({ isSubmitting: true });
    try {
      const db = wx.cloud.database();
      await db.collection('works').add({
        data: {
          roleName: form.roleName,
          source: form.source,
          price: priceNumber,
          category: categoryOption.value,
          coverFileId: form.coverFileId,
          isPublished: form.isPublished,
          createTime: Date.now()
        }
      });
      wx.showToast({ title: '保存成功', icon: 'success' });
      this.resetForm();
      this.fetchWorks();
    } catch (error) {
      console.error('保存作品失败', error);
      wx.showToast({ title: '保存失败', icon: 'none' });
    } finally {
      this.setData({ isSubmitting: false });
    }
  },

  resetForm() {
    this.setData({
      form: {
        roleName: '',
        source: '',
        price: '',
        categoryIndex: 0,
        isPublished: true,
        coverFileId: '',
        coverPreview: ''
      }
    });
  },

  async onTogglePublish(e: any) {
    const id = e.currentTarget.dataset.id;
    const value = e.detail.value;
    if (!id) return;
    try {
      const db = wx.cloud.database();
      await db.collection('works').doc(id).update({ data: { isPublished: !!value } });
      this.fetchWorks();
    } catch (error) {
      console.error('更新上架状态失败', error);
      wx.showToast({ title: '更新失败', icon: 'none' });
    }
  },

  async onDelete(e: any) {
    const id = e.currentTarget.dataset.id;
    if (!id) return;
    const confirmRes = await new Promise<{ confirm: boolean }>((resolve) => {
      wx.showModal({
        title: '删除确认',
        content: '确定删除该作品吗？此操作不可撤销',
        confirmColor: '#ff4d4f',
        success: (res) => resolve({ confirm: res.confirm })
      });
    });
    if (!confirmRes.confirm) return;
    try {
      const db = wx.cloud.database();
      await db.collection('works').doc(id).remove();
      wx.showToast({ title: '已删除', icon: 'success' });
      this.fetchWorks();
    } catch (error) {
      console.error('删除作品失败', error);
      wx.showToast({ title: '删除失败', icon: 'none' });
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
