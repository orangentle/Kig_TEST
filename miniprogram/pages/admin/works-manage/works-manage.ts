interface WorkItem {
  _id?: string;
  roleName: string;
  source: string;
  price?: number;
  category: 'original' | 'game' | 'anime';
  coverFileId: string;
  isPublished: boolean;
  createTime: number;
  displayDate?: string;
}

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
    isSubmitting: false
  },

  onLoad() {
    this.fetchWorks();
  },

  async fetchWorks() {
    this.setData({ isLoading: true });
    try {
      const db = wx.cloud.database();
      const res = await db.collection('works').orderBy('createTime', 'desc').get();
      const works = (res.data || []).map((item: any) => ({
        _id: item._id,
        roleName: item.roleName || item.title || '未命名角色',
        source: item.source || item.description || '作品',
        price: item.price,
        category: item.category || 'original',
        coverFileId: item.coverFileId || item.imageFileId || '',
        isPublished: item.isPublished !== false,
        createTime: item.createTime || Date.now(),
        displayDate: this.formatDate(item.createTime || Date.now())
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
    try {
      const filePath = await new Promise<string>((resolve, reject) => {
        wx.chooseImage({
          count: 1,
          sizeType: ['compressed'],
          success: (res) => resolve(res.tempFilePaths[0]),
          fail: reject
        });
      });
      const cloudPath = `works/${Date.now()}-${Math.floor(Math.random() * 1000)}.jpg`;
      wx.showLoading({ title: '上传中...' });
      const uploadRes = await wx.cloud.uploadFile({ cloudPath, filePath });
      wx.hideLoading();
      this.setData({
        'form.coverFileId': uploadRes.fileID,
        'form.coverPreview': filePath
      });
      wx.showToast({ title: '上传成功', icon: 'success' });
    } catch (error) {
      wx.hideLoading();
      console.error('上传图片失败', error);
      wx.showToast({ title: '上传失败', icon: 'none' });
    }
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
        content: '确定删除该作品吗？',
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
