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
    filteredWorks: [] as WorkItem[],
    stats: { total: 0, published: 0, unpublished: 0 },
    statusFilters: [
      { label: '全部', value: 'all', count: 0 },
      { label: '已上架', value: 'published', count: 0 },
      { label: '未上架', value: 'unpublished', count: 0 }
    ] as { label: string; value: string; count: number }[],
    categoryFilters: [
      { label: '全部分类', value: 'all' },
      { label: '自设', value: 'original' },
      { label: '游戏', value: 'game' },
      { label: '动漫', value: 'anime' }
    ],
    statusFilter: 'all',
    categoryFilter: 'all',
    keyword: '',
    selectionMode: false,
    selectedIds: [] as string[],
    isSelected: {} as Record<string, boolean>,
    formPopupVisible: false,
    actionSheetVisible: false,
    actionItem: null as WorkItem | null,
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
      this.applyFilter();
    } catch (error) {
      console.error('加载作品失败', error);
      wx.showToast({ title: '加载作品失败', icon: 'none' });
    } finally {
      this.setData({ isLoading: false });
    }
  },

  applyFilter() {
    const { works, statusFilter, categoryFilter, keyword, selectedIds } = this.data;
    const kw = (keyword || '').trim().toLowerCase();
    const filteredWorks = works.filter(w => {
      if (statusFilter === 'published' && !w.isPublished) return false;
      if (statusFilter === 'unpublished' && w.isPublished) return false;
      if (categoryFilter !== 'all' && w.category !== categoryFilter) return false;
      if (kw) {
        const hay = `${w.roleName} ${w.source}`.toLowerCase();
        if (!hay.includes(kw)) return false;
      }
      return true;
    });

    const total = works.length;
    const published = works.filter(w => w.isPublished).length;
    const unpublished = total - published;

    const statusFilters = [
      { label: '全部', value: 'all', count: total },
      { label: '已上架', value: 'published', count: published },
      { label: '未上架', value: 'unpublished', count: unpublished }
    ];

    // 选中集合中已被筛选掉的项移除
    const filteredIds = new Set(filteredWorks.map(w => w._id!));
    const cleanedSelected = selectedIds.filter(id => filteredIds.has(id));
    const isSelected: Record<string, boolean> = {};
    cleanedSelected.forEach(id => { isSelected[id] = true; });

    this.setData({
      filteredWorks,
      stats: { total, published, unpublished },
      statusFilters,
      selectedIds: cleanedSelected,
      isSelected
    });
  },

  onSearchInput(e: any) {
    this.setData({ keyword: e.detail.value });
    this.applyFilter();
  },
  onClearKeyword() {
    this.setData({ keyword: '' });
    this.applyFilter();
  },
  onSetStatus(e: any) {
    this.setData({ statusFilter: e.currentTarget.dataset.value });
    this.applyFilter();
  },
  onSetCategory(e: any) {
    this.setData({ categoryFilter: e.currentTarget.dataset.value });
    this.applyFilter();
  },

  onToggleSelectionMode() {
    const next = !this.data.selectionMode;
    this.setData({
      selectionMode: next,
      selectedIds: [],
      isSelected: {}
    });
  },

  onRowTap(e: any) {
    const id = e.currentTarget.dataset.id;
    if (!id) return;
    if (this.data.selectionMode) {
      this.toggleSelect(id);
    }
  },

  onRowLongPress(e: any) {
    const id = e.currentTarget.dataset.id;
    if (!id) return;
    if (!this.data.selectionMode) {
      this.setData({ selectionMode: true });
    }
    this.toggleSelect(id);
  },

  toggleSelect(id: string) {
    const { selectedIds, isSelected } = this.data;
    let next: string[];
    if (isSelected[id]) {
      next = selectedIds.filter(x => x !== id);
    } else {
      next = [...selectedIds, id];
    }
    const map: Record<string, boolean> = {};
    next.forEach(x => { map[x] = true; });
    this.setData({ selectedIds: next, isSelected: map });
  },

  onSelectAllToggle() {
    const { filteredWorks, selectedIds } = this.data;
    if (selectedIds.length === filteredWorks.length && filteredWorks.length > 0) {
      this.setData({ selectedIds: [], isSelected: {} });
    } else {
      const ids = filteredWorks.map(w => w._id!);
      const map: Record<string, boolean> = {};
      ids.forEach(id => { map[id] = true; });
      this.setData({ selectedIds: ids, isSelected: map });
    }
  },

  onRowAction(e: any) {
    const id = e.currentTarget.dataset.id;
    const item = this.data.works.find(w => w._id === id) || null;
    this.setData({ actionItem: item, actionSheetVisible: true });
  },
  onActionSheetClose() {
    this.setData({ actionSheetVisible: false });
  },

  async onActionTogglePublish() {
    const item = this.data.actionItem;
    if (!item || !item._id) return;
    this.setData({ actionSheetVisible: false });
    try {
      const db = wx.cloud.database();
      await db.collection('works').doc(item._id).update({ data: { isPublished: !item.isPublished } });
      wx.showToast({ title: item.isPublished ? '已下架' : '已上架', icon: 'success' });
      this.fetchWorks();
    } catch (e) {
      console.error(e);
      wx.showToast({ title: '操作失败', icon: 'none' });
    }
  },

  async onActionDelete() {
    const item = this.data.actionItem;
    if (!item || !item._id) return;
    this.setData({ actionSheetVisible: false });
    const confirm = await this.confirm('删除确认', `确定删除「${item.roleName}」？此操作不可撤销`);
    if (!confirm) return;
    try {
      const db = wx.cloud.database();
      await db.collection('works').doc(item._id).remove();
      wx.showToast({ title: '已删除', icon: 'success' });
      this.fetchWorks();
    } catch (e) {
      console.error(e);
      wx.showToast({ title: '删除失败', icon: 'none' });
    }
  },

  async onBatchPublish(e: any) {
    const publish = e.currentTarget.dataset.publish === true || e.currentTarget.dataset.publish === 'true';
    const ids = this.data.selectedIds;
    if (ids.length === 0) return;
    const confirm = await this.confirm(
      publish ? '批量上架' : '批量下架',
      `确定将 ${ids.length} 件作品${publish ? '上架' : '下架'}？`
    );
    if (!confirm) return;
    wx.showLoading({ title: '处理中...', mask: true });
    try {
      const db = wx.cloud.database();
      await Promise.all(ids.map(id => db.collection('works').doc(id).update({ data: { isPublished: publish } })));
      wx.hideLoading();
      wx.showToast({ title: '操作完成', icon: 'success' });
      this.setData({ selectionMode: false, selectedIds: [], isSelected: {} });
      this.fetchWorks();
    } catch (e) {
      wx.hideLoading();
      console.error(e);
      wx.showToast({ title: '部分操作失败', icon: 'none' });
    }
  },

  async onBatchDelete() {
    const ids = this.data.selectedIds;
    if (ids.length === 0) return;
    const confirm = await this.confirm('批量删除', `确定删除 ${ids.length} 件作品？此操作不可撤销`);
    if (!confirm) return;
    wx.showLoading({ title: '删除中...', mask: true });
    try {
      const db = wx.cloud.database();
      await Promise.all(ids.map(id => db.collection('works').doc(id).remove()));
      wx.hideLoading();
      wx.showToast({ title: '已删除', icon: 'success' });
      this.setData({ selectionMode: false, selectedIds: [], isSelected: {} });
      this.fetchWorks();
    } catch (e) {
      wx.hideLoading();
      console.error(e);
      wx.showToast({ title: '部分删除失败', icon: 'none' });
    }
  },

  confirm(title: string, content: string): Promise<boolean> {
    return new Promise(resolve => {
      wx.showModal({
        title,
        content,
        confirmColor: '#E07458',
        success: res => resolve(!!res.confirm),
        fail: () => resolve(false)
      });
    });
  },

  // ===== 表单 =====
  onShowFormPopup() {
    this.setData({ formPopupVisible: true });
  },
  onFormPopupClose() {
    this.setData({ formPopupVisible: false });
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
      this.setData({ formPopupVisible: false });
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

  formatDate(ts: number) {
    const d = new Date(ts);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  }
});
