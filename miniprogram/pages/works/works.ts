// works.ts
interface WorkItem {
  id: string;
  roleName: string;
  source: string;
  date: string;
  imageUrl?: string;
  category: 'original' | 'game' | 'anime';
  categoryLabel?: string;
  createTime?: number;
  coverFileId?: string;
}

const CATEGORY_LABEL: Record<string, string> = {
  original: '自设',
  game: '游戏',
  anime: '动漫'
};

Page({
  data: {
    searchValue: '',
    currentSort: 'latest',
    works: [] as WorkItem[],
    filteredWorks: [] as WorkItem[],
    displayedWorks: [] as WorkItem[],
    hasMore: true,
    pageSize: 6,
    currentPage: 1,
    isLoading: true,
    loadFailed: false,
    skeletonItems: [1, 2, 3, 4]
  },

  onLoad() {
    this.loadWorks();
  },

  // 加载作品数据（优先云端，失败回落本地模拟）
  async loadWorks() {
    this.setData({ isLoading: true, loadFailed: false });
    try {
      const db = wx.cloud.database();
      const res = await db.collection('works')
        .where({ isPublished: true })
        .orderBy('createTime', 'desc')
        .get();

      if (res.data && res.data.length > 0) {
        // 收集所有fileID
        const fileIDs = res.data
          .map((item: any) => item.coverFileId || item.imageFileId)
          .filter(Boolean);
        
        // 批量获取临时链接
        let fileURLMap: Record<string, string> = {};
        if (fileIDs.length > 0) {
          try {
            const urlRes = await wx.cloud.getTempFileURL({ fileList: fileIDs });
            urlRes.fileList.forEach((file: any) => {
              if (file.tempFileURL) {
                fileURLMap[file.fileID] = file.tempFileURL;
              }
            });
          } catch (error) {
            console.error('获取临时URL失败', error);
          }
        }

        const works: WorkItem[] = res.data.map((item: any) => {
          const fileId = item.coverFileId || item.imageFileId || '';
          const category = item.category || 'original';
          return {
            id: item._id,
            roleName: item.roleName || item.title || '未命名角色',
            source: item.source || item.description || '作品',
            date: this.formatDate(item.createTime || Date.now()),
            imageUrl: fileURLMap[fileId] || '',
            coverFileId: fileId,
            category,
            categoryLabel: CATEGORY_LABEL[category] || '作品',
            createTime: item.createTime || Date.now()
          } as any;
        });
        this.setData({ works, isLoading: false });
        this.applyFilters(true);
        return;
      }
      this.setData({ works: [], isLoading: false });
      this.applyFilters(true);
      return;
    } catch (error) {
      console.error('加载作品失败', error);
      this.setData({ works: [], isLoading: false, loadFailed: true });
      this.applyFilters(true);
    }
  },

  onRetryLoad() {
    this.loadWorks();
  },

  // 搜索框内容变化
  onSearchChange(e: any) {
    this.setData({
      searchValue: e.detail.value
    });
  },

  // 提交搜索
  onSearch() {
    this.applyFilters();
  },

  // 排序方式变化
  onSortChange(e: any) {
    const sort = e.currentTarget.dataset.sort;
    this.setData({
      currentSort: sort,
      currentPage: 1
    });
    this.applyFilters(true);
  },

  // 应用筛选和排序
  applyFilters(resetPage = false) {
    const { searchValue, currentSort, works, pageSize, currentPage } = this.data as any;
    const page = resetPage ? 1 : currentPage;
    let filtered = [...works];

    // 应用搜索筛选
    if (searchValue) {
      const keyword = searchValue.toLowerCase();
      filtered = filtered.filter(work =>
        (work.roleName || '').toLowerCase().includes(keyword) ||
        (work.source || '').toLowerCase().includes(keyword)
      );
    }

    // 应用分类筛选
    if (['original', 'game', 'anime'].includes(currentSort)) {
      filtered = filtered.filter(work => work.category === currentSort);
    }

    // 应用排序
    if (currentSort === 'latest') {
      filtered.sort((a, b) => (b.createTime || 0) - (a.createTime || 0));
    } else if (currentSort === 'name') {
      filtered.sort((a, b) => a.roleName.localeCompare(b.roleName));
    }

    const visible = filtered.slice(0, pageSize * page);
    this.setData({
      filteredWorks: filtered,
      displayedWorks: visible,
      currentPage: page,
      hasMore: filtered.length > visible.length
    }, () => this.measureSourceOverflow());
  },

  // 加载更多
  loadMore() {
    this.setData({
      currentPage: this.data.currentPage + 1
    });

    const { filteredWorks, pageSize, currentPage } = this.data as any;
    const visible = filteredWorks.slice(0, pageSize * currentPage);
    this.setData({
      displayedWorks: visible,
      hasMore: filteredWorks.length > visible.length
    }, () => this.measureSourceOverflow());
  },

  // 测量每条「来源」是否溢出，溢出则注入折返滚动距离与时长
  measureSourceOverflow() {
    const list = (this.data as any).displayedWorks || [];
    if (!list.length) return;
    const query = wx.createSelectorQuery().in(this);
    list.forEach((_w: any, idx: number) => {
      query.select(`#src-wrap-${idx}`).boundingClientRect();
      query.select(`#src-text-${idx}`).boundingClientRect();
    });
    query.exec((res: any[]) => {
      if (!res || !res.length) return;
      const updates: Record<string, any> = {};
      for (let i = 0; i < list.length; i++) {
        const wrap = res[i * 2];
        const text = res[i * 2 + 1];
        if (!wrap || !text) continue;
        const overflow = text.width - wrap.width;
        if (overflow > 2) {
          updates[`displayedWorks[${i}].needScroll`] = true;
          updates[`displayedWorks[${i}].scrollDistance`] = -Math.round(overflow);
          // 速度约 30px/s，再加 2s 端点停顿基底
          updates[`displayedWorks[${i}].scrollDuration`] = Math.max(4, Math.round(overflow / 30) + 2);
        } else if ((list[i] as any).needScroll) {
          updates[`displayedWorks[${i}].needScroll`] = false;
        }
      }
      if (Object.keys(updates).length) this.setData(updates);
    });
  },

  // 日期格式化
  formatDate(timestamp: number) {
    const d = new Date(timestamp);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  },

  // 点击作品项
  onWorkClick(e: any) {
    const workId = e.currentTarget.dataset.workId;
    wx.navigateTo({
      url: `/pages/works/detail/detail?id=${workId}`
    });
  }
}) 
