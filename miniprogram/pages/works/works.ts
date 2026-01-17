// works.ts
interface WorkItem {
  id: string;
  roleName: string;
  source: string;
  date: string;
  imageUrl?: string;
  category: 'original' | 'game' | 'anime';
  createTime?: number;
  coverFileId?: string;
}

Page({
  data: {
    searchValue: '',
    currentSort: 'latest',
    works: [] as WorkItem[],
    filteredWorks: [] as WorkItem[],
    displayedWorks: [] as WorkItem[],
    hasMore: true,
    pageSize: 6,
    currentPage: 1
  },

  onLoad() {
    this.loadWorks();
  },

  // 加载作品数据（优先云端，失败回落本地模拟）
  async loadWorks() {
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
          return {
            id: item._id,
            roleName: item.roleName || item.title || '未命名角色',
            source: item.source || item.description || '作品',
            date: this.formatDate(item.createTime || Date.now()),
            imageUrl: fileURLMap[fileId] || '',
            coverFileId: fileId,
            category: item.category || 'original',
            createTime: item.createTime || Date.now()
          };
        });
        this.setData({ works });
        this.applyFilters(true);
        return;
      }
    } catch (error) {
      console.error('加载作品失败，使用本地数据', error);
    }

    // 无数据则保持空列表
    this.setData({ works: [] });
    this.applyFilters(true);
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
    });
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
