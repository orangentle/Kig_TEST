// works.ts
interface WorkItem {
  id: string;
  title: string;
  source: string;
  date: string;
  imageUrl?: string;
  category: 'original' | 'game' | 'anime';
}

Page({
  data: {
    searchValue: '',
    currentSort: 'latest',
    works: [] as WorkItem[],
    filteredWorks: [] as WorkItem[],
    hasMore: true,
    pageSize: 6,
    currentPage: 1
  },

  onLoad() {
    this.loadWorks();
  },

  // 加载作品数据
  loadWorks() {
    // 模拟数据，实际应从服务器获取
    const mockData: WorkItem[] = [
      {
        id: '1',
        title: '狐狸头壳',
        source: '原创设计',
        date: '2025-11',
        category: 'original'
      },
      {
        id: '2',
        title: '猫咪头壳',
        source: '原创设计',
        date: '2025-10',
        category: 'original'
      },
      {
        id: '3',
        title: '兔子头壳',
        source: '原创设计',
        date: '2025-09',
        category: 'original'
      },
      {
        id: '4',
        title: '熊猫头壳',
        source: '游戏角色',
        date: '2025-08',
        category: 'game'
      },
      {
        id: '5',
        title: '狼头壳',
        source: '动漫角色',
        date: '2025-07',
        category: 'anime'
      },
      {
        id: '6',
        title: '龙头壳',
        source: '游戏角色',
        date: '2025-06',
        category: 'game'
      }
    ];
    
    this.setData({
      works: mockData,
      filteredWorks: mockData
    });
    
    this.applyFilters();
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
    
    this.applyFilters();
  },

  // 应用筛选和排序
  applyFilters() {
    const { searchValue, currentSort, works } = this.data;
    let filtered = [...works];
    
    // 应用搜索筛选
    if (searchValue) {
      const keyword = searchValue.toLowerCase();
      filtered = filtered.filter(work => 
        work.title.toLowerCase().includes(keyword) || 
        work.source.toLowerCase().includes(keyword)
      );
    }
    
    // 应用分类筛选
    if (['original', 'game', 'anime'].includes(currentSort)) {
      filtered = filtered.filter(work => work.category === currentSort);
    }
    
    // 应用排序
    if (currentSort === 'latest') {
      filtered.sort((a, b) => a.date > b.date ? -1 : 1);
    } else if (currentSort === 'name') {
      filtered.sort((a, b) => a.title.localeCompare(b.title));
    }
    
    // 更新数据
    this.setData({
      filteredWorks: filtered,
      hasMore: filtered.length > this.data.pageSize * this.data.currentPage
    });
  },

  // 加载更多
  loadMore() {
    this.setData({
      currentPage: this.data.currentPage + 1
    });
    
    // 检查是否还有更多数据
    const { filteredWorks, pageSize, currentPage } = this.data;
    this.setData({
      hasMore: filteredWorks.length > pageSize * currentPage
    });
  },

  // 点击作品项
  onWorkClick(e: any) {
    const workId = e.currentTarget.dataset.workId;
    
    wx.showToast({
      title: '作品详情功能开发中',
      icon: 'none'
    });
  }
}) 