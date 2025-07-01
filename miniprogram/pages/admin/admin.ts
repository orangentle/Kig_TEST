// admin.ts
interface OrderItem {
  orderId: string;
  customerName: string;
  roleName: string;
  status: 'urgent' | 'normal' | 'soon';
  progressStage: string;
  progressPercent: number;
  orderTime: string;
  deadline: string;
  stage: 'design' | 'model' | 'print' | 'polish' | 'assembly' | 'quality' | 'shipping';
}

Component({
  data: {
    searchValue: '',
    currentTab: 'all',
    orders: [] as OrderItem[],
    filteredOrders: [] as OrderItem[],
    orderCounts: {
      all: 0,
      design: 0,
      model: 0,
      print: 0,
      polish: 0,
      assembly: 0
    }
  },

  lifetimes: {
    attached() {
      this.loadOrders();
    }
  },

  methods: {
    // 加载订单数据
    loadOrders() {
      // 模拟数据，实际应从服务器获取
      const mockData: OrderItem[] = [
        {
          orderId: 'TB456789123',
          customerName: '张小华',
          roleName: '兔子头壳',
          status: 'soon',
          progressStage: '质检',
          progressPercent: 90,
          orderTime: '2025-09-20',
          deadline: '2025-12-10',
          stage: 'quality'
        },
        {
          orderId: 'TB123456789',
          customerName: '王小明',
          roleName: '狐狸头壳',
          status: 'urgent',
          progressStage: '打印中',
          progressPercent: 50,
          orderTime: '2025-10-15',
          deadline: '2025-12-30',
          stage: 'print'
        },
        {
          orderId: 'TB987654321',
          customerName: '李小红',
          roleName: '猫咪头壳',
          status: 'normal',
          progressStage: '模型制作',
          progressPercent: 30,
          orderTime: '2025-11-05',
          deadline: '2025-01-15',
          stage: 'model'
        },
        {
          orderId: 'TB789123456',
          customerName: '赵小刚',
          roleName: '熊猫头壳',
          status: 'normal',
          progressStage: '设计图确认',
          progressPercent: 20,
          orderTime: '2025-11-20',
          deadline: '2025-02-10',
          stage: 'design'
        }
      ];
      
      // 按照下单时间排序，最早的在顶部
      mockData.sort((a, b) => {
        return new Date(a.orderTime).getTime() - new Date(b.orderTime).getTime();
      });
      
      this.setData({
        orders: mockData
      });
      
      this.updateOrderCounts();
      this.applyFilters();
    },

    // 更新订单数量统计
    updateOrderCounts() {
      const { orders } = this.data;
      const counts = {
        all: orders.length,
        design: orders.filter(o => o.stage === 'design').length,
        model: orders.filter(o => o.stage === 'model').length,
        print: orders.filter(o => o.stage === 'print').length,
        polish: orders.filter(o => o.stage === 'polish').length,
        assembly: orders.filter(o => o.stage === 'assembly').length
      };
      
      this.setData({
        orderCounts: counts
      });
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

    // 标签切换
    onTabChange(e: any) {
      const tab = e.currentTarget.dataset.tab;
      
      this.setData({
        currentTab: tab
      });
      
      this.applyFilters();
    },

    // 应用筛选
    applyFilters() {
      const { searchValue, currentTab, orders } = this.data;
      let filtered = [...orders];
      
      // 应用标签筛选
      if (currentTab !== 'all') {
        filtered = filtered.filter(order => order.stage === currentTab);
      }
      
      // 应用搜索筛选
      if (searchValue) {
        const keyword = searchValue.toLowerCase();
        filtered = filtered.filter(order => 
          order.orderId.toLowerCase().includes(keyword) || 
          order.customerName.toLowerCase().includes(keyword) ||
          order.roleName.toLowerCase().includes(keyword)
        );
      }
      
      this.setData({
        filteredOrders: filtered
      });
    },

    // 点击订单
    onOrderClick(e: any) {
      const orderId = e.currentTarget.dataset.orderId;
      
      wx.navigateTo({
        url: `/pages/order-detail/order-detail?id=${orderId}&admin=true`
      });
    },

    // 新增订单
    onAddOrder() {
      wx.showToast({
        title: '新增订单功能开发中',
        icon: 'none'
      });
    },

    // 导出数据
    onExportData() {
      wx.showToast({
        title: '导出数据功能开发中',
        icon: 'none'
      });
    },

    // 查看归档
    onViewArchive() {
      wx.showToast({
        title: '查看归档功能开发中',
        icon: 'none'
      });
    }
  }
}) 