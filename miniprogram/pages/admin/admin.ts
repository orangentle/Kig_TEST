// admin.ts
interface OrderItem {
  orderId: string;
  tbOrderId: string;  // 添加淘宝订单号字段
  customerName: string;
  roleName: string;
  status: 'urgent' | 'normal' | 'soon';
  progressStage: string;
  progressPercent: number;
  orderTime: string;
  deadline: string;
  stage: 'design' | 'model' | 'print' | 'polish' | 'assembly' | 'quality' | 'shipping';
}

// 新增订单表单数据接口
interface OrderForm {
  tbOrderId: string;  // 淘宝订单号
  queueNumber: string; // 排单号
  customerName: string; // 客户名称
  roleName: string; // 角色名称
  orderTime: string; // 下单时间
  deadline: string; // 预期完成时间
  progressPercent: number; // 制作进度
  progressStage: string; // 进度阶段描述
  stage: string; // 制作阶段
  isUrgent: boolean; // 是否加急
  previewImage: string; // 预期成品展示图
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
    },
    // 新增订单表单相关数据
    showOrderForm: false,
    orderForm: {
      tbOrderId: '',
      queueNumber: '',
      customerName: '',
      roleName: '',
      orderTime: '',
      deadline: '',
      progressPercent: 0,
      progressStage: '订单确认',
      stage: 'confirm',
      isUrgent: false,
      previewImage: ''
    } as OrderForm,
    // 制作阶段选项
    stageOptions: [
      { label: '订单确认', value: 'confirm', percent: 10 },
      { label: '设计图确认', value: 'design', percent: 20 },
      { label: '模型制作', value: 'model', percent: 30 },
      { label: '打印中', value: 'print', percent: 50 },
      { label: '打磨上色', value: 'polish', percent: 70 },
      { label: '组装', value: 'assembly', percent: 80 },
      { label: '质检', value: 'quality', percent: 90 },
      { label: '发货', value: 'shipping', percent: 100 }
    ],
    stageIndex: 0, // 当前选中的制作阶段索引
    todayDate: '',
    tempImagePath: '',
    uploadProgress: 0,
    isSubmitting: false
  },

  lifetimes: {
    attached() {
      this.loadOrders();
      // 设置今天日期作为默认下单时间
      const today = new Date();
      const year = today.getFullYear();
      const month = String(today.getMonth() + 1).padStart(2, '0');
      const day = String(today.getDate()).padStart(2, '0');
      this.setData({
        todayDate: `${year}-${month}-${day}`,
        'orderForm.orderTime': `${year}-${month}-${day}`
      });
      
      // 设置初始阶段索引
      this.updateStageIndex();
    }
  },

  methods: {
    // 加载订单数据
    loadOrders() {
      // 从云数据库获取订单数据
      wx.cloud.callFunction({
        name: 'getOrders',
        success: (res: any) => {
          const orders = res.result.data || [];
          
          // 按照下单时间排序，最早的在顶部
          orders.sort((a: OrderItem, b: OrderItem) => {
            return new Date(a.orderTime).getTime() - new Date(b.orderTime).getTime();
          });
          
          this.setData({
            orders: orders
          });
          
          this.updateOrderCounts();
          this.applyFilters();
        },
        fail: (err: any) => {
          console.error('获取订单失败', err);
          wx.showToast({
            title: '获取订单失败',
            icon: 'none'
          });
          
          // 加载失败时使用模拟数据
          this.loadMockOrders();
        }
      });
    },
    
    // 加载模拟订单数据
    loadMockOrders() {
      // 模拟数据，实际应从服务器获取
      const mockData: OrderItem[] = [
        {
          orderId: 'order-001',
          tbOrderId: 'TB456789123',
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
          orderId: 'order-002',
          tbOrderId: 'TB123456789',
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
          orderId: 'order-003',
          tbOrderId: 'TB987654321',
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
          orderId: 'order-004',
          tbOrderId: 'TB789123456',
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
          order.tbOrderId.toLowerCase().includes(keyword) || 
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

    // 更新阶段索引
    updateStageIndex() {
      const { stageOptions } = this.data;
      const stageValue = this.data.orderForm.stage;
      let index = 0;
      
      for (let i = 0; i < stageOptions.length; i++) {
        if (stageOptions[i].value === stageValue) {
          index = i;
          break;
        }
      }
      
      this.setData({
        stageIndex: index
      });
    },
    
    // 新增订单
    onAddOrder() {
      this.setData({
        showOrderForm: true,
        orderForm: {
          tbOrderId: '',
          queueNumber: '',
          customerName: '',
          roleName: '',
          orderTime: this.data.todayDate,
          deadline: '',
          progressPercent: 10,
          progressStage: '订单确认',
          stage: 'confirm',
          isUrgent: false,
          previewImage: ''
        }
      }, () => {
        // 更新阶段索引
        this.updateStageIndex();
      });
      
      this.setData({
        tempImagePath: '',
        uploadProgress: 0
      });
    },
    
    // 关闭订单表单
    onCloseOrderForm() {
      this.setData({
        showOrderForm: false
      });
    },
    
    // 表单输入变化
    onFormInputChange(e: any) {
      const { field } = e.currentTarget.dataset;
      const { value } = e.detail;
      
      this.setData({
        [`orderForm.${field}`]: value
      });
    },
    
    // 切换是否加急
    onToggleUrgent() {
      this.setData({
        'orderForm.isUrgent': !this.data.orderForm.isUrgent
      });
    },
    
    // 选择日期
    onDateChange(e: any) {
      const { field } = e.currentTarget.dataset;
      const { value } = e.detail;
      
      this.setData({
        [`orderForm.${field}`]: value
      });
    },
    
    // 选择图片
    onChooseImage() {
      wx.chooseImage({
        count: 1,
        sizeType: ['compressed'],
        sourceType: ['album', 'camera'],
        success: (res) => {
          this.setData({
            tempImagePath: res.tempFilePaths[0]
          });
        }
      });
    },
    
    // 选择制作阶段
    onStageChange(e: any) {
      const { value } = e.detail;
      const index = parseInt(value);
      const stageOption = this.data.stageOptions[index];
      
      if (stageOption) {
        this.setData({
          'orderForm.stage': stageOption.value,
          'orderForm.progressStage': stageOption.label,
          'orderForm.progressPercent': stageOption.percent,
          stageIndex: index
        });
      }
    },
    
    // 提交表单
    async onSubmitOrderForm() {
      const { orderForm, tempImagePath } = this.data;
      
      // 表单验证
      if (!orderForm.tbOrderId) {
        wx.showToast({
          title: '请输入淘宝订单号',
          icon: 'none'
        });
        return;
      }
      
      if (!orderForm.customerName) {
        wx.showToast({
          title: '请输入客户名称',
          icon: 'none'
        });
        return;
      }
      
      if (!orderForm.roleName) {
        wx.showToast({
          title: '请输入角色名称',
          icon: 'none'
        });
        return;
      }
      
      if (!orderForm.orderTime) {
        wx.showToast({
          title: '请选择下单时间',
          icon: 'none'
        });
        return;
      }
      
      if (!orderForm.deadline) {
        wx.showToast({
          title: '请选择预期完成时间',
          icon: 'none'
        });
        return;
      }
      
      this.setData({
        isSubmitting: true
      });
      
      try {
        let previewImageUrl = '';
        
        // 如果有选择图片，先上传图片
        if (tempImagePath) {
          const uploadRes = await this.uploadImage(tempImagePath);
          previewImageUrl = uploadRes.fileID;
        }
        
        // 创建新订单
        const orderData = {
          ...orderForm,
          previewImage: previewImageUrl,
          status: orderForm.isUrgent ? 'urgent' : 'normal',
          createTime: new Date()
        };
        
        await wx.cloud.callFunction({
          name: 'createOrder',
          data: orderData
        });
        
        wx.showToast({
          title: '创建订单成功',
          icon: 'success'
        });
        
        // 关闭表单并刷新数据
        this.setData({
          showOrderForm: false,
          isSubmitting: false
        });
        
        this.loadOrders();
      } catch (error) {
        console.error('创建订单失败', error);
        wx.showToast({
          title: '创建订单失败',
          icon: 'none'
        });
        this.setData({
          isSubmitting: false
        });
      }
    },
    
    // 上传图片到云存储
    uploadImage(filePath: string): Promise<any> {
      return new Promise((resolve, reject) => {
        const cloudPath = `images/orders/${new Date().getTime()}_${Math.random().toString(36).slice(-6)}.${filePath.match(/\.(\w+)$/)?.[1] || 'png'}`;
        
        const uploadTask = wx.cloud.uploadFile({
          cloudPath,
          filePath,
          success: (res) => {
            resolve(res);
          },
          fail: (err) => {
            reject(err);
          }
        });
        
        uploadTask.onProgressUpdate((res) => {
          this.setData({
            uploadProgress: res.progress
          });
        });
      });
    },

    // 导出数据
    onExportData() {
      wx.showToast({
        title: '导出数据功能开发中',
        icon: 'none'
      });
    },

    // 订单审核
    onOrderReview() {
      wx.navigateTo({
        url: '/pages/admin/order-review/order-review'
      });
    },

    // 作品管理
    onManageWorks() {
      wx.navigateTo({
        url: '/pages/admin/works-manage/works-manage'
      });
    }
  }
}) 