// admin.ts - 管理后台（支持千级订单：分页 + 批量 + 筛选）
interface OrderItem {
  _id: string;
  orderId?: string;
  tbOrderId: string;
  queueNumber?: string;
  customerName: string;
  roleName: string;
  status: string;
  progressStage: string;
  progressPercent: number;
  orderTime: string;
  deadline: string;
  stage: string;
  isUrgent?: boolean;
  isArchived?: boolean;
  createTime?: any;
}

interface OverviewStats {
  total: number;
  pending: number;
  processing: number;
  urgent: number;
  overdue: number;
  completed: number;
  archived: number;
}

const STAGE_OPTIONS = [
  { value: 'queued',   label: '已排单', percent: 10 },
  { value: 'modeling', label: '建模',   percent: 30 },
  { value: 'painting', label: '上妆',   percent: 55 },
  { value: 'hair',     label: '假毛',   percent: 80 },
  { value: 'shipped',  label: '已发货', percent: 100 }
];

const TABS = [
  { key: 'all',      label: '全部' },
  { key: 'pending',  label: '待审核' },
  { key: 'urgent',   label: '加急' },
  { key: 'overdue',  label: '逾期' },
  { key: 'queued',   label: '已排单' },
  { key: 'modeling', label: '建模' },
  { key: 'painting', label: '上妆' },
  { key: 'hair',     label: '假毛' },
  { key: 'shipped',  label: '已发货' }
];

Component({
  data: {
    // 列表数据
    orders: [] as OrderItem[],
    page: 1,
    pageSize: 20,
    total: 0,
    hasMore: false,
    isLoading: false,
    isLoadingMore: false,

    // 筛选
    currentTab: 'all',
    tabs: TABS,
    searchValue: '',
    sortBy: 'createTime',
    sortOrder: 'desc' as 'asc' | 'desc',
    dateFrom: '',
    dateTo: '',
    includeArchived: false,

    // 选择模式
    selectionMode: false,
    selectedIds: [] as string[],
    selectedSet: {} as Record<string, boolean>,

    // 概览
    stats: {
      total: 0, pending: 0, processing: 0, urgent: 0,
      overdue: 0, completed: 0, archived: 0
    } as OverviewStats,

    // 弹窗
    showFilterPopup: false,
    showSortPopup: false,
    showBatchStagePopup: false,
    showOrderForm: false,
    showAddSheet: false,

    // 阶段选项
    stageOptions: STAGE_OPTIONS,
    stageIndex: 0,
    batchStageIndex: 0,

    // 新增订单表单
    orderForm: {
      tbOrderId: '', queueNumber: '', customerName: '', roleName: '',
      orderTime: '', deadline: '',
      progressPercent: 10, progressStage: '已排单', stage: 'queued',
      isUrgent: false, previewImage: ''
    },
    todayDate: '',
    tempImagePath: '',
    uploadProgress: 0,
    isSubmitting: false,

    // 排序选项
    sortOptions: [
      { key: 'createTime-desc', label: '最新下单' },
      { key: 'createTime-asc',  label: '最早下单' },
      { key: 'deadline-asc',    label: '即将到期' },
      { key: 'progressPercent-desc', label: '进度靠后' },
      { key: 'progressPercent-asc',  label: '进度靠前' }
    ]
  },

  lifetimes: {
    attached() {
      const today = new Date();
      const ymd = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
      this.setData({ todayDate: ymd, 'orderForm.orderTime': ymd });
      this.refresh();
    }
  },

  pageLifetimes: {
    show() {
      // 从子页面返回后刷新
      if (this.data.orders.length > 0) {
        this.refresh(true);
      }
    }
  },

  methods: {
    // ============ 加载数据 ============
    async refresh(silent: boolean = false) {
      this.setData({ page: 1, orders: [], selectedIds: [], selectedSet: {} });
      await Promise.all([this.loadOrders(silent), this.loadStats()]);
    },

    async loadOrders(silent: boolean = false) {
      if (!silent) this.setData({ isLoading: true });

      const params = this.buildQueryParams();
      try {
        const res: any = await wx.cloud.callFunction({
          name: 'getOrders',
          data: params
        });
        const result = res.result || {};
        if (result.success === false) throw new Error(result.error || '加载失败');

        const list: OrderItem[] = result.data || [];
        this.setData({
          orders: list,
          total: result.total || 0,
          hasMore: !!result.hasMore,
          isLoading: false
        });
      } catch (err) {
        console.warn('云数据库加载失败，使用模拟数据', err);
        this.loadMockOrders();
        this.setData({ isLoading: false });
      }
    },

    async loadMore() {
      if (this.data.isLoadingMore || !this.data.hasMore) return;
      this.setData({ isLoadingMore: true, page: this.data.page + 1 });

      const params = this.buildQueryParams();
      try {
        const res: any = await wx.cloud.callFunction({
          name: 'getOrders',
          data: params
        });
        const result = res.result || {};
        const list: OrderItem[] = result.data || [];
        this.setData({
          orders: [...this.data.orders, ...list],
          hasMore: !!result.hasMore,
          isLoadingMore: false
        });
      } catch (err) {
        console.error('加载更多失败', err);
        this.setData({ isLoadingMore: false, page: this.data.page - 1 });
      }
    },

    async loadStats() {
      const queries = [
        { key: 'total',      params: { countOnly: true } },
        { key: 'pending',    params: { countOnly: true, status: 'pending' } },
        { key: 'urgent',     params: { countOnly: true, isUrgent: true } },
        { key: 'overdue',    params: { countOnly: true, overdueOnly: true } },
        { key: 'completed',  params: { countOnly: true, status: 'completed' } },
        { key: 'archived',   params: { countOnly: true, isArchived: true } }
      ];

      try {
        const results = await Promise.all(
          queries.map(q => wx.cloud.callFunction({ name: 'getOrders', data: q.params }))
        );
        const stats: any = { processing: 0 };
        results.forEach((r: any, i) => {
          stats[queries[i].key] = r.result?.total || 0;
        });
        stats.processing = Math.max(0, stats.total - stats.completed - stats.archived - stats.pending);
        this.setData({ stats });
      } catch (err) {
        // 静默失败，不影响主列表
        console.warn('概览统计加载失败', err);
      }
    },

    buildQueryParams() {
      const { currentTab, searchValue, sortBy, sortOrder, page, pageSize, dateFrom, dateTo, includeArchived } = this.data;
      const params: any = {
        page,
        pageSize,
        sortBy,
        sortOrder,
        isArchived: includeArchived ? undefined : false
      };

      if (searchValue) params.keyword = searchValue.trim();
      if (dateFrom) params.dateFrom = dateFrom;
      if (dateTo) params.dateTo = dateTo;

      // tab 映射
      if (currentTab === 'pending') params.status = 'pending';
      else if (currentTab === 'urgent') params.isUrgent = true;
      else if (currentTab === 'overdue') params.overdueOnly = true;
      else if (currentTab !== 'all') params.stage = currentTab;

      return params;
    },

    loadMockOrders() {
      const mock: OrderItem[] = [
        { _id: 'm1', tbOrderId: 'TB456789123', queueNumber: 'RS-2025-001', customerName: '张小华', roleName: '兔子头壳', status: 'soon', progressStage: '假毛', progressPercent: 80, orderTime: '2025-09-20', deadline: '2025-12-10', stage: 'hair', isUrgent: false },
        { _id: 'm2', tbOrderId: 'TB123456789', queueNumber: 'RS-2025-002', customerName: '王小明', roleName: '狐狸头壳', status: 'urgent', progressStage: '建模', progressPercent: 30, orderTime: '2025-10-15', deadline: '2025-12-30', stage: 'modeling', isUrgent: true },
        { _id: 'm3', tbOrderId: 'TB987654321', queueNumber: 'RS-2025-003', customerName: '李小红', roleName: '猫咪头壳', status: 'normal', progressStage: '建模', progressPercent: 30, orderTime: '2025-11-05', deadline: '2026-01-15', stage: 'modeling', isUrgent: false },
        { _id: 'm4', tbOrderId: 'TB789123456', queueNumber: 'RS-2025-004', customerName: '赵小刚', roleName: '熊猫头壳', status: 'normal', progressStage: '已排单', progressPercent: 10, orderTime: '2025-11-20', deadline: '2026-02-10', stage: 'queued', isUrgent: false },
        { _id: 'm5', tbOrderId: 'TB555000111', queueNumber: 'RS-2025-005', customerName: '钱小光', roleName: '柴犬头壳', status: 'normal', progressStage: '上妆', progressPercent: 55, orderTime: '2025-10-28', deadline: '2025-12-20', stage: 'painting', isUrgent: false },
        { _id: 'm6', tbOrderId: 'TB222333444', queueNumber: '', customerName: '孙小丽', roleName: '小狼头壳', status: 'pending', progressStage: '待审核', progressPercent: 0, orderTime: '2025-11-25', deadline: '2026-03-01', stage: 'queued', isUrgent: false }
      ];
      this.setData({
        orders: mock,
        total: mock.length,
        hasMore: false,
        stats: { total: 6, pending: 1, processing: 5, urgent: 1, overdue: 0, completed: 0, archived: 0 }
      });
    },

    // ============ 筛选 / 搜索 / 排序 ============
    onTabTap(e: any) {
      const tab = e.currentTarget.dataset.tab;
      if (tab === this.data.currentTab) return;
      this.setData({ currentTab: tab });
      this.refresh();
    },

    onStatCardTap(e: any) {
      const key = e.currentTarget.dataset.key;
      const map: Record<string, string> = {
        total: 'all', pending: 'pending', urgent: 'urgent',
        overdue: 'overdue', processing: 'all', completed: 'all', archived: 'all'
      };
      const tab = map[key] || 'all';
      const includeArchived = key === 'archived';
      this.setData({ currentTab: tab, includeArchived });
      this.refresh();
    },

    onSearchChange(e: any) {
      this.setData({ searchValue: e.detail.value });
    },

    onSearchSubmit() {
      this.refresh();
    },

    onSearchClear() {
      this.setData({ searchValue: '' });
      this.refresh();
    },

    openSortPopup() {
      this.setData({ showSortPopup: true });
    },

    closeSortPopup() {
      this.setData({ showSortPopup: false });
    },

    onSortPick(e: any) {
      const key = e.currentTarget.dataset.key;
      const [sortBy, sortOrder] = key.split('-');
      this.setData({ sortBy, sortOrder, showSortPopup: false });
      this.refresh();
    },

    openFilterPopup() {
      this.setData({ showFilterPopup: true });
    },

    closeFilterPopup() {
      this.setData({ showFilterPopup: false });
    },

    onFilterDateFromChange(e: any) {
      this.setData({ dateFrom: e.detail.value });
    },

    onFilterDateToChange(e: any) {
      this.setData({ dateTo: e.detail.value });
    },

    onToggleIncludeArchived() {
      this.setData({ includeArchived: !this.data.includeArchived });
    },

    onApplyFilter() {
      this.setData({ showFilterPopup: false });
      this.refresh();
    },

    onResetFilter() {
      this.setData({ dateFrom: '', dateTo: '', includeArchived: false, showFilterPopup: false });
      this.refresh();
    },

    // ============ 选择模式 / 批量 ============
    onToggleSelectionMode() {
      const next = !this.data.selectionMode;
      this.setData({
        selectionMode: next,
        selectedIds: [],
        selectedSet: {}
      });
    },

    onToggleSelect(e: any) {
      if (!this.data.selectionMode) return;
      const id = e.currentTarget.dataset.id;
      const set = { ...this.data.selectedSet };
      let ids = [...this.data.selectedIds];
      if (set[id]) {
        delete set[id];
        ids = ids.filter(x => x !== id);
      } else {
        set[id] = true;
        ids.push(id);
      }
      this.setData({ selectedIds: ids, selectedSet: set });
    },

    onSelectAllCurrent() {
      const allIds = this.data.orders.map(o => o._id);
      const allSelected = allIds.every(id => this.data.selectedSet[id]);
      if (allSelected) {
        this.setData({ selectedIds: [], selectedSet: {} });
      } else {
        const set: Record<string, boolean> = {};
        allIds.forEach(id => set[id] = true);
        this.setData({ selectedIds: allIds, selectedSet: set });
      }
    },

    onClearSelection() {
      this.setData({ selectedIds: [], selectedSet: {} });
    },

    async runBatch(action: string, payload?: any) {
      const ids = this.data.selectedIds;
      if (ids.length === 0) {
        wx.showToast({ title: '请先选择订单', icon: 'none' });
        return;
      }
      wx.showLoading({ title: '处理中...' });
      try {
        const res: any = await wx.cloud.callFunction({
          name: 'batchUpdateOrders',
          data: { orderIds: ids, action, payload }
        });
        wx.hideLoading();
        const r = res.result || {};
        if (r.success) {
          wx.showToast({
            title: `成功 ${r.succeeded}${r.failed ? ` / 失败 ${r.failed}` : ''}`,
            icon: 'none'
          });
          this.setData({ selectionMode: false, selectedIds: [], selectedSet: {} });
          this.refresh(true);
        } else {
          wx.showToast({ title: r.error || '操作失败', icon: 'none' });
        }
      } catch (err) {
        wx.hideLoading();
        wx.showToast({ title: '网络异常', icon: 'none' });
      }
    },

    onBatchAdvance() {
      wx.showModal({
        title: '批量推进',
        content: `将所选 ${this.data.selectedIds.length} 条订单推进到下一阶段?`,
        confirmColor: '#ff8800',
        success: (res) => {
          if (res.confirm) this.runBatch('advance-stage');
        }
      });
    },

    onBatchMarkUrgent() {
      this.runBatch('mark-urgent');
    },

    onBatchUnmarkUrgent() {
      this.runBatch('unmark-urgent');
    },

    onBatchArchive() {
      wx.showModal({
        title: '批量归档',
        content: `归档所选 ${this.data.selectedIds.length} 条订单?归档后将从主列表移除。`,
        confirmColor: '#ff8800',
        success: (res) => {
          if (res.confirm) this.runBatch('archive');
        }
      });
    },

    onBatchAssignQueue() {
      this.runBatch('assign-queue');
    },

    onBatchUnlock() {
      wx.showModal({
        title: '解锁订单',
        content: `解锁所选 ${this.data.selectedIds.length} 条订单，允许客户修改？`,
        confirmColor: '#ff8800',
        success: (res) => {
          if (res.confirm) this.runBatch('unlock');
        }
      });
    },

    onBatchSetStageOpen() {
      this.setData({ showBatchStagePopup: true });
    },

    onBatchSetStageClose() {
      this.setData({ showBatchStagePopup: false });
    },

    onBatchSetStagePick(e: any) {
      const value = e.currentTarget.dataset.value;
      this.setData({ showBatchStagePopup: false });
      this.runBatch('set-stage', { stage: value });
    },

    // ============ 触底加载 ============
    onScrollToLower() {
      this.loadMore();
    },

    // ============ 单个订单点击 ============
    onOrderTap(e: any) {
      if (this.data.selectionMode) {
        this.onToggleSelect(e);
        return;
      }
      const tbOrderId = e.currentTarget.dataset.tbId;
      wx.navigateTo({
        url: `/pages/order-detail/order-detail?id=${tbOrderId}&admin=true`
      });
    },

    onOrderLongPress(e: any) {
      if (!this.data.selectionMode) {
        const id = e.currentTarget.dataset.id;
        const set: Record<string, boolean> = { [id]: true };
        this.setData({ selectionMode: true, selectedIds: [id], selectedSet: set });
      }
    },

    // ============ 操作菜单 ============
    onAddOrder() {
      this.setData({
        showOrderForm: true,
        showAddSheet: false,
        orderForm: {
          tbOrderId: '', queueNumber: '', customerName: '', roleName: '',
          orderTime: this.data.todayDate, deadline: '',
          progressPercent: 10, progressStage: '已排单', stage: 'queued',
          isUrgent: false, previewImage: ''
        },
        tempImagePath: '', uploadProgress: 0, stageIndex: 0
      });
    },

    onCloseOrderForm() {
      this.setData({ showOrderForm: false });
    },

    onFormInputChange(e: any) {
      const { field } = e.currentTarget.dataset;
      this.setData({ [`orderForm.${field}`]: e.detail.value });
    },

    onToggleUrgent() {
      this.setData({ 'orderForm.isUrgent': !this.data.orderForm.isUrgent });
    },

    onDateChange(e: any) {
      const { field } = e.currentTarget.dataset;
      this.setData({ [`orderForm.${field}`]: e.detail.value });
    },

    onStageChange(e: any) {
      const idx = parseInt(e.detail.value);
      const opt = this.data.stageOptions[idx];
      this.setData({
        stageIndex: idx,
        'orderForm.stage': opt.value,
        'orderForm.progressStage': opt.label,
        'orderForm.progressPercent': opt.percent
      });
    },

    onChooseImage() {
      wx.chooseMedia({
        count: 1,
        mediaType: ['image'],
        sizeType: ['compressed'],
        success: (res) => {
          this.setData({ tempImagePath: res.tempFiles[0].tempFilePath });
        }
      });
    },

    uploadImage(filePath: string): Promise<any> {
      return new Promise((resolve, reject) => {
        const ext = filePath.match(/\.(\w+)$/)?.[1] || 'png';
        const cloudPath = `images/orders/${Date.now()}_${Math.random().toString(36).slice(-6)}.${ext}`;
        const task = wx.cloud.uploadFile({ cloudPath, filePath, success: resolve, fail: reject });
        task.onProgressUpdate((res) => this.setData({ uploadProgress: res.progress }));
      });
    },

    async onSubmitOrderForm() {
      const { orderForm, tempImagePath } = this.data;
      const required = ['tbOrderId', 'customerName', 'roleName', 'orderTime', 'deadline'];
      const labels: any = { tbOrderId: '淘宝订单号', customerName: '客户名称', roleName: '角色名称', orderTime: '下单时间', deadline: '预期完成时间' };
      for (const f of required) {
        if (!(orderForm as any)[f]) {
          wx.showToast({ title: `请填写${labels[f]}`, icon: 'none' });
          return;
        }
      }

      this.setData({ isSubmitting: true });
      try {
        let previewImage = '';
        if (tempImagePath) {
          const r = await this.uploadImage(tempImagePath);
          previewImage = r.fileID;
        }
        await wx.cloud.callFunction({
          name: 'createOrder',
          data: {
            ...orderForm,
            previewImage,
            status: orderForm.isUrgent ? 'urgent' : 'normal',
            createTime: new Date()
          }
        });
        wx.showToast({ title: '创建成功', icon: 'success' });
        this.setData({ showOrderForm: false, isSubmitting: false });
        this.refresh();
      } catch (err) {
        wx.showToast({ title: '创建失败', icon: 'none' });
        this.setData({ isSubmitting: false });
      }
    },

    onOpenAddSheet() {
      this.setData({ showAddSheet: true });
    },

    onCloseAddSheet() {
      this.setData({ showAddSheet: false });
    },

    onOrderReview() {
      wx.navigateTo({ url: '/pages/admin/order-review/order-review' });
    },

    onManageWorks() {
      wx.navigateTo({ url: '/pages/admin/works-manage/works-manage' });
    },

    onExport() {
      wx.showToast({ title: '导出功能开发中', icon: 'none' });
    }
  }
});
