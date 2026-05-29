// admin.ts - 管理后台（支持千级订单：分页 + 批量 + 筛选）
import type { Order, StageDef } from '../../types/order';
import { STAGE_FLOW } from '../../types/order';

type OrderItem = Order;

interface OverviewStats {
  total: number;
  pending: number;
  processing: number;
  urgent: number;
  overdue: number;
  completed: number;
  archived: number;
}

const STAGE_OPTIONS: StageDef[] = STAGE_FLOW;

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

    // 新增订单表单（与 Order 数据结构对齐）
    orderForm: {
      // 基本
      tbOrderId: '', queueNumber: '', customerName: '', roleName: '', ip: '',
      // 时间
      orderTime: '', deadline: '',
      // 进度
      progressPercent: 10, progressStage: '已排单', stage: 'queued',
      // 状态
      isUrgent: false,
      // 联系
      taobaoName: '', qq: '', phone: '',
      // 身材
      height: 0, weight: 0, headCircumference: 0, shoulderWidth: 0,
      // 选项
      needAccessory: false, needReplaceFace: false, replaceFaceCount: 1,
      // 备注
      remark: ''
    },
    todayDate: '',
    referenceImages: [] as string[],   // 角色参考图本地路径（最多 3）
    replaceFaceImages: [] as string[], // 替换脸图片本地路径
    faceCountOptions: [1, 2, 3],
    faceCountIndex: 0,
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
        console.error('订单加载失败', err);
        this.setData({ isLoading: false, orders: [], total: 0, hasMore: false });
        wx.showToast({ title: '加载失败，请重试', icon: 'none' });
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

    onBatchApprove() {
      wx.showModal({
        title: '审核通过',
        content: `确认通过所选 ${this.data.selectedIds.length} 条订单的审核？通过后将进入"已排单"阶段。`,
        confirmColor: '#52c41a',
        success: (res) => {
          if (res.confirm) this.runBatch('review-approve');
        }
      });
    },

    onBatchReject() {
      wx.showModal({
        title: '驳回订单',
        content: `驳回所选 ${this.data.selectedIds.length} 条订单。请填写驳回原因（客户可在订单详情查看）：`,
        editable: true,
        placeholderText: '如：身材数据缺失、参考图模糊…',
        confirmColor: '#ff4d4f',
        success: (res) => {
          if (!res.confirm) return;
          const remark = (res.content || '').trim() || '信息有误，请联系客服';
          this.runBatch('review-reject', { remark });
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
          tbOrderId: '', queueNumber: '', customerName: '', roleName: '', ip: '',
          orderTime: this.data.todayDate, deadline: '',
          progressPercent: 10, progressStage: '已排单', stage: 'queued',
          isUrgent: false,
          taobaoName: '', qq: '', phone: '',
          height: 0, weight: 0, headCircumference: 0, shoulderWidth: 0,
          needAccessory: false, needReplaceFace: false, replaceFaceCount: 1,
          remark: ''
        },
        referenceImages: [],
        replaceFaceImages: [],
        faceCountIndex: 0,
        stageIndex: 0
      });
    },

    onCloseOrderForm() {
      this.setData({ showOrderForm: false });
    },

    onFormInputChange(e: any) {
      const { field } = e.currentTarget.dataset;
      this.setData({ [`orderForm.${field}`]: e.detail.value });
    },

    onFormNumberChange(e: any) {
      const { field } = e.currentTarget.dataset;
      const v = parseFloat(e.detail.value);
      this.setData({ [`orderForm.${field}`]: isNaN(v) ? 0 : v });
    },

    onToggleField(e: any) {
      const { field } = e.currentTarget.dataset;
      const cur = (this.data.orderForm as any)[field];
      this.setData({ [`orderForm.${field}`]: !cur });
    },

    onFaceCountChange(e: any) {
      const idx = parseInt(e.detail.value);
      const count = this.data.faceCountOptions[idx];
      const trimmed = this.data.replaceFaceImages.slice(0, count);
      this.setData({
        faceCountIndex: idx,
        'orderForm.replaceFaceCount': count,
        replaceFaceImages: trimmed
      });
    },

    onChooseReferenceImages() {
      const remain = 3 - this.data.referenceImages.length;
      if (remain <= 0) {
        wx.showToast({ title: '最多 3 张', icon: 'none' });
        return;
      }
      wx.chooseMedia({
        count: remain,
        mediaType: ['image'],
        sizeType: ['compressed'],
        success: (res) => {
          const next = [
            ...this.data.referenceImages,
            ...res.tempFiles.map(f => f.tempFilePath)
          ].slice(0, 3);
          this.setData({ referenceImages: next });
        }
      });
    },

    onRemoveReferenceImage(e: any) {
      const idx = e.currentTarget.dataset.index;
      const arr = [...this.data.referenceImages];
      arr.splice(idx, 1);
      this.setData({ referenceImages: arr });
    },

    onChooseFaceImages() {
      const limit = this.data.orderForm.replaceFaceCount;
      const remain = limit - this.data.replaceFaceImages.length;
      if (remain <= 0) {
        wx.showToast({ title: `最多 ${limit} 张`, icon: 'none' });
        return;
      }
      wx.chooseMedia({
        count: remain,
        mediaType: ['image'],
        sizeType: ['compressed'],
        success: (res) => {
          const next = [
            ...this.data.replaceFaceImages,
            ...res.tempFiles.map(f => f.tempFilePath)
          ].slice(0, limit);
          this.setData({ replaceFaceImages: next });
        }
      });
    },

    onRemoveFaceImage(e: any) {
      const idx = e.currentTarget.dataset.index;
      const arr = [...this.data.replaceFaceImages];
      arr.splice(idx, 1);
      this.setData({ replaceFaceImages: arr });
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

    async uploadBatch(paths: string[], prefix: string): Promise<string[]> {
      const urls: string[] = [];
      for (let i = 0; i < paths.length; i++) {
        const p = paths[i];
        if (p.startsWith('cloud://')) { urls.push(p); continue; }
        const ext = p.match(/\.(\w+)$/)?.[1] || 'jpg';
        const cloudPath = `orders/${prefix}/${Date.now()}_${i}.${ext}`;
        const r: any = await wx.cloud.uploadFile({ cloudPath, filePath: p });
        urls.push(r.fileID);
      }
      return urls;
    },

    async onSubmitOrderForm() {
      const { orderForm, referenceImages, replaceFaceImages } = this.data;
      const required = ['tbOrderId', 'customerName', 'roleName', 'orderTime', 'deadline'];
      const labels: any = {
        tbOrderId: '淘宝订单号', customerName: '客户名称', roleName: '角色名称',
        orderTime: '下单时间', deadline: '预期完成时间'
      };
      for (const f of required) {
        if (!(orderForm as any)[f]) {
          wx.showToast({ title: `请填写${labels[f]}`, icon: 'none' });
          return;
        }
      }
      if (!orderForm.height || !orderForm.headCircumference) {
        wx.showToast({ title: '请填写身高和头围', icon: 'none' });
        return;
      }
      if (orderForm.needReplaceFace && replaceFaceImages.length < orderForm.replaceFaceCount) {
        wx.showToast({ title: `请上传 ${orderForm.replaceFaceCount} 张替换脸图`, icon: 'none' });
        return;
      }

      this.setData({ isSubmitting: true });
      try {
        const referenceImageUrls = await this.uploadBatch(referenceImages, 'reference');
        const replaceFaceImageUrls = orderForm.needReplaceFace
          ? await this.uploadBatch(replaceFaceImages, 'replace-face')
          : [];

        await wx.cloud.callFunction({
          name: 'createOrder',
          data: {
            // 基本
            tbOrderId: orderForm.tbOrderId,
            queueNumber: orderForm.queueNumber,
            customerName: orderForm.customerName,
            roleName: orderForm.roleName,
            ip: orderForm.ip,
            // 时间
            orderTime: orderForm.orderTime,
            deadline: orderForm.deadline,
            // 进度
            stage: orderForm.stage,
            progressStage: orderForm.progressStage,
            progressPercent: orderForm.progressPercent,
            // 状态
            isUrgent: orderForm.isUrgent,
            status: orderForm.isUrgent ? 'urgent' : 'normal',
            // 联系
            userInfo: {
              taobaoName: orderForm.taobaoName,
              qq: orderForm.qq,
              phone: orderForm.phone
            },
            // 身材
            bodyMeasurements: {
              height: orderForm.height,
              weight: orderForm.weight,
              headCircumference: orderForm.headCircumference,
              shoulderWidth: orderForm.shoulderWidth
            },
            // 选项
            options: {
              needAccessory: orderForm.needAccessory,
              needReplaceFace: orderForm.needReplaceFace,
              replaceFaceCount: orderForm.needReplaceFace ? orderForm.replaceFaceCount : 0,
              isUrgent: orderForm.isUrgent
            },
            // 图片
            referenceImages: referenceImageUrls,
            replaceFaceImages: replaceFaceImageUrls,
            // 备注
            remark: orderForm.remark
          }
        });
        wx.showToast({ title: '创建成功', icon: 'success' });
        this.setData({ showOrderForm: false, isSubmitting: false });
        this.refresh();
      } catch (err) {
        console.error('创建订单失败', err);
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
    },

    // ============ 危险操作：清理订单 ============
    onClearMockOrders() {
      wx.showModal({
        title: '清理测试数据',
        content: '将删除所有 queueNumber 以 RatStudio-2026- 开头的测试订单。确定继续？',
        confirmText: '清理',
        confirmColor: '#d04848',
        success: async (res) => {
          if (!res.confirm) return;
          wx.showLoading({ title: '清理中...' });
          try {
            const r: any = await wx.cloud.callFunction({
              name: 'clearOrders',
              data: { mode: 'mock' }
            });
            wx.hideLoading();
            const result = r.result || {};
            if (result.success) {
              wx.showToast({ title: `已清理 ${result.removed} 条`, icon: 'none' });
              this.setData({ showFilterPopup: false });
              this.refresh();
            } else {
              wx.showToast({ title: result.error || '清理失败', icon: 'none' });
            }
          } catch (err) {
            wx.hideLoading();
            wx.showToast({ title: '网络异常', icon: 'none' });
          }
        }
      });
    },

    onBatchDelete() {
      const count = this.data.selectedIds.length;
      wx.showModal({
        title: '删除订单',
        content: `确认删除所选 ${count} 条订单？此操作不可恢复。`,
        confirmText: '删除',
        confirmColor: '#d04848',
        success: (res) => {
          if (res.confirm) this.runBatch('delete');
        }
      });
    }
  }
});
