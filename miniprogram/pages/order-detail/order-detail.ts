// order-detail.ts
import type { Order } from '../../types/order';
import { STAGE_FLOW } from '../../types/order';

type OrderDetail = Order;

interface Step {
  title: string;
  content: string;
}

const STAGE_OPTIONS = [
  { value: 'pending', label: '待审核', percent: 0 },
  ...STAGE_FLOW
];

Page({
  data: {
    tbOrderId: '',
    order: {
      tbOrderId: '',
      queueNumber: '',
      orderId: '',
      customerName: '',
      roleName: '',
      orderTime: '',
      deadline: '',
      progressPercent: 0,
      progressStage: '',
      stage: '',
      status: ''
    } as unknown as OrderDetail,
    currentStepIndex: 0,
    progressPercent: 0,
    steps: [
      { title: '已排单', content: '订单收到啦~ 鼠鼠已经记在小本本上 📝' },
      { title: '建模', content: '在电脑里捏出小模型,打印机嘎嘎转~ 🖨️' },
      { title: '上妆', content: '打磨、喷漆,给宝贝化个美美的妆 ✨' },
      { title: '假毛', content: '一根一根种毛中,温柔对待每一缕~' },
      { title: '已发货', content: '检查无误,打包出发去找你啦~ 🚚💨' }
    ] as Step[],
    isAdmin: false,
    notFound: false,
    loadError: false,

    // 编辑订单
    showEditForm: false,
    isSaving: false,
    editForm: {
      tbOrderId: '', queueNumber: '', customerName: '', roleName: '', ip: '',
      taobaoName: '', qq: '', phone: '',
      height: 0, weight: 0, headCircumference: 0, shoulderWidth: 0,
      needAccessory: false, needReplaceFace: false, replaceFaceCount: 1,
      isUrgent: false,
      orderTime: '', deadline: '',
      stage: 'queued', progressStage: '已排单', progressPercent: 10,
      remark: ''
    } as any,
    editReferenceImages: [] as string[],
    editFaceImages: [] as string[],
    faceCountOptions: [1, 2, 3],
    faceCountIndex: 0,
    stageOptions: STAGE_OPTIONS,
    stageIndex: 1
  },

  onLoad(options) {
    if (!options || !options.id) {
      this.setData({ notFound: true });
      return;
    }
    const tbOrderId = options.id;
    this.setData({
      tbOrderId,
      isAdmin: options.admin === 'true'
    });
    this.loadOrderDetail(tbOrderId);
  },

  loadOrderDetail(tbOrderId: string) {
    wx.showLoading({ title: '鼠鼠在搬数据~' });

    wx.cloud.callFunction({
      name: 'getOrders',
      data: { tbOrderId },
      success: (res: any) => {
        wx.hideLoading();
        const orderList = res.result && res.result.data;
        if (!orderList || orderList.length === 0) {
          this.setData({ notFound: true });
          return;
        }
        const orderData = orderList.find((o: any) => o.tbOrderId === tbOrderId) || orderList[0];
        // 兼容老订单没有 orderTime 字段: 用 createTime 兜底
        if (!orderData.orderTime && orderData.createTime) {
          const d = new Date(orderData.createTime);
          if (!isNaN(d.getTime())) {
            orderData.orderTime = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
          }
        }
        this.setData({
          order: orderData,
          currentStepIndex: this.getStepIndexFromStage(orderData.stage),
          progressPercent: orderData.progressPercent || 0,
          notFound: false,
          loadError: false
        });
      },
      fail: (err) => {
        wx.hideLoading();
        console.error('获取订单详情失败', err);
        this.setData({ loadError: true });
        wx.showToast({ title: '咦,加载迷路了,再试一次?', icon: 'none' });
      }
    });
  },

  getStepIndexFromStage(stage: string): number {
    if (stage === 'pending') return -1;
    const map: Record<string, number> = {
      queued: 0, modeling: 1, painting: 2, hair: 3, shipped: 4
    };
    return map[stage] || 0;
  },

  onBack() {
    wx.navigateBack();
  },

  onCancelOrder() {
    const o: any = this.data.order || {};
    if (!o || o.status !== 'pending') {
      wx.showToast({ title: '当前状态不可取消', icon: 'none' });
      return;
    }
    if (o.isLocked) {
      wx.showToast({ title: '订单已锁定，请联系客服', icon: 'none' });
      return;
    }
    wx.showModal({
      title: '确认取消订单？',
      content: '取消后无法恢复，定金退款请联系客服。',
      confirmText: '确认取消',
      cancelText: '再想想',
      confirmColor: '#cf1322',
      success: async (res) => {
        if (!res.confirm) return;
        wx.showLoading({ title: '取消中...' });
        try {
          const { result } = await wx.cloud.callFunction({
            name: 'cancelOrder',
            data: { tbOrderId: o.tbOrderId, reason: '用户主动取消' }
          }) as any;
          wx.hideLoading();
          if (result && result.success) {
            wx.showToast({ title: '已取消', icon: 'success' });
            this.loadOrderDetail(o.tbOrderId);
          } else {
            wx.showModal({
              title: '取消失败',
              content: (result && result.error) || '请稍后再试',
              showCancel: false
            });
          }
        } catch (err: any) {
          wx.hideLoading();
          wx.showToast({ title: err?.message || '取消失败', icon: 'none' });
        }
      }
    });
  },

  onPreviewImage(e: any) {
    const { urls, current } = e.currentTarget.dataset;
    if (!urls || !urls.length) return;
    wx.previewImage({ urls, current });
  },

  onAdminCopy(e: any) {
    const text = e.currentTarget.dataset.text;
    if (!text) return;
    wx.setClipboardData({
      data: String(text),
      success: () => wx.showToast({ title: '复制好啦~', icon: 'success' })
    });
  },

  onAdminCall(e: any) {
    const text = e.currentTarget.dataset.text;
    if (!text) return;
    wx.makePhoneCall({
      phoneNumber: String(text),
      fail: () => wx.showToast({ title: '拨打失败 (´;ω;`)', icon: 'none' })
    });
  },

  // ========== 编辑订单 ==========

  openEditForm() {
    const o: any = this.data.order || {};
    const ui = o.userInfo || {};
    const bm = o.bodyMeasurements || {};
    const op = o.options || {};
    const stage = o.stage || 'queued';
    const stageIndex = Math.max(0, STAGE_OPTIONS.findIndex(s => s.value === stage));
    const replaceFaceCount = op.replaceFaceCount || 1;
    const faceCountIndex = Math.max(0, this.data.faceCountOptions.indexOf(replaceFaceCount));

    this.setData({
      showEditForm: true,
      editForm: {
        tbOrderId: o.tbOrderId || '',
        queueNumber: o.queueNumber || '',
        customerName: o.customerName || '',
        roleName: o.roleName || '',
        ip: o.ip || '',
        taobaoName: ui.taobaoName || '',
        qq: ui.qq || '',
        phone: ui.phone || '',
        height: bm.height || 0,
        weight: bm.weight || 0,
        headCircumference: bm.headCircumference || 0,
        shoulderWidth: bm.shoulderWidth || 0,
        needAccessory: !!op.needAccessory,
        needReplaceFace: !!op.needReplaceFace,
        replaceFaceCount,
        isUrgent: !!o.isUrgent,
        orderTime: o.orderTime || '',
        deadline: o.deadline || '',
        stage,
        progressStage: o.progressStage || '已排单',
        progressPercent: o.progressPercent || 0,
        remark: o.remark || ''
      },
      editReferenceImages: Array.isArray(o.referenceImages) ? [...o.referenceImages] : [],
      editFaceImages: Array.isArray(o.replaceFaceImages) ? [...o.replaceFaceImages] : [],
      stageIndex,
      faceCountIndex
    });
  },

  closeEditForm() {
    this.setData({ showEditForm: false });
  },

  onEditInput(e: any) {
    const { field } = e.currentTarget.dataset;
    this.setData({ [`editForm.${field}`]: e.detail.value });
  },

  onEditNumber(e: any) {
    const { field } = e.currentTarget.dataset;
    const v = parseFloat(e.detail.value);
    this.setData({ [`editForm.${field}`]: isNaN(v) ? 0 : v });
  },

  onEditToggle(e: any) {
    const { field } = e.currentTarget.dataset;
    const cur = (this.data.editForm as any)[field];
    this.setData({ [`editForm.${field}`]: !cur });
  },

  onEditDate(e: any) {
    const { field } = e.currentTarget.dataset;
    this.setData({ [`editForm.${field}`]: e.detail.value });
  },

  onEditFaceCountChange(e: any) {
    const idx = parseInt(e.detail.value);
    const count = this.data.faceCountOptions[idx];
    const trimmed = this.data.editFaceImages.slice(0, count);
    this.setData({
      faceCountIndex: idx,
      'editForm.replaceFaceCount': count,
      editFaceImages: trimmed
    });
  },

  onEditStageChange(e: any) {
    const idx = parseInt(e.detail.value);
    const opt = this.data.stageOptions[idx];
    this.setData({
      stageIndex: idx,
      'editForm.stage': opt.value,
      'editForm.progressStage': opt.label,
      'editForm.progressPercent': opt.percent
    });
  },

  onChooseEditRefImages() {
    const remain = 3 - this.data.editReferenceImages.length;
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
          ...this.data.editReferenceImages,
          ...res.tempFiles.map(f => f.tempFilePath)
        ].slice(0, 3);
        this.setData({ editReferenceImages: next });
      }
    });
  },

  onRemoveEditRefImage(e: any) {
    const idx = e.currentTarget.dataset.index;
    const arr = [...this.data.editReferenceImages];
    arr.splice(idx, 1);
    this.setData({ editReferenceImages: arr });
  },

  onChooseEditFaceImages() {
    const limit = this.data.editForm.replaceFaceCount;
    const remain = limit - this.data.editFaceImages.length;
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
          ...this.data.editFaceImages,
          ...res.tempFiles.map(f => f.tempFilePath)
        ].slice(0, limit);
        this.setData({ editFaceImages: next });
      }
    });
  },

  onRemoveEditFaceImage(e: any) {
    const idx = e.currentTarget.dataset.index;
    const arr = [...this.data.editFaceImages];
    arr.splice(idx, 1);
    this.setData({ editFaceImages: arr });
  },

  async uploadBatch(paths: string[], prefix: string): Promise<string[]> {
    const urls: string[] = [];
    for (let i = 0; i < paths.length; i++) {
      const p = paths[i];
      if (p.startsWith('cloud://') || p.startsWith('http')) { urls.push(p); continue; }
      const ext = p.match(/\.(\w+)$/)?.[1] || 'jpg';
      const cloudPath = `orders/${prefix}/${Date.now()}_${i}.${ext}`;
      const r: any = await wx.cloud.uploadFile({ cloudPath, filePath: p });
      urls.push(r.fileID);
    }
    return urls;
  },

  async onSaveEdit() {
    const f = this.data.editForm;
    if (!f.tbOrderId) {
      wx.showToast({ title: '淘宝订单号不能空~', icon: 'none' });
      return;
    }
    if (!f.roleName) {
      wx.showToast({ title: '角色名称不能空~', icon: 'none' });
      return;
    }
    if (f.needReplaceFace && this.data.editFaceImages.length < f.replaceFaceCount) {
      wx.showToast({ title: `请上传 ${f.replaceFaceCount} 张替换脸图`, icon: 'none' });
      return;
    }

    const orderDocId = (this.data.order as any)._id;
    if (!orderDocId) {
      wx.showToast({ title: '订单 ID 缺失,刷新一下试试', icon: 'none' });
      return;
    }

    this.setData({ isSaving: true });
    wx.showLoading({ title: '鼠鼠在保存~' });

    try {
      const referenceImages = await this.uploadBatch(this.data.editReferenceImages, 'reference');
      const replaceFaceImages = f.needReplaceFace
        ? await this.uploadBatch(this.data.editFaceImages, 'replace-face')
        : [];

      const oldUserInfo = (this.data.order as any).userInfo || {};
      const fields: any = {
        tbOrderId: String(f.tbOrderId).trim(),
        queueNumber: f.queueNumber || '',
        customerName: f.customerName || '',
        roleName: f.roleName,
        ip: f.ip || '',
        orderTime: f.orderTime || '',
        deadline: f.deadline || '',
        stage: f.stage,
        progressStage: f.progressStage,
        progressPercent: f.progressPercent,
        isUrgent: !!f.isUrgent,
        status: f.isUrgent ? 'urgent' : ((this.data.order as any).status || 'normal'),
        userInfo: {
          ...oldUserInfo,
          taobaoName: f.taobaoName || '',
          qq: f.qq || '',
          phone: f.phone || ''
        },
        bodyMeasurements: {
          height: f.height || 0,
          weight: f.weight || 0,
          headCircumference: f.headCircumference || 0,
          shoulderWidth: f.shoulderWidth || 0
        },
        options: {
          needAccessory: !!f.needAccessory,
          needReplaceFace: !!f.needReplaceFace,
          replaceFaceCount: f.needReplaceFace ? f.replaceFaceCount : 0,
          isUrgent: !!f.isUrgent
        },
        referenceImages,
        replaceFaceImages,
        remark: f.remark || ''
      };

      const res: any = await wx.cloud.callFunction({
        name: 'batchUpdateOrders',
        data: { orderIds: [orderDocId], action: 'set-fields', payload: { fields } }
      });

      wx.hideLoading();
      this.setData({ isSaving: false });

      if (res.result && res.result.success) {
        wx.showToast({ title: '保存好啦~ ✨', icon: 'success' });
        this.setData({ showEditForm: false });
        this.loadOrderDetail(fields.tbOrderId);
      } else {
        const msg = (res.result && res.result.error) || '保存失败,再试试?';
        wx.showToast({ title: msg, icon: 'none' });
      }
    } catch (err: any) {
      console.error('保存订单失败', err);
      wx.hideLoading();
      this.setData({ isSaving: false });
      wx.showToast({ title: '保存迷路了 (´;ω;`)', icon: 'none' });
    }
  }
})
