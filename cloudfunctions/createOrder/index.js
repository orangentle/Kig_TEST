// 云函数入口文件
const cloud = require('wx-server-sdk');

cloud.init({
  env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const ordersCollection = db.collection('orders');

const VALID_STAGES = ['pending', 'queued', 'modeling', 'painting', 'hair', 'shipped'];

exports.main = async (event, context) => {
  try {
    const wxContext = cloud.getWXContext();

    // 校验淘宝订单号唯一性
    const tbOrderIdTrim = (event.tbOrderId || '').trim();
    if (!tbOrderIdTrim) {
      return { success: false, error: '淘宝订单号必填' };
    }
    const dup = await ordersCollection.where({ tbOrderId: tbOrderIdTrim }).limit(1).get();
    if (dup.data.length > 0) {
      return { success: false, error: '该淘宝订单号已存在' };
    }

    const stage = VALID_STAGES.includes(event.stage) ? event.stage : 'queued';
    const isUrgent = !!event.isUrgent;
    const status = event.status || (isUrgent ? 'urgent' : 'normal');

    // 与 Order 数据结构对齐
    const orderData = {
      // 基本
      tbOrderId: tbOrderIdTrim,
      queueNumber: (event.queueNumber || '').trim(),
      orderId: `KG${Date.now().toString().slice(-8)}`,
      customerName: event.customerName || '',
      roleName: event.roleName || '',
      ip: event.ip || '',

      // 时间
      orderTime: event.orderTime || '',
      deadline: event.deadline || '',
      createTime: db.serverDate(),
      updateTime: db.serverDate(),

      // 进度
      stage,
      progressStage: event.progressStage || '已排单',
      progressPercent: typeof event.progressPercent === 'number' ? event.progressPercent : 10,

      // 状态
      status,
      isUrgent,
      isArchived: false,
      isLocked: false,

      // 联系/身材/选项（嵌套快照）
      userInfo: event.userInfo || {},
      bodyMeasurements: event.bodyMeasurements || {},
      options: event.options || {},

      // 图片
      referenceImages: Array.isArray(event.referenceImages) ? event.referenceImages : [],
      replaceFaceImages: Array.isArray(event.replaceFaceImages) ? event.replaceFaceImages : [],

      // 备注 / 创建者
      remark: event.remark || '',
      createdBy: wxContext.OPENID,
      _openid: wxContext.OPENID
    };

    const result = await ordersCollection.add({ data: orderData });

    return {
      success: true,
      orderId: orderData.orderId,
      _id: result._id
    };
  } catch (error) {
    console.error('创建订单失败', error);
    return {
      success: false,
      error: error.message
    };
  }
};
