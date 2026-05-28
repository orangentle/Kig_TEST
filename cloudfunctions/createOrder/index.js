// 云函数入口文件
const cloud = require('wx-server-sdk');

cloud.init({
  env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const ordersCollection = db.collection('orders');

// 云函数入口函数
exports.main = async (event, context) => {
  try {
    // 获取用户的openId
    const wxContext = cloud.getWXContext();

    // 校验淘宝订单号唯一性
    const tbOrderIdTrim = (event.tbOrderId || '').trim();
    if (tbOrderIdTrim) {
      const dup = await ordersCollection.where({ tbOrderId: tbOrderIdTrim }).limit(1).get();
      if (dup.data.length > 0) {
        return { success: false, error: '该淘宝订单号已存在' };
      }
    }

    // 构建订单数据
    const orderData = {
      // 基本信息
      tbOrderId: tbOrderIdTrim, // 淘宝订单号
      queueNumber: event.queueNumber || '', // 排单号
      orderId: `KG${Date.now().toString().slice(-8)}`, // 生成系统订单号
      customerName: event.customerName || '', // 客户名称
      roleName: event.roleName || '', // 角色名称
      
      // 时间信息
      orderTime: event.orderTime || '', // 下单时间
      deadline: event.deadline || '', // 预期完成时间
      createTime: db.serverDate(), // 创建时间（服务器时间）
      
      // 进度信息
      progressPercent: event.progressPercent || 0, // 制作进度
      progressStage: event.progressStage || '订单确认', // 进度阶段
      stage: event.stage || 'confirm', // 制作阶段
      
      // 状态信息
      status: event.status || (event.isUrgent ? 'urgent' : 'normal'), // 订单状态
      isArchived: false, // 是否归档
      
      // 图片信息
      previewImage: event.previewImage || '', // 预期成品展示图
      
      // 创建者信息
      createdBy: wxContext.OPENID, // 创建者的openId
      updatedTime: db.serverDate() // 最后更新时间
    };
    
    // 将订单数据添加到数据库
    const result = await ordersCollection.add({
      data: orderData
    });
    
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