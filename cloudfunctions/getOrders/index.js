// 云函数入口文件
const cloud = require('wx-server-sdk');

cloud.init({
  env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const ordersCollection = db.collection('orders');
const MAX_LIMIT = 100;

// 云函数入口函数
exports.main = async (event, context) => {
  try {
    // 获取订单状态过滤条件
    const { isArchived = false } = event;
    
    // 构建查询条件
    const condition = { isArchived };
    
    // 获取总数
    const countResult = await ordersCollection.where(condition).count();
    const total = countResult.total;
    
    // 计算需要分几次取
    const batchTimes = Math.ceil(total / MAX_LIMIT);
    
    // 承载所有读操作的promise的数组
    const tasks = [];
    
    for (let i = 0; i < batchTimes; i++) {
      const promise = ordersCollection
        .where(condition)
        .skip(i * MAX_LIMIT)
        .limit(MAX_LIMIT)
        .get();
      
      tasks.push(promise);
    }
    
    // 等待所有
    const results = await Promise.all(tasks);
    
    // 合并数据
    let orders = [];
    results.forEach(result => {
      orders = orders.concat(result.data);
    });
    
    return {
      success: true,
      data: orders,
      total
    };
  } catch (error) {
    console.error('获取订单列表失败', error);
    return {
      success: false,
      error: error.message
    };
  }
}; 