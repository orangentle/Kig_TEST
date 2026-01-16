// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init({
  env: 'cloud1-9ga5mdp1028f94eb'
})

// 云函数入口函数
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext()
  const db = cloud.database()
  
  try {
    // 清空现有订单数据（谨慎使用，仅用于测试）
    // 如果是生产环境，应该检查是否存在数据后再添加
    try {
      await db.collection('orders').where({
        _id: db.command.exists(true)
      }).remove()
      console.log('清空现有订单数据成功')
    } catch (error) {
      console.error('清空订单数据失败', error)
    }
    
    // 准备测试订单数据
    const testOrders = [
      {
        orderId: 'TB100000001',
        roleName: '雷电将军头壳',
        orderTime: new Date('2026-02-20').getTime(),
        status: 'processing',
        createTime: Date.now(),
        details: { color: '紫黑', size: '标准', notes: '需做雷光发饰效果' }
      },
      {
        orderId: 'TB100000002',
        roleName: '甘雨头壳',
        orderTime: new Date('2026-02-05').getTime(),
        status: 'completed',
        createTime: Date.now(),
        details: { color: '冰蓝', size: '标准', notes: '冰弓蝴蝶结与发角细节' }
      },
      {
        orderId: 'TB100000003',
        roleName: '胡桃头壳',
        orderTime: new Date('2026-01-28').getTime(),
        status: 'completed',
        createTime: Date.now(),
        details: { color: '深棕', size: '小号', notes: '蝶引来生纹路与帽饰' }
      }
    ]
    
    // 添加测试订单数据
    const addPromises = testOrders.map(order => {
      return db.collection('orders').add({
        data: order
      })
    })
    
    const results = await Promise.all(addPromises)
    
    return {
      success: true,
      results,
      message: '初始化订单数据成功'
    }
  } catch (error) {
    console.error('初始化订单数据失败', error)
    return {
      success: false,
      error,
      message: '初始化订单数据失败'
    }
  }
}