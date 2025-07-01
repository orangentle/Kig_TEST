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
        orderId: 'TB123456789',
        roleName: '狐狸头壳',
        orderTime: new Date('2025-10-15').getTime(),
        status: 'processing',
        createTime: Date.now(),
        details: {
          color: '橙色',
          size: '标准',
          notes: '需要特殊定制耳朵'
        }
      },
      {
        orderId: 'TB987654321',
        roleName: '猫咪头壳',
        orderTime: new Date('2025-09-01').getTime(),
        status: 'completed',
        createTime: Date.now(),
        details: {
          color: '黑色',
          size: '标准',
          notes: '需要可拆卸的眼睛'
        }
      },
      {
        orderId: 'TB456789123',
        roleName: '兔子头壳',
        orderTime: new Date('2025-08-15').getTime(),
        status: 'completed',
        createTime: Date.now(),
        details: {
          color: '白色',
          size: '大号',
          notes: '长耳朵设计'
        }
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