// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV }) // 使用当前云环境
const db = cloud.database()
const ordersCollection = db.collection('orders')
const usersCollection = db.collection('users')

async function assertAdmin(openid) {
  if (!openid) return false
  const r = await usersCollection.where({ _openid: openid, isAdmin: true }).limit(1).get()
  return r.data.length > 0
}

// 云函数入口函数
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext()
  const { orderId } = event
  
  if (!orderId) {
    return {
      code: 400,
      message: '订单ID不能为空',
      data: null
    }
  }

  try {
    const isAdmin = await assertAdmin(wxContext.OPENID)
    const where = isAdmin
      ? { tbOrderId: orderId }
      : { tbOrderId: orderId, _openid: wxContext.OPENID }

    // 查询订单详情
    const orderQuery = await ordersCollection.where(where).limit(1).get()

    if (orderQuery.data && orderQuery.data.length > 0) {
      return {
        code: 200,
        message: '获取订单详情成功',
        data: orderQuery.data[0]
      }
    } else {
      return {
        code: 404,
        message: '未找到订单',
        data: null
      }
    }
  } catch (error) {
    console.error('获取订单详情失败', error)
    return {
      code: 500,
      message: '获取订单详情失败',
      data: null,
      error: error
    }
  }
} 