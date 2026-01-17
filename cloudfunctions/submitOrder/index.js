// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init({
  env: 'cloud1-9ga5mdp1028f94eb'
})

const db = cloud.database()

// 生成订单号
function generateOrderId() {
  const now = new Date()
  const year = now.getFullYear().toString().slice(-2)
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const random = Math.random().toString(36).substring(2, 8).toUpperCase()
  return `KIG${year}${month}${day}${random}`
}

// 云函数入口函数
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext()
  const openid = wxContext.OPENID
  
  const {
    roleName,
    sourceWork,
    bodyMeasurements,
    referenceImages,
    options,
    remark
  } = event
  
  // 参数验证
  if (!roleName || !sourceWork) {
    return {
      success: false,
      error: '角色名称和来源作品不能为空'
    }
  }
  
  if (!referenceImages || referenceImages.length === 0) {
    return {
      success: false,
      error: '请上传至少一张表情参考图'
    }
  }
  
  try {
    // 获取用户信息
    const userResult = await db.collection('users').where({
      _openid: openid
    }).get()
    
    let userInfo = null
    if (userResult.data && userResult.data.length > 0) {
      userInfo = {
        nickName: userResult.data[0].nickName,
        avatarUrl: userResult.data[0].avatarUrl,
        phone: userResult.data[0].phone,
        taobaoName: userResult.data[0].taobaoName,
        qq: userResult.data[0].qq
      }
    }
    
    // 生成订单号
    const orderId = generateOrderId()
    
    // 创建订单数据
    const orderData = {
      orderId: orderId,
      _openid: openid,
      userInfo: userInfo,
      // 基本信息
      roleName: roleName,
      sourceWork: sourceWork,
      // 身材数据
      bodyMeasurements: bodyMeasurements || {},
      // 参考图片
      referenceImages: referenceImages,
      // 定制选项
      options: {
        needReplaceFace: options?.needReplaceFace || false,
        needHeadwear: options?.needHeadwear || false,
        needAntiGravity: options?.needAntiGravity || false,
        needCornsilkPerm: options?.needCornsilkPerm || false,
        isUrgent: options?.isUrgent || false
      },
      // 加急标识（单独存储便于查询）
      isUrgent: options?.isUrgent || false,
      // 备注
      remark: remark || '',
      // 订单状态: pending(待审核), approved(已通过), rejected(已拒绝), processing(制作中), completed(已完成)
      status: 'pending',
      // 淘宝订单号（管理员审核通过后填写）
      taobaoOrderId: '',
      // 审核信息
      reviewInfo: {
        reviewTime: null,
        reviewBy: null,
        reviewRemark: ''
      },
      // 时间戳
      createTime: db.serverDate(),
      updateTime: db.serverDate()
    }
    
    // 添加到数据库
    const result = await db.collection('orders').add({
      data: orderData
    })
    
    return {
      success: true,
      orderId: orderId,
      _id: result._id
    }
  } catch (error) {
    console.error('提交订单失败', error)
    return {
      success: false,
      error: error.message || '提交订单失败'
    }
  }
}
