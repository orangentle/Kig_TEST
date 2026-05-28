// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init({
  env: 'cloud1-9ga5mdp1028f94eb'
})

const db = cloud.database()

function generateOrderId() {
  const now = new Date()
  const year = now.getFullYear().toString().slice(-2)
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const random = Math.random().toString(36).substring(2, 8).toUpperCase()
  return `KIG${year}${month}${day}${random}`
}

exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext()
  const openid = wxContext.OPENID

  const {
    roleName,
    ip,
    bodyMeasurements,
    referenceImages,
    replaceFaceImages,
    options,
    remark
  } = event

  if (!roleName || !ip) {
    return { success: false, error: '角色名称和角色所属 IP 不能为空' }
  }

  if (!referenceImages || referenceImages.length === 0) {
    return { success: false, error: '请上传至少一张角色多视角图' }
  }

  const needReplaceFace = !!(options && options.needReplaceFace)
  const replaceFaceCount = needReplaceFace ? (options.replaceFaceCount || 0) : 0

  if (needReplaceFace) {
    if (!replaceFaceImages || replaceFaceImages.length !== replaceFaceCount) {
      return { success: false, error: `替换脸数量与图片数量不一致，应为 ${replaceFaceCount} 张` }
    }
  }

  try {
    const userResult = await db.collection('users').where({ _openid: openid }).get()

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

    const orderId = generateOrderId()

    const orderData = {
      orderId,
      _openid: openid,
      userInfo,
      // 基本信息
      roleName,
      ip,
      // 身材数据
      bodyMeasurements: bodyMeasurements || {},
      // 图片
      referenceImages,
      replaceFaceImages: replaceFaceImages || [],
      // 定制选项
      options: {
        needAccessory: !!(options && options.needAccessory),
        needReplaceFace,
        replaceFaceCount,
        isUrgent: !!(options && options.isUrgent)
      },
      isUrgent: !!(options && options.isUrgent),
      remark: remark || '',
      // 审核状态
      status: 'pending',
      // 锁定：提交后默认锁定，客户无法修改；管理员可解锁
      isLocked: true,
      lockedAt: db.serverDate(),
      // 淘宝订单号（审核通过后填写）
      taobaoOrderId: '',
      reviewInfo: {
        reviewTime: null,
        reviewBy: null,
        reviewRemark: ''
      },
      createTime: db.serverDate(),
      updateTime: db.serverDate()
    }

    const result = await db.collection('orders').add({ data: orderData })

    return { success: true, orderId, _id: result._id }
  } catch (error) {
    console.error('提交订单失败', error)
    return { success: false, error: error.message || '提交订单失败' }
  }
}
