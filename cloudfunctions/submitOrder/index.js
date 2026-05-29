// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init({
  env: 'shushugongfang-d2gl1995c27f9730e'
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
    tbOrderId,
    roleName,
    ip,
    bodyMeasurements,
    referenceImages,
    replaceFaceImages,
    options,
    remark
  } = event

  const tbOrderIdTrim = (tbOrderId || '').toString().trim()
  if (!tbOrderIdTrim) {
    return { success: false, error: '请填写淘宝定金订单号' }
  }
  if (!roleName || !ip) {
    return { success: false, error: '角色名称和角色所属 IP 不能为空' }
  }

  if (!referenceImages || referenceImages.length === 0) {
    return { success: false, error: '请上传至少一张角色参考图' }
  }

  // 同一个淘宝定金订单号只能提交一次定制表
  try {
    const dup = await db.collection('orders').where({ tbOrderId: tbOrderIdTrim }).limit(1).get()
    if (dup.data && dup.data.length > 0) {
      return { success: false, error: '该淘宝订单号已提交过定制表，如需修改请联系客服' }
    }
  } catch (e) {
    console.error('校验订单号唯一性失败', e)
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
      tbOrderId: tbOrderIdTrim,
      _openid: openid,
      userInfo,
      customerName: (userInfo && userInfo.nickName) || '',
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
      // 进度阶段：待审核，审核通过后才进入 queued
      stage: 'pending',
      progressStage: '待审核',
      progressPercent: 0,
      // 归档 / 锁定
      isArchived: false,
      isLocked: true,
      lockedAt: db.serverDate(),
      // 淘宝订单号（审核通过后填写真实号）
      taobaoOrderId: '',
      reviewInfo: {
        reviewTime: null,
        reviewBy: null,
        reviewRemark: ''
      },
      orderTime: (() => {
        const d = new Date();
        return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
      })(),
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
