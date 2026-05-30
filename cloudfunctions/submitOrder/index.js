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
    return { success: false, error: '鼠鼠找不到淘宝订单号呜呜~ 先填一下嘛 (｡>﹏<｡)' }
  }
  if (!roleName || !ip) {
    return { success: false, error: '角色名字和所属作品都要填哦~ 鼠鼠才能开工呀 ♡' }
  }

  if (!referenceImages || referenceImages.length === 0) {
    return { success: false, error: '至少要一张参考图嘛~ 不然鼠鼠不知道捏成什么样 (˃ ⌑ ˂ഃ )' }
  }

  // 淘宝订单号全局唯一（含已取消），与 DB 唯一索引 idx_tbOrderId 对齐
  // 取消订单不释放号位，如需复用需联系客服真删
  try {
    const dup = await db.collection('orders')
      .where({ tbOrderId: tbOrderIdTrim })
      .limit(1)
      .get()
    if (dup.data && dup.data.length > 0) {
      return { success: false, error: '这个单号已经有小伙伴用过啦~ 确认下是不是填错了？如需改动请联系客服喔 ♡' }
    }
  } catch (e) {
    console.error('校验订单号唯一性失败', e)
    return { success: false, error: '订单号校验出了点小问题~ 鼠鼠喘口气再试 (｡•́︿•̀｡)' }
  }

  const needReplaceFace = !!(options && options.needReplaceFace)
  const replaceFaceCount = needReplaceFace ? (options.replaceFaceCount || 0) : 0

  if (needReplaceFace) {
    if (!replaceFaceImages || replaceFaceImages.length !== replaceFaceCount) {
      return { success: false, error: `替换脸要 ${replaceFaceCount} 张图哦~ 一脸一图鼠鼠才不会认错呀 ♡` }
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
      isLocked: false,
      lockedAt: null,
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
    // 唯一索引冲突（并发提交或绕过 WHERE 检查时）
    const msg = (error && (error.errMsg || error.message)) || ''
    if (/duplicate|E11000|unique/i.test(msg)) {
      return { success: false, error: '这个单号已经有小伙伴用过啦~ 确认下是不是填错了？如需改动请联系客服喔 ♡' }
    }
    return { success: false, error: error.message || '提交失败惹~ 鼠鼠再试一次嘛 (｡•́︿•̀｡)' }
  }
}
