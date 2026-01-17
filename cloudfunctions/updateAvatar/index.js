// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init({
  env: 'cloud1-9ga5mdp1028f94eb'
})

// 云函数入口函数
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext()
  const { avatarUrl, nickName, taobaoName, qq, phone, bodyMeasurements } = event
  
  try {
    // 获取用户信息
    const db = cloud.database()
    const userCollection = db.collection('users')
    
    // 查询当前用户
    const userResult = await userCollection.where({
      _openid: wxContext.OPENID
    }).get()
    
    if (userResult.data && userResult.data.length > 0) {
      // 用户存在，更新用户信息
      const userId = userResult.data[0]._id
      
      // 准备更新数据
      const updateData = {
        updateTime: Date.now()
      }
      
      // 只更新提供的字段
      if (avatarUrl !== undefined) updateData.avatarUrl = avatarUrl
      if (nickName !== undefined) updateData.nickName = nickName
      if (taobaoName !== undefined) updateData.taobaoName = taobaoName
      if (qq !== undefined) updateData.qq = qq
      if (phone !== undefined) updateData.phone = phone
      if (bodyMeasurements !== undefined) updateData.bodyMeasurements = bodyMeasurements
      
      const result = await userCollection.doc(userId).update({
        data: updateData
      })
      
      return {
        success: true,
        result,
        openid: wxContext.OPENID
      }
    } else {
      // 用户不存在
      return {
        success: false,
        error: 'User not found',
        openid: wxContext.OPENID
      }
    }
  } catch (error) {
    console.error('更新用户信息失败', error)
    return {
      success: false,
      error: error.message,
      openid: wxContext.OPENID
    }
  }
} 