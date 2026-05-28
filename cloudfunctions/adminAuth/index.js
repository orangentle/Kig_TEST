// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init({
  env: 'shushugongfang-d2gl1995c27f9730e'
})

// 管理员密码，实际应用中应该存储在数据库中
const ADMIN_PASSWORD = '123456'

// 云函数入口函数
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext()
  const { password } = event
  
  // 验证密码
  const success = password === ADMIN_PASSWORD
  
  // 如果验证成功，可以在数据库中标记该用户为管理员
  if (success) {
    try {
      const db = cloud.database()
      // 查询用户是否存在
      const userResult = await db.collection('users').where({
        _openid: wxContext.OPENID
      }).get()
      
      if (userResult.data && userResult.data.length > 0) {
        // 更新用户为管理员
        await db.collection('users').doc(userResult.data[0]._id).update({
          data: {
            isAdmin: true,
            updateTime: Date.now()
          }
        })
      }
    } catch (error) {
      console.error('标记管理员状态失败', error)
    }
  }

  return {
    event,
    success,
    openid: wxContext.OPENID,
  }
} 