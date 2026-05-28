// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init({
  env: 'shushugongfang-d2gl1995c27f9730e'
})

// 云函数入口函数
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext()

  return {
    event,
    openid: wxContext.OPENID,
    appid: wxContext.APPID,
    unionid: wxContext.UNIONID,
  }
} 