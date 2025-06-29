// app.ts
interface IAppOption {
  globalData: {
    userInfo?: WechatMiniprogram.UserInfo,
    hasLogin: boolean,
    logoUrl: string,
    emptyImageUrl: string
  }
}

App<IAppOption>({
  globalData: {
    hasLogin: false,
    logoUrl: 'https://cdn-icons-png.flaticon.com/512/3069/3069187.png',
    emptyImageUrl: 'https://cdn-icons-png.flaticon.com/512/5445/5445197.png'
  },
  onLaunch() {
    // 展示本地存储能力
    const logs = wx.getStorageSync('logs') || []
    logs.unshift(Date.now())
    wx.setStorageSync('logs', logs)

    // 检查用户登录状态
    const userInfo = wx.getStorageSync('userInfo')
    if (userInfo) {
      this.globalData.userInfo = userInfo
      this.globalData.hasLogin = true
    }

    // 登录
    wx.login({
      success: res => {
        console.log(res.code)
        // 发送 res.code 到后台换取 openId, sessionKey, unionId
      },
    })
  },
})