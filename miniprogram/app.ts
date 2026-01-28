// app.ts
interface IAppOption {
  globalData: {
    userInfo?: WechatMiniprogram.UserInfo,
    hasLogin: boolean,
    logoUrl: string,
    emptyImageUrl: string,
    cloudEnv: string,
    cloudStorageBase: string  // 云存储基础路径
  }
}

App<IAppOption>({
  globalData: {
    hasLogin: false,
    logoUrl: '/assets/images/icon.jpg',
    emptyImageUrl: 'https://cdn-icons-png.flaticon.com/512/5445/5445197.png',
    cloudEnv: 'cloud1-9ga5mdp1028f94eb',
    cloudStorageBase: 'cloud://cloud1-9ga5mdp1028f94eb.636c-cloud1-9ga5mdp1028f94eb-1330924565'
  },
  onLaunch() {
    // 展示本地存储能力
    const logs = wx.getStorageSync('logs') || []
    logs.unshift(Date.now())
    wx.setStorageSync('logs', logs)

    // 初始化云开发
    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力')
    } else {
      wx.cloud.init({
        env: this.globalData.cloudEnv,
        traceUser: true
      })
      
      console.log('云环境初始化成功：', this.globalData.cloudEnv)
    }

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