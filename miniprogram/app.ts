// app.ts
interface IAppOption {
  globalData: {
    userInfo?: WechatMiniprogram.UserInfo,
    hasLogin: boolean,
    emptyImageUrl: string,
    cloudEnv: string,
    cloudStorageBase: string,
    brandLogoFileId: string,
    brandLogoUrl: string,
    brandLogoReady?: Promise<string>,
    ratLogoFileId: string,
    ratLogoUrl: string,
    ratLogoReady?: Promise<string>
  },
  resolveCloudUrl(fileId: string, cacheKey: string): Promise<string>
}

App<IAppOption>({
  globalData: {
    hasLogin: false,
    emptyImageUrl: 'https://cdn-icons-png.flaticon.com/512/5445/5445197.png',
    cloudEnv: 'shushugongfang-d2gl1995c27f9730e',
    cloudStorageBase: 'cloud://shushugongfang-d2gl1995c27f9730e.7368-shushugongfang-d2gl1995c27f9730e-1333429405',
    brandLogoFileId: 'cloud://shushugongfang-d2gl1995c27f9730e.7368-shushugongfang-d2gl1995c27f9730e-1333429405/static/brand/okr_logo.png',
    brandLogoUrl: '',
    ratLogoFileId: 'cloud://shushugongfang-d2gl1995c27f9730e.7368-shushugongfang-d2gl1995c27f9730e-1333429405/static/brand/rat_logo.png',
    ratLogoUrl: ''
  },
  onLaunch() {
    const logs = wx.getStorageSync('logs') || []
    logs.unshift(Date.now())
    wx.setStorageSync('logs', logs)

    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力')
    } else {
      wx.cloud.init({
        env: this.globalData.cloudEnv,
        traceUser: true
      })

      // 命中缓存时同步填充 globalData，避免页面拿到空字符串
      const brandCached = wx.getStorageSync('brandLogoUrl')
      if (brandCached && brandCached.url && brandCached.expireAt > Date.now()) {
        this.globalData.brandLogoUrl = brandCached.url
      }
      const ratCached = wx.getStorageSync('ratLogoUrl')
      if (ratCached && ratCached.url && ratCached.expireAt > Date.now()) {
        this.globalData.ratLogoUrl = ratCached.url
      }

      this.globalData.brandLogoReady = this.resolveCloudUrl(this.globalData.brandLogoFileId, 'brandLogoUrl')
        .then(url => { if (url) this.globalData.brandLogoUrl = url; return url })
      this.globalData.ratLogoReady = this.resolveCloudUrl(this.globalData.ratLogoFileId, 'ratLogoUrl')
        .then(url => { if (url) this.globalData.ratLogoUrl = url; return url })
    }

    const userInfo = wx.getStorageSync('userInfo')
    if (userInfo) {
      this.globalData.userInfo = userInfo
      this.globalData.hasLogin = true
    }

    wx.login({
      timeout: 8000,
      success: () => { /* code 由具体业务调用时再用 */ },
      fail: err => {
        console.warn('[wx.login] 失败', err && err.errMsg)
      }
    })

    // 全局兜底：未捕获的 Promise rejection 会被冒泡成 "Error: timeout" 之类
    if (typeof wx.onUnhandledRejection === 'function') {
      wx.onUnhandledRejection(({ reason, promise }) => {
        console.warn('[unhandledRejection]', reason, promise)
      })
    }
    if (typeof wx.onError === 'function') {
      wx.onError(err => {
        console.warn('[onError]', err)
      })
    }
  },

  // 云存储 tempFileURL 默认 2 小时有效，本地缓存 1 小时
  async resolveCloudUrl(fileId: string, cacheKey: string) {
    try {
      const cached = wx.getStorageSync(cacheKey)
      if (cached && cached.url && cached.expireAt > Date.now()) {
        return cached.url
      }
      const res = await wx.cloud.getTempFileURL({ fileList: [fileId] })
      const item = res.fileList && res.fileList[0]
      const url = (item && item.tempFileURL) || ''
      if (url) {
        wx.setStorageSync(cacheKey, { url, expireAt: Date.now() + 60 * 60 * 1000 })
      } else {
        console.warn('[cloudUrl] 拿到空 URL', cacheKey, fileId, item)
      }
      return url
    } catch (e) {
      console.warn('[cloudUrl] 解析异常', cacheKey, fileId, e)
      return ''
    }
  },
})
