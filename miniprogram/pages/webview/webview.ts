Component({
  data: {
    url: ''
  },
  
  lifetimes: {
    attached() {
      const eventChannel = this.getOpenerEventChannel();
      
      // 尝试从页面参数获取url
      const query = wx.getStorageSync('webviewUrl') || '';
      if (query) {
        this.setData({
          url: query
        });
        wx.removeStorageSync('webviewUrl');
        return;
      }
      
      // 如果没有获取到url，返回上一页
      wx.showToast({
        title: '链接无效',
        icon: 'error'
      });
      
      setTimeout(() => {
        wx.navigateBack();
      }, 1500);
    }
  }
}) 