// order-detail.ts
// 获取应用实例
const appInstance = getApp<IAppOption>();

interface OrderDetail {
  orderId: string;
  queueId: string;
  roleName: string;
  expectedCompletionDate: string;
  currentStep: number;
  previewImage: string;
}

interface Step {
  title: string;
  content: string;
}

Component({
  data: {
    orderId: '',
    order: {
      orderId: '',
      queueId: '',
      roleName: '',
      expectedCompletionDate: '',
      currentStep: 3,
      previewImage: ''
    } as OrderDetail,
    currentStep: 3, // 默认当前步骤为打印阶段
    steps: [
      { title: '订单确认', content: '确认订单信息和需求' },
      { title: '设计图确认', content: '确认头壳设计图纸' },
      { title: '模型制作', content: '根据设计图制作3D模型' },
      { title: '打印', content: '3D打印头壳部件' },
      { title: '打磨上色', content: '对打印件进行打磨和上色' },
      { title: '组装', content: '组装头壳各部件' },
      { title: '质检', content: '对成品进行质量检查' },
      { title: '发货', content: '包装并发货' }
    ] as Step[]
  },

  lifetimes: {
    attached() {
      // 从页面参数获取订单ID
      const pages = getCurrentPages();
      const currentPage = pages[pages.length - 1];
      // @ts-ignore
      const options = currentPage.options;
      
      console.log('页面参数:', options);
      
      if (options && options.id) {
        const orderId = options.id;
        console.log('获取到订单ID:', orderId);
        this.setData({ orderId });
        this.loadOrderDetail(orderId);
      } else {
        console.log('未获取到订单ID，使用默认ID');
        // 如果没有获取到ID，使用默认ID
        this.loadOrderDetail('TB123456789');
      }
    }
  },

  methods: {
    // 加载订单详情
    loadOrderDetail(orderId: string) {
      console.log('加载订单详情:', orderId);
      
      wx.showLoading({
        title: '加载中...'
      });
      
      // 模拟API请求
      setTimeout(() => {
        // 模拟数据
        let mockData: OrderDetail;
        
        // 根据订单ID返回不同的模拟数据
        if (orderId === 'TB987654321') {
          mockData = {
            orderId: 'TB987654321',
            queueId: 'RatStudio-2025-002',
            roleName: '猫咪头壳',
            expectedCompletionDate: '2025-11-15',
            currentStep: 7, // 已完成
            previewImage: 'https://img.alicdn.com/imgextra/i1/2201504856228/O1CN01KZLNVJ1V6zEiOJJcP_!!2201504856228.jpg'
          };
        } else if (orderId === 'TB456789123') {
          mockData = {
            orderId: 'TB456789123',
            queueId: 'RatStudio-2025-003',
            roleName: '兔子头壳',
            expectedCompletionDate: '2025-10-20',
            currentStep: 7, // 已完成
            previewImage: 'https://img.alicdn.com/imgextra/i1/2201504856228/O1CN01KZLNVJ1V6zEiOJJcP_!!2201504856228.jpg'
          };
        } else {
          // 默认数据或TB123456789
          mockData = {
            orderId: orderId || 'TB123456789',
            queueId: 'RatStudio-2025-001',
            roleName: '狐狸头壳',
            expectedCompletionDate: '2025-12-30',
            currentStep: 3, // 打印阶段
            previewImage: 'https://img.alicdn.com/imgextra/i1/2201504856228/O1CN01KZLNVJ1V6zEiOJJcP_!!2201504856228.jpg'
          };
        }
        
        console.log('设置订单数据:', mockData);
        
        this.setData({
          order: mockData,
          currentStep: mockData.currentStep
        });
        
        wx.hideLoading();
      }, 1000);
    },
    
    // 联系客服
    onContactService() {
      wx.showModal({
        title: '联系客服',
        content: '即将打开淘宝店铺客服页面',
        success: (res) => {
          if (res.confirm) {
            // 使用淘宝短链接
            const taobaoUrl = 'https://m.tb.cn/h.hf1womHplfjsH5V';
            wx.setStorageSync('webviewUrl', taobaoUrl);
            
            wx.navigateTo({
              url: '/pages/webview/webview',
              success: () => {
                console.log('成功打开淘宝网页');
              },
              fail: (err) => {
                console.error('打开淘宝网页失败', err);
                wx.showToast({
                  title: '打开失败，请稍后重试',
                  icon: 'none'
                });
              }
            });
          }
        }
      });
    }
  }
}) 