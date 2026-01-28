// pages/gathering/merch/merch.ts
import { getMerchBannerUrl, getProductMainUrl, getProductDetailUrl } from '../../../utils/cloudStorage';

interface Banner {
  id: string;
  imageUrl: string;
  title: string;
  desc: string;
  linkUrl?: string;
}

interface Category {
  id: string;
  name: string;
  icon: string;
}

interface Product {
  id: string;
  name: string;
  desc: string;
  imageUrl: string;
  images: string[];
  price: number;
  originalPrice?: number;
  sales: number;
  tag?: string;
  categoryId: string;
  specs?: string[];
  content: string;
}

interface SortOption {
  label: string;
  value: string;
}

Page({
  data: {
    banners: [] as Banner[],
    categories: [] as Category[],
    currentCategory: 'all',
    currentCategoryName: '全部商品',
    products: [] as Product[],
    allProducts: [] as Product[],
    sortOptions: [
      { label: '综合排序' },
      { label: '销量优先' },
      { label: '价格从低到高' },
      { label: '价格从高到低' },
      { label: '最新上架' }
    ],
    currentSort: { label: '综合排序', value: 'default' },
    showSortSheet: false,
    showDetail: false,
    currentProduct: {} as Product,
    selectedSpec: '',
    isFavorite: false
  },

  onLoad() {
    this.loadBanners();
    this.loadCategories();
    this.loadProducts();
  },

  goBack() {
    wx.navigateBack();
  },

  loadBanners() {
    // TODO: 实际应从云数据库加载
    const banners: Banner[] = [
      {
        id: 'banner_001',
        imageUrl: getMerchBannerUrl('banner_001'),
        title: '偶壳OKR限定周边',
        desc: '独家设计，限量发售'
      },
      {
        id: 'banner_002',
        imageUrl: getMerchBannerUrl('banner_002'),
        title: '鼠鼠工作室原创',
        desc: '精心设计，品质保证'
      }
    ];
    this.setData({ banners });
  },

  loadCategories() {
    const categories: Category[] = [
      { id: 'all', name: '全部', icon: 'view-module' },
      { id: 'clothing', name: '娃衣', icon: 'relativity' },
      { id: 'accessory', name: '配饰', icon: 'gift' },
      { id: 'prop', name: '道具', icon: 'tools' },
      { id: 'stationery', name: '文具', icon: 'edit' },
      { id: 'bag', name: '包袋', icon: 'wallet' },
      { id: 'other', name: '其他', icon: 'ellipsis' }
    ];
    this.setData({ categories });
  },

  loadProducts() {
    // TODO: 实际应从云数据库加载
    const products: Product[] = [
      {
        id: 'prod_001',
        name: '春日樱花和服套装',
        desc: '1/6 BJD适用，含腰带发饰',
        imageUrl: getProductMainUrl('prod_001'),
        images: [getProductMainUrl('prod_001'), getProductDetailUrl('prod_001', 1)],
        price: 168,
        originalPrice: 198,
        sales: 56,
        tag: '热卖',
        categoryId: 'clothing',
        specs: ['粉色', '白色', '紫色'],
        content: '精选优质面料，手工缝制。套装包含：和服主体、腰带、发饰。适合1/6 BJD（约27cm）使用。\n\n注意事项：\n- 手工制品可能有轻微色差\n- 请勿机洗，建议轻柔手洗\n- 活动限定，售完即止'
      },
      {
        id: 'prod_002',
        name: '迷你摄影道具套装',
        desc: '包含小椅子、花瓶、书本等',
        imageUrl: getProductMainUrl('prod_002'),
        images: [getProductMainUrl('prod_002')],
        price: 89,
        sales: 128,
        tag: '新品',
        categoryId: 'prop',
        specs: ['田园风', '复古风', '现代风'],
        content: '精致微缩道具，完美搭配娃娃摄影。套装包含8件小道具，可根据场景自由搭配。'
      },
      {
        id: 'prod_003',
        name: '偶壳OKR徽章套装',
        desc: '5枚装，原创设计',
        imageUrl: getProductMainUrl('prod_003'),
        images: [getProductMainUrl('prod_003')],
        price: 35,
        sales: 234,
        categoryId: 'accessory',
        content: '原创设计徽章套装，包含5枚不同款式。可装饰包包、帽子等。金属材质，经久耐用。'
      },
      {
        id: 'prod_004',
        name: '娃聚限定帆布袋',
        desc: '大容量，可装多只娃娃',
        imageUrl: getProductMainUrl('prod_004'),
        images: [getProductMainUrl('prod_004')],
        price: 58,
        originalPrice: 78,
        sales: 89,
        tag: '限定',
        categoryId: 'bag',
        specs: ['米白色', '浅蓝色'],
        content: '大容量帆布袋，内有分隔层设计，可安全携带娃娃外出。加厚帆布材质，耐用不易破。'
      },
      {
        id: 'prod_005',
        name: '娃娃主题手账本',
        desc: 'A5尺寸，含贴纸',
        imageUrl: getProductMainUrl('prod_005'),
        images: [getProductMainUrl('prod_005')],
        price: 42,
        sales: 167,
        categoryId: 'stationery',
        content: '记录你的娃娃日常！A5尺寸手账本，内页含娃娃主题插图，附赠一张贴纸。'
      },
      {
        id: 'prod_006',
        name: '微型家具套装',
        desc: '1/12比例，木质材质',
        imageUrl: getProductMainUrl('prod_006'),
        images: [getProductMainUrl('prod_006')],
        price: 128,
        sales: 45,
        tag: '精品',
        categoryId: 'prop',
        specs: ['客厅套装', '卧室套装', '书房套装'],
        content: '精致木质微型家具，1/12比例。手工打磨，细节精美。适合OB11等小型娃娃使用。'
      }
    ];
    
    this.setData({ 
      products,
      allProducts: products
    });
  },

  selectCategory(e: WechatMiniprogram.TouchEvent) {
    const categoryId = e.currentTarget.dataset.id;
    const category = this.data.categories.find(c => c.id === categoryId);
    
    let filteredProducts = this.data.allProducts;
    if (categoryId !== 'all') {
      filteredProducts = this.data.allProducts.filter(p => p.categoryId === categoryId);
    }
    
    this.setData({
      currentCategory: categoryId,
      currentCategoryName: category?.name || '全部商品',
      products: filteredProducts
    });
  },

  showSortOptions() {
    this.setData({ showSortSheet: true });
  },

  hideSortOptions() {
    this.setData({ showSortSheet: false });
  },

  onSortSelect(e: WechatMiniprogram.CustomEvent) {
    const index = e.detail.index;
    const options = [
      { label: '综合排序', value: 'default' },
      { label: '销量优先', value: 'sales' },
      { label: '价格从低到高', value: 'price_asc' },
      { label: '价格从高到低', value: 'price_desc' },
      { label: '最新上架', value: 'newest' }
    ];
    
    const currentSort = options[index];
    let sortedProducts = [...this.data.products];
    
    switch (currentSort.value) {
      case 'sales':
        sortedProducts.sort((a, b) => b.sales - a.sales);
        break;
      case 'price_asc':
        sortedProducts.sort((a, b) => a.price - b.price);
        break;
      case 'price_desc':
        sortedProducts.sort((a, b) => b.price - a.price);
        break;
      default:
        break;
    }
    
    this.setData({
      currentSort,
      products: sortedProducts,
      showSortSheet: false
    });
  },

  onBannerTap(e: WechatMiniprogram.TouchEvent) {
    const item = e.currentTarget.dataset.item;
    if (item.linkUrl) {
      wx.navigateTo({ url: item.linkUrl });
    }
  },

  showProductDetail(e: WechatMiniprogram.TouchEvent) {
    const product = e.currentTarget.dataset.product;
    this.setData({
      currentProduct: product,
      selectedSpec: product.specs?.[0] || '',
      isFavorite: false,
      showDetail: true
    });
  },

  closeDetail() {
    this.setData({ showDetail: false });
  },

  onDetailClose(e: WechatMiniprogram.CustomEvent) {
    if (!e.detail.visible) {
      this.setData({ showDetail: false });
    }
  },

  selectSpec(e: WechatMiniprogram.TouchEvent) {
    const spec = e.currentTarget.dataset.spec;
    this.setData({ selectedSpec: spec });
  },

  toggleFavorite() {
    this.setData({ isFavorite: !this.data.isFavorite });
    wx.showToast({
      title: this.data.isFavorite ? '已收藏' : '已取消收藏',
      icon: 'none'
    });
  },

  shareProduct() {
    // 分享功能
    wx.showToast({
      title: '功能开发中',
      icon: 'none'
    });
  },

  addToCart() {
    if (this.data.currentProduct.specs && !this.data.selectedSpec) {
      wx.showToast({
        title: '请选择规格',
        icon: 'none'
      });
      return;
    }
    
    wx.showToast({
      title: '已加入购物车',
      icon: 'success'
    });
  },

  buyNow() {
    if (this.data.currentProduct.specs && !this.data.selectedSpec) {
      wx.showToast({
        title: '请选择规格',
        icon: 'none'
      });
      return;
    }
    
    // TODO: 跳转到订单确认页
    wx.showToast({
      title: '功能开发中',
      icon: 'none'
    });
  }
});
