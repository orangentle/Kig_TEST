// pages/gathering/album/album.ts
import { getPhotoUrl, uploadPhoto, CLOUD_PATHS } from '../../../utils/cloudStorage';

const DEFAULT_AVATAR = 'https://mmbiz.qpic.cn/mmbiz/icTdbqWNOwNRna42FI242Lcia07jQodd2FJGIYQfG0LAJGFxM4FbnQP6yfMxBgJ0F3YRqJCJ1aPAK2dQagdusBZg/0';

async function getDefaultCover(): Promise<string> {
  const app = getApp<IAppOption>();
  return app.globalData.brandLogoUrl || (app.globalData.brandLogoReady ? await app.globalData.brandLogoReady : '');
}

interface Album {
  id: string;
  name: string;
  coverUrl: string;
  photoCount: number;
  date: string;
}

interface Photo {
  id: string;
  albumId: string;
  url: string;
  thumbnailUrl?: string;
  userName: string;
  userAvatar: string;
  uploadTime: string;
  likes: number;
  isLiked: boolean;
  comments?: number;
  description?: string;
  originalIndex?: number;
}

Page({
  data: {
    albums: [] as Album[],
    currentAlbum: 'all',
    currentPhotos: [] as Photo[],
    allPhotos: [] as Photo[],
    leftPhotos: [] as Photo[],
    rightPhotos: [] as Photo[],
    showPhotoDetail: false,
    currentPhotoDetail: {} as Photo
  },

  async onLoad() {
    const cover = await getDefaultCover();
    this.loadAlbums(cover);
    this.loadPhotos();
  },

  goBack() {
    wx.navigateBack();
  },

  loadAlbums(coverUrl: string) {
    // TODO: 实际应从云数据库加载
    const albums: Album[] = [
      {
        id: '2026spring',
        name: '2026春季娃聚',
        coverUrl,
        photoCount: 156,
        date: '2026-03-15'
      },
      {
        id: '2025winter',
        name: '2025冬季茶话会',
        coverUrl,
        photoCount: 89,
        date: '2025-12-20'
      },
      {
        id: '2025autumn',
        name: '2025秋季摄影会',
        coverUrl,
        photoCount: 234,
        date: '2025-10-05'
      },
      {
        id: '2025summer',
        name: '2025夏日市集',
        coverUrl,
        photoCount: 312,
        date: '2025-07-18'
      }
    ];
    this.setData({ albums });
  },

  loadPhotos() {
    // TODO: 实际应从云数据库加载
    const photos: Photo[] = [
      {
        id: 'p001',
        albumId: '2026spring',
        url: getPhotoUrl('2026spring', 'p001'),
        userName: '小鼠爱娃',
        userAvatar: DEFAULT_AVATAR,
        uploadTime: '2026-03-15 15:30',
        likes: 128,
        isLiked: false,
        comments: 23,
        description: '今天的娃聚太开心了！认识了好多新朋友~'
      },
      {
        id: 'p002',
        albumId: '2026spring',
        url: getPhotoUrl('2026spring', 'p002'),
        userName: '娃娃收藏家',
        userAvatar: DEFAULT_AVATAR,
        uploadTime: '2026-03-15 16:00',
        likes: 89,
        isLiked: true,
        comments: 15
      },
      {
        id: 'p003',
        albumId: '2026spring',
        url: getPhotoUrl('2026spring', 'p003'),
        userName: '摄影小白',
        userAvatar: DEFAULT_AVATAR,
        uploadTime: '2026-03-15 16:30',
        likes: 156,
        isLiked: false,
        comments: 32,
        description: '樱花和娃娃真的太配了！'
      },
      {
        id: 'p004',
        albumId: '2025winter',
        url: getPhotoUrl('2025winter', 'p004'),
        userName: 'BJD新手',
        userAvatar: DEFAULT_AVATAR,
        uploadTime: '2025-12-20 14:00',
        likes: 67,
        isLiked: false,
        comments: 8
      },
      {
        id: 'p005',
        albumId: '2025autumn',
        url: getPhotoUrl('2025autumn', 'p005'),
        userName: '手作达人',
        userAvatar: DEFAULT_AVATAR,
        uploadTime: '2025-10-05 11:00',
        likes: 234,
        isLiked: true,
        comments: 45,
        description: '自己给娃娃做的秋装，大家觉得怎么样？'
      },
      {
        id: 'p006',
        albumId: '2025summer',
        url: getPhotoUrl('2025summer', 'p006'),
        userName: '夏日阳光',
        userAvatar: DEFAULT_AVATAR,
        uploadTime: '2025-07-18 10:00',
        likes: 189,
        isLiked: false,
        comments: 28
      }
    ];
    
    this.setData({ allPhotos: photos });
  },

  selectAlbum(e: WechatMiniprogram.TouchEvent) {
    const albumId = e.currentTarget.dataset.id;
    this.setData({ currentAlbum: albumId });
    
    if (albumId !== 'all') {
      this.filterPhotos(albumId);
    }
  },

  filterPhotos(albumId: string) {
    let photos = this.data.allPhotos;
    if (albumId !== 'all') {
      photos = this.data.allPhotos.filter(p => p.albumId === albumId);
    }
    
    // 添加原始索引
    photos = photos.map((photo, index) => ({
      ...photo,
      originalIndex: index
    }));
    
    // 瀑布流分列
    const leftPhotos: Photo[] = [];
    const rightPhotos: Photo[] = [];
    
    photos.forEach((photo, index) => {
      if (index % 2 === 0) {
        leftPhotos.push(photo);
      } else {
        rightPhotos.push(photo);
      }
    });
    
    this.setData({
      currentPhotos: photos,
      leftPhotos,
      rightPhotos
    });
  },

  previewPhoto(e: WechatMiniprogram.TouchEvent) {
    const photo = e.currentTarget.dataset.photo;
    
    // 使用微信原生图片预览
    const urls = this.data.currentPhotos.map(p => p.url);
    wx.previewImage({
      current: photo.url,
      urls: urls
    });
  },

  showPhotoDetail(photo: Photo) {
    this.setData({
      currentPhotoDetail: photo,
      showPhotoDetail: true
    });
  },

  closePhotoDetail() {
    this.setData({ showPhotoDetail: false });
  },

  onPhotoDetailClose(e: WechatMiniprogram.CustomEvent) {
    if (!e.detail.visible) {
      this.setData({ showPhotoDetail: false });
    }
  },

  likePhoto(e: WechatMiniprogram.TouchEvent) {
    const photo = e.currentTarget.dataset.photo;
    const isCurrentlyLiked = photo.isLiked;
    
    // 更新照片点赞状态
    const updatePhotos = (photos: Photo[]) => {
      return photos.map(p => {
        if (p.id === photo.id) {
          return {
            ...p,
            isLiked: !isCurrentlyLiked,
            likes: isCurrentlyLiked ? p.likes - 1 : p.likes + 1
          };
        }
        return p;
      });
    };
    
    this.setData({
      leftPhotos: updatePhotos(this.data.leftPhotos),
      rightPhotos: updatePhotos(this.data.rightPhotos),
      currentPhotos: updatePhotos(this.data.currentPhotos),
      allPhotos: updatePhotos(this.data.allPhotos)
    });
    
    wx.showToast({
      title: isCurrentlyLiked ? '取消点赞' : '已点赞',
      icon: 'none'
    });
  },

  likeCurrentPhoto() {
    const photo = this.data.currentPhotoDetail;
    const isCurrentlyLiked = photo.isLiked;
    
    // 更新当前详情照片
    this.setData({
      currentPhotoDetail: {
        ...photo,
        isLiked: !isCurrentlyLiked,
        likes: isCurrentlyLiked ? photo.likes - 1 : photo.likes + 1
      }
    });
    
    // 同时更新列表中的照片
    this.likePhoto({ currentTarget: { dataset: { photo } } } as any);
  },

  showComments() {
    wx.showToast({
      title: '评论功能开发中',
      icon: 'none'
    });
  },

  sharePhoto() {
    wx.showToast({
      title: '分享功能开发中',
      icon: 'none'
    });
  },

  savePhoto() {
    wx.showLoading({ title: '保存中...' });
    
    // TODO: 实际保存逻辑
    // const url = this.data.currentPhotoDetail.url;
    // 模拟保存
    setTimeout(() => {
      wx.hideLoading();
      wx.showToast({
        title: '已保存到相册',
        icon: 'success'
      });
    }, 1000);
  },

  uploadPhoto() {
    wx.chooseMedia({
      count: 9,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: async (res) => {
        wx.showLoading({ title: '上传中...' });
        
        try {
          const albumId = this.data.currentAlbum;
          const uploadPromises = res.tempFiles.map(async (file, index) => {
            const timestamp = Date.now();
            const cloudPath = `${CLOUD_PATHS.gathering.photos}${albumId}/photo_${timestamp}_${index}.jpg`;
            return uploadPhoto(file.tempFilePath, cloudPath);
          });
          
          await Promise.all(uploadPromises);
          
          wx.hideLoading();
          wx.showToast({
            title: '上传成功',
            icon: 'success'
          });
          
          // 刷新照片列表
          this.loadPhotos();
          this.filterPhotos(albumId);
        } catch (error) {
          wx.hideLoading();
          wx.showToast({
            title: '上传失败',
            icon: 'error'
          });
          console.error('上传照片失败:', error);
        }
      }
    });
  }
});
