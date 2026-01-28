/**
 * 云存储工具函数
 * 用于管理娃聚模块的图片资源
 * 
 * 云存储目录结构:
 * 
 * gathering/
 * ├── okr0.0/                    # OKR0.0启动聚 (2025-10-01)
 * │   ├── cover.jpg              # 活动封面
 * │   └── photos/                # 活动照片
 * │       ├── p001.jpg
 * │       └── ...
 * ├── ssca2026/                  # 沈阳SSCA漫展小聚 (2026-02-06~08)
 * │   ├── cover.jpg
 * │   └── photos/
 * │       └── ...
 * ├── okr1.0/                    # OKR1.0聚会 (2026-05-02~03)
 * │   ├── cover.jpg
 * │   └── photos/
 * │       └── ...
 * ├── banners/                   # 娃聚主页轮播图
 * │   ├── banner_001.jpg
 * │   └── ...
 * └── merch/                     # 聚会周边商品
 *     ├── banners/               # 周边商城轮播图
 *     │   ├── banner_001.jpg
 *     │   └── ...
 *     └── products/              # 商品图片
 *         ├── prod_001/
 *         │   ├── main.jpg
 *         │   ├── detail_1.jpg
 *         │   └── ...
 *         └── ...
 * 
 * 注：用户头像使用个人中心的头像，不内额外存储
 */

const app = getApp<IAppOption>();

// 活动ID映射
export const EVENT_IDS = {
  'event_001': 'okr0.0',    // OKR0.0启动聚
  'event_002': 'ssca2026',  // 沈阳SSCA漫展小聚
  'event_003': 'okr1.0',    // OKR1.0聚会
};

// 云存储路径配置
export const CLOUD_PATHS = {
  // 娃聚模块
  gathering: {
    banners: 'gathering/banners/',           // 娃聚主页轮播图
    // 活动目录
    events: {
      'okr0.0': 'gathering/okr0.0/',         // OKR0.0启动聚
      'ssca2026': 'gathering/ssca2026/',     // 沈阳SSCA漫展小聚  
      'okr1.0': 'gathering/okr1.0/',         // OKR1.0聚会
    },
    // 聚会周边
    merch: {
      banners: 'gathering/merch/banners/',   // 周边商城轮播图
      products: 'gathering/merch/products/', // 商品图片
    },
  },
};

/**
 * 获取完整的云存储路径
 * @param relativePath 相对路径，如 'gathering/okr0.0/cover.jpg'
 * @returns 完整的云存储路径
 */
export function getCloudPath(relativePath: string): string {
  const base = app.globalData.cloudStorageBase;
  // 确保路径格式正确
  const cleanPath = relativePath.startsWith('/') ? relativePath.slice(1) : relativePath;
  return `${base}/${cleanPath}`;
}

/**
 * 根据活动ID获取活动目录名
 * @param eventId 活动ID (如 'event_001')
 * @returns 活动目录名 (如 'okr0.0')
 */
export function getEventFolderName(eventId: string): string {
  return EVENT_IDS[eventId as keyof typeof EVENT_IDS] || eventId;
}

/**
 * 获取娃聚活动封面图路径
 * @param eventId 活动ID
 * @param ext 文件扩展名，默认jpg
 */
export function getEventCoverUrl(eventId: string, ext: string = 'jpg'): string {
  const folderName = getEventFolderName(eventId);
  return getCloudPath(`gathering/${folderName}/cover.${ext}`);
}

/**
 * 获取活动相册封面图路径（使用活动封面）
 * @param eventId 活动ID
 * @param ext 文件扩展名，默认jpg
 */
export function getAlbumCoverUrl(eventId: string, ext: string = 'jpg'): string {
  return getEventCoverUrl(eventId, ext);
}

/**
 * 获取活动照片路径
 * @param eventId 活动ID
 * @param photoId 照片ID
 * @param ext 文件扩展名，默认jpg
 */
export function getPhotoUrl(eventId: string, photoId: string, ext: string = 'jpg'): string {
  const folderName = getEventFolderName(eventId);
  return getCloudPath(`gathering/${folderName}/photos/${photoId}.${ext}`);
}

/**
 * 获取商品主图路径
 * @param productId 商品ID
 * @param ext 文件扩展名，默认jpg
 */
export function getProductMainUrl(productId: string, ext: string = 'jpg'): string {
  return getCloudPath(`${CLOUD_PATHS.gathering.merch.products}${productId}/main.${ext}`);
}

/**
 * 获取商品详情图路径
 * @param productId 商品ID
 * @param index 图片序号
 * @param ext 文件扩展名，默认jpg
 */
export function getProductDetailUrl(productId: string, index: number, ext: string = 'jpg'): string {
  return getCloudPath(`${CLOUD_PATHS.gathering.merch.products}${productId}/detail_${index}.${ext}`);
}

/**
 * 获取娃聚Banner图路径
 * @param bannerId banner ID
 * @param ext 文件扩展名，默认jpg
 */
export function getGatheringBannerUrl(bannerId: string, ext: string = 'jpg'): string {
  return getCloudPath(`${CLOUD_PATHS.gathering.banners}${bannerId}.${ext}`);
}

/**
 * 获取商城Banner图路径
 * @param bannerId banner ID
 * @param ext 文件扩展名，默认jpg
 */
export function getMerchBannerUrl(bannerId: string, ext: string = 'jpg'): string {
  return getCloudPath(`${CLOUD_PATHS.gathering.merch.banners}${bannerId}.${ext}`);
}

/**
 * 上传照片到云存储
 * @param filePath 本地文件路径
 * @param cloudPath 云存储相对路径
 * @returns Promise<string> 云文件ID
 */
export async function uploadPhoto(filePath: string, cloudPath: string): Promise<string> {
  try {
    const result = await wx.cloud.uploadFile({
      cloudPath: cloudPath,
      filePath: filePath,
    });
    return result.fileID;
  } catch (error) {
    console.error('上传文件失败:', error);
    throw error;
  }
}

/**
 * 获取临时访问链接（用于分享等场景）
 * @param fileID 云文件ID
 * @returns Promise<string> 临时链接
 */
export async function getTempFileUrl(fileID: string): Promise<string> {
  try {
    const result = await wx.cloud.getTempFileURL({
      fileList: [fileID],
    });
    if (result.fileList && result.fileList[0]) {
      return result.fileList[0].tempFileURL;
    }
    throw new Error('获取临时链接失败');
  } catch (error) {
    console.error('获取临时链接失败:', error);
    throw error;
  }
}

// 默认占位图（当云存储图片加载失败时使用）
export const DEFAULT_IMAGES = {
  event: 'https://picsum.photos/seed/default_event/400/300',
  album: 'https://picsum.photos/seed/default_album/400/300',
  photo: 'https://picsum.photos/seed/default_photo/400/400',
  product: 'https://picsum.photos/seed/default_product/400/400',
  banner: 'https://picsum.photos/seed/default_banner/750/300',
  avatar: 'https://mmbiz.qpic.cn/mmbiz/icTdbqWNOwNRna42FI242Lcia07jQodd2FJGIYQfG0LAJGFxM4FbnQP6yfMxBgJ0F3YRqJCJ1aPAK2dQagdusBZg/0',
};
