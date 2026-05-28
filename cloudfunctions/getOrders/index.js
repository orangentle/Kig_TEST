// cloudfunctions/getOrders/index.js
// 支持分页、筛选、排序、关键字搜索的订单查询
const cloud = require('wx-server-sdk');

cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();
const _ = db.command;
const ordersCollection = db.collection('orders');
const usersCollection = db.collection('users');

async function isAdmin(openid) {
  if (!openid) return false;
  try {
    const r = await usersCollection.where({ _openid: openid, isAdmin: true }).limit(1).get();
    return r.data.length > 0;
  } catch (_) { return false; }
}

exports.main = async (event) => {
  try {
    const { OPENID } = cloud.getWXContext();
    const admin = await isAdmin(OPENID);

    const {
      page = 1,
      pageSize = 20,
      stage,              // 单个阶段或数组
      status,             // urgent/normal/soon/pending/...
      keyword,            // 搜索 tbOrderId / customerName / roleName
      sortBy = 'createTime',
      sortOrder = 'desc',
      dateFrom,           // 下单时间 >=
      dateTo,             // 下单时间 <=
      isArchived = false,
      isUrgent,           // 加急
      overdueOnly = false, // 是否仅查逾期
      tbOrderId,          // 兼容旧调用：按订单号查单个
      countOnly = false   // 仅返回统计（用于概览卡片）
    } = event;

    // 单订单查询（兼容 order-detail 的旧用法）
    if (tbOrderId) {
      const where = admin ? { tbOrderId } : { tbOrderId, _openid: OPENID };
      const r = await ordersCollection.where(where).get();
      return { success: true, data: r.data, total: r.data.length };
    }

    // 非管理员调用列表接口（除单订单外）一律拒绝，防止越权
    if (!admin) {
      return { success: false, error: '无管理员权限' };
    }

    // 构建查询条件
    const condition = { isArchived };

    if (stage) {
      condition.stage = Array.isArray(stage) ? _.in(stage) : stage;
    }
    if (status) {
      condition.status = Array.isArray(status) ? _.in(status) : status;
    }
    if (typeof isUrgent === 'boolean') {
      condition.isUrgent = isUrgent;
    }
    if (dateFrom || dateTo) {
      const r = {};
      if (dateFrom) r.gte = dateFrom;
      if (dateTo) r.lte = dateTo;
      condition.createTime = _.and(
        dateFrom ? _.gte(new Date(dateFrom)) : _.exists(true),
        dateTo ? _.lte(new Date(dateTo)) : _.exists(true)
      );
    }
    if (overdueOnly) {
      condition.deadline = _.lt(new Date());
      condition.stage = _.neq('shipped');
    }

    let query = ordersCollection.where(condition);

    // 关键字搜索（云数据库不支持 OR + 正则混合，分别查再合并）
    if (keyword && keyword.trim()) {
      const kw = keyword.trim();
      const regex = db.RegExp({ regexp: kw, options: 'i' });
      // 用 _.or 组合多字段模糊
      query = ordersCollection.where(
        _.and([
          condition,
          _.or([
            { tbOrderId: regex },
            { customerName: regex },
            { roleName: regex },
            { queueNumber: regex }
          ])
        ])
      );
    }

    // 仅统计
    if (countOnly) {
      const c = await query.count();
      return { success: true, total: c.total };
    }

    // 分页 + 排序
    const skip = Math.max(0, (page - 1) * pageSize);
    const limit = Math.min(pageSize, 100);

    const [listRes, countRes] = await Promise.all([
      query
        .orderBy(sortBy, sortOrder === 'asc' ? 'asc' : 'desc')
        .skip(skip)
        .limit(limit)
        .get(),
      query.count()
    ]);

    return {
      success: true,
      data: listRes.data,
      total: countRes.total,
      page,
      pageSize: limit,
      hasMore: skip + listRes.data.length < countRes.total
    };
  } catch (error) {
    console.error('获取订单列表失败', error);
    return { success: false, error: error.message };
  }
};
