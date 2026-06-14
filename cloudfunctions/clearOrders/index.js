// cloudfunctions/clearOrders/index.js
// 清理订单：支持仅清理假数据 / 清空全部
// 调用：wx.cloud.callFunction({ name: 'clearOrders', data: { mode: 'mock' | 'all', confirm: 'CLEAR-ALL' } })
const cloud = require('wx-server-sdk');
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();
const orders = db.collection('orders');
const users = db.collection('users');

const BATCH = 100;

async function assertAdmin(openid) {
  if (!openid) return false;
  const r = await users.where({ _openid: openid, isAdmin: true }).limit(1).get();
  return r.data.length > 0;
}

async function removeWhere(where, label) {
  let removed = 0;
  while (true) {
    const r = await orders.where(where).limit(BATCH).get();
    if (r.data.length === 0) break;
    await Promise.all(r.data.map(d => orders.doc(d._id).remove()));
    removed += r.data.length;
    if (r.data.length < BATCH) break;
  }
  console.log(`[${label}] removed ${removed}`);
  return removed;
}

exports.main = async (event) => {
  const { mode = 'mock', confirm = '' } = event || {};

  try {
    const { OPENID } = cloud.getWXContext();
    const isAdmin = await assertAdmin(OPENID);
    if (!isAdmin) {
      return { success: false, error: '无管理员权限' };
    }

    if (mode === 'mock') {
      // 只删除 initOrderData 生成的假数据：queueNumber 以 RatStudio-2026- 开头
      const removed = await removeWhere(
        { queueNumber: db.RegExp({ regexp: '^RatStudio-2026-', options: 'i' }) },
        'mock'
      );
      return { success: true, mode, removed };
    }

    if (mode === 'all') {
      if (confirm !== 'CLEAR-ALL') {
        return { success: false, error: '危险操作需要确认码 CLEAR-ALL' };
      }
      const removed = await removeWhere({}, 'all');
      return { success: true, mode, removed };
    }

    return { success: false, error: '未知 mode，仅支持 mock / all' };
  } catch (error) {
    console.error('清理订单失败', error);
    return { success: false, error: error.message };
  }
};
