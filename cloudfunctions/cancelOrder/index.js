// cloudfunctions/cancelOrder/index.js
// 用户主动取消订单：仅本人 + 仅 pending（待审核）状态可取消
const cloud = require('wx-server-sdk');

cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();
const ordersCollection = db.collection('orders');

exports.main = async (event) => {
  try {
    const { OPENID } = cloud.getWXContext();
    const { tbOrderId, reason } = event || {};

    if (!OPENID) {
      return { success: false, error: '咦~ 鼠鼠认不出你呀,先登录嘛 (｡>﹏<｡)' };
    }
    if (!tbOrderId) {
      return { success: false, error: '订单号丢了呜呜~ 刷新一下再试吧' };
    }

    const r = await ordersCollection
      .where({ tbOrderId, _openid: OPENID })
      .limit(1)
      .get();

    if (!r.data.length) {
      return { success: false, error: '找不到这单呢~ 是不是不是你的订单呀? (◍•ᴗ•◍)' };
    }

    const order = r.data[0];
    if (order.status !== 'pending') {
      return { success: false, error: '这单已经在赶工啦~ 想取消要联系客服喔 ♡' };
    }
    if (order.isLocked) {
      return { success: false, error: '订单被锁住啦~ 联系客服才能解锁哦 (´･ω･`)' };
    }

    await ordersCollection.doc(order._id).update({
      data: {
        status: 'canceled',
        stage: 'pending',
        progressStage: '已取消',
        progressPercent: 0,
        cancelInfo: {
          cancelTime: new Date(),
          cancelBy: OPENID,
          cancelReason: (reason || '用户主动取消').slice(0, 100)
        },
        updateTime: new Date()
      }
    });

    return { success: true };
  } catch (error) {
    console.error('取消订单失败', error);
    return { success: false, error: error.message };
  }
};
