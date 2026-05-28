// cloudfunctions/batchUpdateOrders/index.js
// 批量更新订单：推进阶段 / 设/取消加急 / 归档 / 分配排单号 / 自定义字段
const cloud = require('wx-server-sdk');

cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();
const _ = db.command;
const ordersCollection = db.collection('orders');
const usersCollection = db.collection('users');

const STAGE_FLOW = [
  { value: 'queued',   label: '已排单', percent: 10 },
  { value: 'modeling', label: '建模',   percent: 30 },
  { value: 'painting', label: '上妆',   percent: 55 },
  { value: 'hair',     label: '假毛',   percent: 80 },
  { value: 'shipped',  label: '已发货', percent: 100 }
];

async function assertAdmin(openid) {
  if (!openid) return false;
  const r = await usersCollection.where({ _openid: openid, isAdmin: true }).limit(1).get();
  return r.data.length > 0;
}

function nextStage(current) {
  const i = STAGE_FLOW.findIndex(s => s.value === current);
  if (i < 0 || i >= STAGE_FLOW.length - 1) return null;
  return STAGE_FLOW[i + 1];
}

function findStage(value) {
  return STAGE_FLOW.find(s => s.value === value);
}

// 发货通知（简易版）。tmplId 需在小程序后台申请「发货通知」订阅消息模板后填入
const SHIPPING_TMPL_ID = 'REPLACE_WITH_SHIPPING_TMPL_ID';
async function sendShippingNotice(item) {
  if (!item._openid || !SHIPPING_TMPL_ID || SHIPPING_TMPL_ID.startsWith('REPLACE_')) return;
  try {
    await cloud.openapi.subscribeMessage.send({
      touser: item._openid,
      templateId: SHIPPING_TMPL_ID,
      page: `pages/order-detail/order-detail?id=${item.orderId || ''}`,
      data: {
        thing1: { value: (item.roleName || '您的头壳').slice(0, 20) },
        thing2: { value: '已发货，请注意查收' }
      }
    });
  } catch (err) {
    console.warn('subscribeMessage.send failed', item._openid, err && err.errMsg);
  }
}

exports.main = async (event, context) => {
  try {
    const { OPENID } = cloud.getWXContext();
    const isAdmin = await assertAdmin(OPENID);
    if (!isAdmin) {
      return { success: false, error: '无管理员权限' };
    }

    const { orderIds = [], action, payload = {} } = event;
    if (!Array.isArray(orderIds) || orderIds.length === 0) {
      return { success: false, error: '未选择订单' };
    }
    if (orderIds.length > 500) {
      return { success: false, error: '单次最多 500 条' };
    }

    const now = new Date();
    let updateData = null;
    let perItemUpdates = null; // 当不同订单需要不同更新值时

    switch (action) {
      case 'advance-stage': {
        // 把每条订单往后推一格
        const items = await ordersCollection
          .where({ _id: _.in(orderIds) })
          .field({ stage: true, _openid: true, orderId: true, roleName: true })
          .get();
        perItemUpdates = items.data.map(item => {
          const next = nextStage(item.stage);
          if (!next) return null;
          return {
            _id: item._id,
            _openid: item._openid,
            orderId: item.orderId,
            roleName: item.roleName,
            nextStage: next,
            data: {
              stage: next.value,
              progressStage: next.label,
              progressPercent: next.percent,
              status: next.value === 'shipped' ? 'completed' : undefined,
              updateTime: now
            }
          };
        }).filter(Boolean);
        break;
      }

      case 'set-stage': {
        const target = findStage(payload.stage);
        if (!target) return { success: false, error: '无效阶段' };
        updateData = {
          stage: target.value,
          progressStage: target.label,
          progressPercent: target.percent,
          updateTime: now
        };
        break;
      }

      case 'mark-urgent':
        updateData = { isUrgent: true, status: 'urgent', updateTime: now };
        break;

      case 'review-approve':
        // 审核通过：进入"已排单"，可正常生产
        updateData = {
          status: 'normal',
          stage: 'queued',
          progressStage: '已排单',
          progressPercent: 10,
          reviewInfo: {
            reviewTime: now,
            reviewBy: OPENID,
            reviewRemark: (payload && payload.remark) || '审核通过'
          },
          updateTime: now
        };
        break;

      case 'review-reject':
        // 驳回：标记 rejected，仍锁定，需要客户改单后由客服重新提交
        updateData = {
          status: 'rejected',
          stage: 'pending',
          progressStage: '已驳回',
          progressPercent: 0,
          reviewInfo: {
            reviewTime: now,
            reviewBy: OPENID,
            reviewRemark: (payload && payload.remark) || '信息有误，请联系客服'
          },
          updateTime: now
        };
        break;

      case 'unmark-urgent':
        updateData = { isUrgent: false, status: 'normal', updateTime: now };
        break;

      case 'archive':
        updateData = { isArchived: true, updateTime: now };
        break;

      case 'unarchive':
        updateData = { isArchived: false, updateTime: now };
        break;

      case 'lock':
        updateData = { isLocked: true, lockedAt: now, updateTime: now };
        break;

      case 'unlock':
        updateData = { isLocked: false, lockedAt: null, updateTime: now };
        break;

      case 'assign-queue': {
        // 自动分配排单号：RatStudio-YYYY-NNN（起始序号由 payload.startSeq 或当前最大值 + 1）
        const year = new Date().getFullYear();
        const prefix = `RatStudio-${year}-`;

        let seq = payload.startSeq;
        if (!seq) {
          const latest = await ordersCollection
            .where({ queueNumber: db.RegExp({ regexp: `^${prefix}`, options: 'i' }) })
            .orderBy('queueNumber', 'desc')
            .limit(1)
            .get();
          if (latest.data.length > 0) {
            const m = (latest.data[0].queueNumber || '').match(/-(\d+)$/);
            seq = m ? parseInt(m[1]) + 1 : 1;
          } else {
            seq = 1;
          }
        }

        perItemUpdates = orderIds.map((id, idx) => ({
          _id: id,
          data: {
            queueNumber: `${prefix}${String(seq + idx).padStart(3, '0')}`,
            updateTime: now
          }
        }));
        break;
      }

      case 'set-fields':
        if (!payload.fields || typeof payload.fields !== 'object') {
          return { success: false, error: '缺少 fields' };
        }
        updateData = { ...payload.fields, updateTime: now };
        break;

      case 'delete': {
        // 删除选中订单（单条或批量）
        const results = await Promise.all(
          orderIds.map(id =>
            ordersCollection.doc(id).remove()
              .then(() => true)
              .catch(err => { console.error('delete fail', id, err); return false; })
          )
        );
        const ok = results.filter(Boolean).length;
        return {
          success: true,
          action,
          requested: orderIds.length,
          succeeded: ok,
          failed: orderIds.length - ok
        };
      }

      default:
        return { success: false, error: '未知操作: ' + action };
    }

    // 执行更新
    let succeeded = 0;
    let failed = 0;

    if (updateData) {
      // 清理 undefined
      Object.keys(updateData).forEach(k => updateData[k] === undefined && delete updateData[k]);
      const r = await ordersCollection
        .where({ _id: _.in(orderIds) })
        .update({ data: updateData });
      succeeded = r.stats.updated;
      failed = orderIds.length - succeeded;
    } else if (perItemUpdates && perItemUpdates.length > 0) {
      const results = await Promise.all(
        perItemUpdates.map(u =>
          ordersCollection.doc(u._id).update({ data: u.data })
            .then(() => true)
            .catch(err => { console.error('update fail', u._id, err); return false; })
        )
      );
      succeeded = results.filter(Boolean).length;
      failed = results.length - succeeded;

      // 发货通知：对推进到 shipped 的订单发送订阅消息
      if (action === 'advance-stage') {
        const shipped = perItemUpdates.filter(u => u.nextStage && u.nextStage.value === 'shipped');
        await Promise.all(shipped.map(u => sendShippingNotice(u))).catch(e => console.warn('发货通知发送失败', e));
      }
    }

    return {
      success: true,
      action,
      requested: orderIds.length,
      succeeded,
      failed
    };
  } catch (error) {
    console.error('批量更新失败', error);
    return { success: false, error: error.message };
  }
};
