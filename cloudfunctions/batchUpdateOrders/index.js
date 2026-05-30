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

// 订阅消息模板 ID（在 https://mp.weixin.qq.com → 订阅消息 → 我的模板 申请后填入）
// 已排单：模板「排单成功通知」，字段 name3(付款人名) / thing5(温馨提示)
// 已发货：模板「账单通知」，字段 thing14(开单人) / thing25(订单内容)
const QUEUED_TMPL_ID   = 'hNOQznZnaw3VR7Bcnmnv0HJBO1F8P_kxIGpIACi3VTk';
const SHIPPING_TMPL_ID = 'rJMkotC0cffPQfvmFi_kwhfFcCo5fQcmFfa3ai_xkFk';

function isPlaceholder(id) {
  return !id || id.startsWith('REPLACE_');
}

async function getNickName(openid) {
  try {
    const r = await usersCollection
      .where({ _openid: openid })
      .field({ nickName: true })
      .limit(1)
      .get();
    const name = r.data[0] && r.data[0].nickName;
    return (name || '客户').slice(0, 10);
  } catch (_) {
    return '客户';
  }
}

async function sendQueuedNotice(item) {
  if (!item._openid || isPlaceholder(QUEUED_TMPL_ID)) return;
  try {
    const nickName = await getNickName(item._openid);
    await cloud.openapi.subscribeMessage.send({
      touser: item._openid,
      templateId: QUEUED_TMPL_ID,
      page: `pages/order-detail/order-detail?id=${item.orderId || ''}`,
      data: {
        name3:  { value: nickName },
        thing5: { value: '审核通过已进入排单' }
      }
    });
  } catch (err) {
    console.warn('已排单通知发送失败', item._openid, err && err.errMsg);
  }
}

async function sendShippingNotice(item) {
  if (!item._openid || isPlaceholder(SHIPPING_TMPL_ID)) return;
  try {
    await cloud.openapi.subscribeMessage.send({
      touser: item._openid,
      templateId: SHIPPING_TMPL_ID,
      page: `pages/order-detail/order-detail?id=${item.orderId || ''}`,
      data: {
        thing14: { value: '鼠鼠工坊' },
        thing25: { value: ((item.roleName || '头壳') + ' 已发货').slice(0, 20) }
      }
    });
  } catch (err) {
    console.warn('发货通知发送失败', item._openid, err && err.errMsg);
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

      case 'review-approve': {
        // 审核通过：进入"已排单"，可正常生产
        const items = await ordersCollection
          .where({ _id: _.in(orderIds) })
          .field({ stage: true, _openid: true, orderId: true, roleName: true })
          .get();
        perItemUpdates = items.data.map(item => ({
          _id: item._id,
          _openid: item._openid,
          orderId: item.orderId,
          roleName: item.roleName,
          enteredQueued: item.stage !== 'queued',
          data: {
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
          }
        }));
        break;
      }

      case 'review-reject': {
        // 驳回 = 直接删除订单(防止假单污染 + 释放 tbOrderId 唯一索引槽位)
        // 同时把驳回原因塞进用户文档,客户下次打开个人中心会看到引导弹窗
        const remark = (payload && payload.remark) || '信息有误,请按提示重新下单';
        const items = await ordersCollection
          .where({ _id: _.in(orderIds) })
          .field({ _openid: true, tbOrderId: true, roleName: true, orderId: true })
          .get();

        // 1) 写驳回通知到用户文档(数组,允许多条堆积)
        const notices = items.data.map(it => ({
          tbOrderId: it.tbOrderId || '',
          roleName: it.roleName || '',
          orderId: it.orderId || '',
          reason: remark,
          rejectTime: now
        }));
        await Promise.all(items.data.map((it, idx) =>
          usersCollection.where({ _openid: it._openid }).update({
            data: { pendingRejectNotices: _.push([notices[idx]]) }
          }).catch(err => console.warn('写驳回通知失败', it._openid, err))
        ));

        // 2) 删除订单
        const results = await Promise.all(items.data.map(it =>
          ordersCollection.doc(it._id).remove()
            .then(() => true)
            .catch(err => { console.error('reject-delete fail', it._id, err); return false; })
        ));
        const ok = results.filter(Boolean).length;
        return {
          success: true,
          action,
          requested: orderIds.length,
          succeeded: ok,
          failed: orderIds.length - ok
        };
      }

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

      // 已排单通知：审核通过且确实是从非 queued 转入 queued 时推送
      if (action === 'review-approve') {
        const queued = perItemUpdates.filter(u => u.enteredQueued);
        await Promise.all(queued.map(u => sendQueuedNotice(u))).catch(e => console.warn('已排单通知发送失败', e));
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
