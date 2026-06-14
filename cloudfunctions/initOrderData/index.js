// cloudfunctions/initOrderData/index.js
// 生成测试订单：覆盖各阶段 / 加急 / 逾期 / 待审核 / 已完成
// 调用方式：wx.cloud.callFunction({ name: 'initOrderData', data: { count: 50, clear: false } })
const cloud = require('wx-server-sdk');

cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();
const orders = db.collection('orders');
const users = db.collection('users');

async function assertAdmin(openid) {
  if (!openid) return false;
  const r = await users.where({ _openid: openid, isAdmin: true }).limit(1).get();
  return r.data.length > 0;
}

const STAGES = [
  { value: 'queued',   label: '已排单', percent: 10 },
  { value: 'modeling', label: '建模',   percent: 30 },
  { value: 'painting', label: '上妆',   percent: 55 },
  { value: 'hair',     label: '假毛',   percent: 80 },
  { value: 'shipped',  label: '已发货', percent: 100 }
];

const ROLES = [
  '兔子头壳', '狐狸头壳', '猫咪头壳', '熊猫头壳', '柴犬头壳',
  '小狼头壳', '雷电将军头壳', '甘雨头壳', '胡桃头壳', '可莉头壳',
  '原神-钟离头壳', '小恶魔头壳', '天使头壳', '机娘头壳', '龙娘头壳'
];

const SURNAMES = ['张', '王', '李', '赵', '钱', '孙', '周', '吴', '郑', '陈', '林', '黄', '何', '高', '马'];
const GIVEN = ['小华', '小明', '小红', '小刚', '小光', '小丽', '雪', '阳', '萌', '诗', '雨', '婷', '帆', '宇', '欣'];

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
function randInt(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
function fmtDate(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}
function daysFromNow(n) {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return d;
}

function genOrder(i) {
  const stage = pick(STAGES);
  const orderDate = daysFromNow(-randInt(1, 90));
  const deadlineDate = daysFromNow(randInt(-15, 60)); // 一部分会是已逾期
  const isUrgent = Math.random() < 0.15;
  const isPending = Math.random() < 0.1;
  const seq = String(i + 1).padStart(3, '0');

  let status = 'normal';
  if (isPending) status = 'pending';
  else if (stage.value === 'shipped') status = 'completed';
  else if (isUrgent) status = 'urgent';

  return {
    tbOrderId: `TB${Date.now().toString().slice(-6)}${seq}`,
    queueNumber: `RatStudio-2026-${seq}`,
    customerName: pick(SURNAMES) + pick(GIVEN),
    roleName: pick(ROLES),
    orderTime: fmtDate(orderDate),
    deadline: fmtDate(deadlineDate),
    stage: isPending ? 'queued' : stage.value,
    progressStage: isPending ? '待审核' : stage.label,
    progressPercent: isPending ? 0 : stage.percent,
    status,
    isUrgent,
    isArchived: false,
    details: { color: '默认', size: '标准', notes: '' },
    createTime: new Date(orderDate.getTime() + randInt(0, 86400000)),
    updateTime: new Date()
  };
}

exports.main = async (event) => {
  const { count = 50, clear = false } = event || {};

  try {
    const { OPENID } = cloud.getWXContext();
    const isAdmin = await assertAdmin(OPENID);
    if (!isAdmin) {
      return { success: false, error: '无管理员权限' };
    }

    if (clear) {
      // 分批删除（云函数单次最多 1000 条）
      let removed = 0;
      while (true) {
        const r = await orders.limit(100).get();
        if (r.data.length === 0) break;
        await Promise.all(r.data.map(d => orders.doc(d._id).remove()));
        removed += r.data.length;
        if (r.data.length < 100) break;
      }
      console.log(`已清空 ${removed} 条旧订单`);
    }

    const list = Array.from({ length: count }, (_, i) => genOrder(i));

    // 并发插入
    const results = await Promise.allSettled(
      list.map(o => orders.add({ data: o }))
    );
    const succeeded = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.length - succeeded;

    return {
      success: true,
      requested: count,
      succeeded,
      failed,
      cleared: clear
    };
  } catch (error) {
    console.error('生成测试数据失败', error);
    return { success: false, error: error.message };
  }
};
