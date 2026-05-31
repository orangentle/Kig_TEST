// 鼠鼠工坊 · 表单校验工具
// 所有 validate* 返回 { ok: true } 或 { ok: false, msg: '提示语' }

export interface ValidateResult {
  ok: boolean;
  msg?: string;
}

const ok: ValidateResult = { ok: true };
const fail = (msg: string): ValidateResult => ({ ok: false, msg });

// ---- 工具 ----
const isBlank = (v: any) => v === undefined || v === null || String(v).trim() === '';
const len = (v: string) => Array.from(String(v).trim()).length; // 中文按 1 个字算

// 危险/无意义字符：控制字符、零宽字符、双向文本覆盖、BOM
// eslint-disable-next-line no-control-regex
const BAD_CHARS = /[\x00-\x1F\x7F\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]/;

/** 文本中是否含控制字符 / 零宽 / RTL 覆盖等危险 unicode */
function hasBadChars(v: any): boolean {
  return BAD_CHARS.test(String(v));
}

// ---- 单字段校验 ----

/** QQ：选填；如填则必须 5-11 位纯数字 */
export function validateQQ(v: any, required = false): ValidateResult {
  if (isBlank(v)) return required ? fail('QQ 还没填呢，鼠鼠联系不到你呀~') : ok;
  if (!/^[1-9]\d{4,10}$/.test(String(v).trim())) {
    return fail('鼠鼠看不懂这个 QQ 号呜...要 5-11 位数字喵~');
  }
  return ok;
}

/** 淘宝订单号：必填；10-20 位纯数字；非全相同数字；首位非 0 */
export function validateTbOrderId(v: any): ValidateResult {
  if (isBlank(v)) return fail('鼠鼠找不到淘宝订单号呜呜~ 先填一下嘛 (｡>﹏<｡)');
  const s = String(v).trim();
  if (!/^[1-9]\d{9,19}$/.test(s)) {
    return fail('这单号鼠鼠不认识呢...淘宝订单号是一长串数字喵 (10-20 位、首位非 0)~');
  }
  if (/^(\d)\1+$/.test(s)) {
    return fail('鼠鼠数了下...这单号好像在糊弄人喵 (≧へ≦)');
  }
  return ok;
}

/** 手机号：选填；如填则 11 位 1[3-9] 开头 */
export function validatePhone(v: any, required = false): ValidateResult {
  if (isBlank(v)) return required ? fail('手机号还没填，鼠鼠拨不通的~') : ok;
  if (!/^1[3-9]\d{9}$/.test(String(v).trim())) {
    return fail('这个号码鼠鼠拨不通喵，麻烦填 11 位真手机号~');
  }
  return ok;
}

/** 昵称 / 客户名：1-20 字 */
export function validateName(v: any, label = '昵称', required = true, max = 20): ValidateResult {
  if (isBlank(v)) return required ? fail(`鼠鼠还不知道怎么叫你呢，填个${label}吧~`) : ok;
  if (hasBadChars(v)) return fail(`${label}里混进了奇怪的看不见的字符喵，重新输入一下吧~`);
  const n = len(v);
  if (n < 1) return fail(`${label}至少留 1 个字喵`);
  if (n > max) return fail(`${label}长得鼠鼠记不住啦（>﹏<），最多 ${max} 个字喵`);
  return ok;
}

/** 淘宝名：选填，最多 30 字 */
export function validateTaobaoName(v: any): ValidateResult {
  if (isBlank(v)) return ok;
  if (hasBadChars(v)) return fail('淘宝名里有奇怪字符，鼠鼠认不得喵~');
  if (len(v) > 30) return fail('淘宝名太长啦，鼠鼠抄不下来，最多 30 个字喵~');
  return ok;
}

/** 数值范围（身材数据） */
export function validateRange(v: any, label: string, min: number, max: number, required = false): ValidateResult {
  if (isBlank(v) || Number(v) === 0) {
    return required ? fail(`${label}还没填呢喵~`) : ok;
  }
  const num = Number(v);
  if (!Number.isFinite(num)) return fail(`${label}填得怪怪的，鼠鼠看不懂喵~`);
  if (num < min || num > max) {
    return fail(`${label}超出鼠鼠的尺子啦，正常值 ${min}–${max} 之间喵~`);
  }
  return ok;
}

/** 文本长度（备注、IP 等） */
export function validateText(v: any, label: string, max: number, required = false): ValidateResult {
  if (isBlank(v)) return required ? fail(`${label}还没填呢~`) : ok;
  if (hasBadChars(v)) return fail(`${label}里有奇怪字符，鼠鼠看不懂喵~`);
  if (len(v) > max) return fail(`${label}太长啦，最多 ${max} 个字喵~`);
  return ok;
}

// ---- 组合校验：联系方式 + 身材数据 ----

export interface ContactBodyForm {
  qq?: any;
  phone?: any;
  taobaoName?: any;
  height?: any;
  weight?: any;
  headCircumference?: any;
  shoulderWidth?: any;
}

/**
 * 一次性校验联系方式 + 身材数据，返回首个失败项。
 * heightRequired/headRequired 用于下单场景（必填），编辑资料时传 false。
 */
export function validateContactAndBody(
  form: ContactBodyForm,
  opts: { heightRequired?: boolean; headRequired?: boolean } = {}
): ValidateResult {
  const checks: ValidateResult[] = [
    validateQQ(form.qq),
    validatePhone(form.phone),
    validateTaobaoName(form.taobaoName),
    validateRange(form.height, '身高', 50, 250, !!opts.heightRequired),
    validateRange(form.weight, '体重', 20, 300, false),
    validateRange(form.headCircumference, '头围', 30, 80, !!opts.headRequired),
    validateRange(form.shoulderWidth, '肩宽', 20, 80, false),
  ];
  for (const r of checks) if (!r.ok) return r;
  return ok;
}

/** 弹 toast 的便捷封装 */
export function toastIfFail(r: ValidateResult): boolean {
  if (r.ok) return true;
  wx.showToast({ title: r.msg || '填写有误喵~', icon: 'none', duration: 2200 });
  return false;
}
