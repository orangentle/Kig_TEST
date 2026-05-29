// work-day-calc.ts
// 工期计算器：根据"开始日期 + 工作日数"算出预计完成日
// 节假日数据来自 timor.tech 公共 API，按年缓存到 storage

interface HolidayInfo {
  holiday: boolean; // true=节假日, false=调休补班
  name: string;
  wage?: number;
  date: string; // "YYYY-MM-DD"
}

interface HolidayMap {
  // key: "MM-DD"
  [k: string]: HolidayInfo;
}

interface YearCache {
  year: number;
  fetchedAt: number;
  data: HolidayMap;
}

const STORAGE_PREFIX = 'holiday_year_';

function fmt(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function parseDate(s: string): Date {
  const [y, m, d] = s.split('-').map(Number);
  return new Date(y, m - 1, d);
}

Page({
  data: {
    startDate: '',
    workDays: 30,
    today: '',
    result: null as null | {
      endDate: string;
      totalDays: number;
      weekendCount: number;
      holidayCount: number;
      makeupCount: number;
      holidayList: { date: string; name: string }[];
    },
    isCalculating: false,
    apiHint: ''
  },

  onLoad() {
    const today = fmt(new Date());
    this.setData({ startDate: today, today });
  },

  onStartDateChange(e: any) {
    this.setData({ startDate: e.detail.value, result: null });
  },

  onWorkDaysChange(e: any) {
    const v = parseInt(e.detail.value, 10);
    this.setData({ workDays: isNaN(v) ? 0 : Math.max(0, Math.min(365, v)), result: null });
  },

  async ensureHolidayData(years: number[]): Promise<HolidayMap[]> {
    const maps: HolidayMap[] = [];
    for (const year of years) {
      const cacheKey = STORAGE_PREFIX + year;
      const cached: YearCache | undefined = wx.getStorageSync(cacheKey) || undefined;
      // 30 天缓存
      if (cached && cached.data && Date.now() - cached.fetchedAt < 30 * 24 * 3600 * 1000) {
        maps.push(cached.data);
        continue;
      }
      try {
        const map = await this.fetchYear(year);
        wx.setStorageSync(cacheKey, { year, fetchedAt: Date.now(), data: map } as YearCache);
        maps.push(map);
      } catch (err) {
        console.warn(`节假日 API 失败 ${year}，使用空表`, err);
        // 失败兜底：用空表（仅按周末计算）
        maps.push({});
        this.setData({ apiHint: '节假日数据没拉到, 这次只按周末算啦~' });
      }
    }
    return maps;
  },

  fetchYear(year: number): Promise<HolidayMap> {
    return new Promise((resolve, reject) => {
      wx.request({
        url: `https://timor.tech/api/holiday/year/${year}`,
        method: 'GET',
        timeout: 8000,
        success: (res: any) => {
          if (res.statusCode === 200 && res.data && res.data.code === 0) {
            resolve(res.data.holiday as HolidayMap);
          } else {
            reject(new Error('返回格式异常'));
          }
        },
        fail: reject
      });
    });
  },

  isHoliday(date: Date, map: HolidayMap): HolidayInfo | null {
    const key = `${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
    return map[key] || null;
  },

  async onCalculate() {
    const { startDate, workDays } = this.data;
    if (!startDate || workDays <= 0) {
      wx.showToast({ title: '日期和工作日都要填哦~', icon: 'none' });
      return;
    }

    this.setData({ isCalculating: true, apiHint: '' });
    wx.showLoading({ title: '鼠鼠在算...' });

    try {
      const start = parseDate(startDate);
      // 预估范围：工作日数 * 1.6 (周末+节假日缓冲)
      const estimateDays = Math.ceil(workDays * 1.6) + 30;
      const startYear = start.getFullYear();
      const probe = new Date(start);
      probe.setDate(probe.getDate() + estimateDays);
      const endYear = probe.getFullYear();
      const yearList: number[] = [];
      for (let y = startYear; y <= endYear; y++) yearList.push(y);

      const maps = await this.ensureHolidayData(yearList);
      const yearToMap: Record<number, HolidayMap> = {};
      yearList.forEach((y, i) => { yearToMap[y] = maps[i]; });
      const mergedFind = (d: Date): HolidayInfo | null => {
        const map = yearToMap[d.getFullYear()];
        return map ? this.isHoliday(d, map) : null;
      };

      let cursor = new Date(start);
      let workCounted = 0;
      let totalDays = 0;
      let weekendCount = 0;
      let holidayCount = 0;
      let makeupCount = 0;
      const holidayList: { date: string; name: string }[] = [];

      // 第一天若是工作日也计数；开始日 = 第 1 个工作日
      while (workCounted < workDays) {
        const dow = cursor.getDay(); // 0=Sun, 6=Sat
        const isWeekend = dow === 0 || dow === 6;
        const info = mergedFind(cursor);

        let isWorkDay: boolean;
        if (info && info.holiday) {
          // 法定节假日
          isWorkDay = false;
          holidayCount += 1;
          holidayList.push({ date: fmt(cursor), name: info.name });
        } else if (info && info.holiday === false) {
          // 调休补班 → 强制工作日
          isWorkDay = true;
          makeupCount += 1;
        } else if (isWeekend) {
          isWorkDay = false;
          weekendCount += 1;
        } else {
          isWorkDay = true;
        }

        if (isWorkDay) workCounted += 1;
        totalDays += 1;
        if (workCounted >= workDays) break;
        cursor.setDate(cursor.getDate() + 1);
      }

      this.setData({
        result: {
          endDate: fmt(cursor),
          totalDays,
          weekendCount,
          holidayCount,
          makeupCount,
          holidayList
        }
      });
    } catch (err) {
      console.error(err);
      wx.showToast({ title: '算挂了,再试试?', icon: 'none' });
    } finally {
      wx.hideLoading();
      this.setData({ isCalculating: false });
    }
  },

  onCopyResult() {
    if (!this.data.result) return;
    const r = this.data.result;
    const text = `工期估算\n开始: ${this.data.startDate}\n工作日: ${this.data.workDays}\n预计完成: ${r.endDate}\n共 ${r.totalDays} 自然日 (含 ${r.weekendCount} 个周末日 + ${r.holidayCount} 个节假日)`;
    wx.setClipboardData({
      data: text,
      success: () => wx.showToast({ title: '复制好啦~', icon: 'success' })
    });
  },

  onBack() {
    wx.navigateBack();
  }
});
