// pages/gathering/schedule/schedule.ts

interface CalendarDay {
  date: string;
  day: number;
  isCurrentMonth: boolean;
  isToday: boolean;
  hasEvent: boolean;
  isSelected: boolean;
}

interface AgendaItem {
  time: string;
  content: string;
}

interface ScheduleEvent {
  id: string;
  name: string;
  date: string;
  dateText: string;
  monthText: string;
  dayNum: number;
  weekday: string;
  startTime: string;
  endTime: string;
  location: string;
  host?: string;
  type: 'gathering' | 'photo' | 'workshop' | 'market';
  typeText: string;
  description?: string;
  agenda?: AgendaItem[];
  canSignup: boolean;
}

Page({
  data: {
    currentYear: new Date().getFullYear(),
    currentMonth: new Date().getMonth() + 1,
    weekdays: ['日', '一', '二', '三', '四', '五', '六'],
    calendarDays: [] as CalendarDay[],
    selectedDate: '',
    selectedDateText: '今天',
    dayEvents: [] as ScheduleEvent[],
    upcomingEvents: [] as ScheduleEvent[],
    allEvents: [] as ScheduleEvent[],
    showDetail: false,
    currentEvent: {} as ScheduleEvent
  },

  onLoad() {
    this.loadEvents();
    this.generateCalendar();
    this.selectToday();
    this.loadUpcomingEvents();
  },

  goBack() {
    wx.navigateBack();
  },

  loadEvents() {
    // 活动数据
    const mockEvents: ScheduleEvent[] = [
      {
        id: 'event_001',
        name: 'OKR0.0启动聚',
        date: '2025-10-01',
        dateText: '2025年10月1日',
        monthText: '10月',
        dayNum: 1,
        weekday: '周三',
        startTime: '09:00',
        endTime: '17:00',
        location: '沈阳乌托邦聚会别墅V4和V5',
        host: '偶壳OKR',
        type: 'gathering',
        typeText: '娃聚',
        description: '偶壳OKR首次启动聚会！100位娃友齐聚一堂，共同见证OKR社区的诞生。',
        agenda: [
          { time: '10:00-11:30', content: '入场签到 & 穿戴Kig准备\n签到位置：V4一层门口休息厅A（Switch区）主签到处发放胸牌\n更衣安排：V4二层更衣区，人多时V4一层卧室、V5卧室也可更衣' },
          { time: '11:30-14:30', content: '自由交流集邮、直播互动' },
          { time: '14:30-15:00', content: 'Bingo游戏抽奖\n发放Bingo卡片，由D100骰子投出奖品\n奖池：10个鼠鼠抱枕、10个吉吉抱枕、1个鼠鼠U类1000元优惠券、3个鼠鼠U类500元优惠券、1个秀吉姬1000元优惠券、3个秀吉姬500元优惠券' },
          { time: '15:00-16:00', content: '大合照 & 集体互动拍照\n合影地点：室外露营地' },
          { time: '16:00-17:00', content: '自由活动 & 散场准备' }
        ],
        canSignup: false
      },
      {
        id: 'event_003',
        name: 'OKR1.0聚会',
        date: '2026-05-02',
        dateText: '2026年5月2日-3日（暂定）',
        monthText: '5月',
        dayNum: 2,
        weekday: '周六-周日',
        startTime: '待定',
        endTime: '待定',
        location: '沈阳棋盘山绿地铂瑞酒店',
        host: '偶壳OKR',
        type: 'gathering',
        typeText: '娃聚',
        description: 'OKR1.0正式聚会！地点定在风景优美的沈阳棋盘山绿地铂瑞酒店，人数不限。',
        agenda: [
          { time: '待定', content: '活动安排敬请期待...' }
        ],
        canSignup: false
      }
    ];
    
    this.setData({ allEvents: mockEvents });
  },

  generateCalendar() {
    const { currentYear, currentMonth } = this.data;
    const firstDay = new Date(currentYear, currentMonth - 1, 1);
    const lastDay = new Date(currentYear, currentMonth, 0);
    const startWeekday = firstDay.getDay();
    const daysInMonth = lastDay.getDate();
    
    const today = new Date();
    const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    
    const calendarDays: CalendarDay[] = [];
    
    // 上个月的日期
    const prevMonth = currentMonth === 1 ? 12 : currentMonth - 1;
    const prevYear = currentMonth === 1 ? currentYear - 1 : currentYear;
    const prevMonthLastDay = new Date(prevYear, prevMonth, 0).getDate();
    
    for (let i = startWeekday - 1; i >= 0; i--) {
      const day = prevMonthLastDay - i;
      const dateStr = `${prevYear}-${String(prevMonth).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      calendarDays.push({
        date: dateStr,
        day,
        isCurrentMonth: false,
        isToday: dateStr === todayStr,
        hasEvent: this.hasEventOnDate(dateStr),
        isSelected: false
      });
    }
    
    // 当月的日期
    for (let day = 1; day <= daysInMonth; day++) {
      const dateStr = `${currentYear}-${String(currentMonth).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      calendarDays.push({
        date: dateStr,
        day,
        isCurrentMonth: true,
        isToday: dateStr === todayStr,
        hasEvent: this.hasEventOnDate(dateStr),
        isSelected: false
      });
    }
    
    // 下个月的日期
    const remainingDays = 42 - calendarDays.length;
    const nextMonth = currentMonth === 12 ? 1 : currentMonth + 1;
    const nextYear = currentMonth === 12 ? currentYear + 1 : currentYear;
    
    for (let day = 1; day <= remainingDays; day++) {
      const dateStr = `${nextYear}-${String(nextMonth).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      calendarDays.push({
        date: dateStr,
        day,
        isCurrentMonth: false,
        isToday: dateStr === todayStr,
        hasEvent: this.hasEventOnDate(dateStr),
        isSelected: false
      });
    }
    
    this.setData({ calendarDays });
  },

  hasEventOnDate(dateStr: string): boolean {
    return this.data.allEvents.some(event => event.date === dateStr);
  },

  selectToday() {
    const today = new Date();
    const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    this.selectDateByString(todayStr);
  },

  selectDate(e: WechatMiniprogram.TouchEvent) {
    const day = e.currentTarget.dataset.day as CalendarDay;
    this.selectDateByString(day.date);
  },

  selectDateByString(dateStr: string) {
    const calendarDays = this.data.calendarDays.map(day => ({
      ...day,
      isSelected: day.date === dateStr
    }));
    
    const dayEvents = this.data.allEvents.filter(event => event.date === dateStr);
    
    const dateParts = dateStr.split('-');
    const selectedDateText = `${dateParts[1]}月${dateParts[2]}日`;
    
    this.setData({
      selectedDate: dateStr,
      selectedDateText,
      calendarDays,
      dayEvents
    });
  },

  prevMonth() {
    let { currentYear, currentMonth } = this.data;
    if (currentMonth === 1) {
      currentMonth = 12;
      currentYear--;
    } else {
      currentMonth--;
    }
    this.setData({ currentYear, currentMonth }, () => {
      this.generateCalendar();
    });
  },

  nextMonth() {
    let { currentYear, currentMonth } = this.data;
    if (currentMonth === 12) {
      currentMonth = 1;
      currentYear++;
    } else {
      currentMonth++;
    }
    this.setData({ currentYear, currentMonth }, () => {
      this.generateCalendar();
    });
  },

  loadUpcomingEvents() {
    const today = new Date();
    const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    
    const upcomingEvents = this.data.allEvents
      .filter(event => event.date >= todayStr)
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(0, 5);
    
    this.setData({ upcomingEvents });
  },

  showEventDetail(e: WechatMiniprogram.TouchEvent) {
    const event = e.currentTarget.dataset.event;
    this.setData({
      currentEvent: event,
      showDetail: true
    });
  },

  closeDetail() {
    this.setData({ showDetail: false });
  },

  onDetailClose(e: WechatMiniprogram.CustomEvent) {
    if (!e.detail.visible) {
      this.setData({ showDetail: false });
    }
  },

  goToSignup() {
    this.setData({ showDetail: false });
    wx.navigateTo({
      url: '/pages/gathering/signup/signup'
    });
  }
});
