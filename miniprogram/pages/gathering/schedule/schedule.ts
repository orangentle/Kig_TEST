// pages/gathering/schedule/schedule.ts
import type { GatheringEventView } from '../../../types/gathering';
import { getEventViews } from '../events';

interface CalendarDay {
  date: string;
  day: number;
  isCurrentMonth: boolean;
  isToday: boolean;
  hasEvent: boolean;
  isSelected: boolean;
}

Page({
  data: {
    currentYear: new Date().getFullYear(),
    currentMonth: new Date().getMonth() + 1,
    weekdays: ['日', '一', '二', '三', '四', '五', '六'],
    calendarDays: [] as CalendarDay[],
    selectedDate: '',
    selectedDateText: '今天',
    dayEvents: [] as GatheringEventView[],
    upcomingEvents: [] as GatheringEventView[],
    allEvents: [] as GatheringEventView[],
    showDetail: false,
    currentEvent: {} as GatheringEventView
  },

  onLoad(options: Record<string, string>) {
    this.loadEvents();
    this.generateCalendar();
    this.selectToday();
    this.loadUpcomingEvents();
    // 从主页/其他入口带 eventId 进来时，自动定位到该活动并展开详情
    if (options && options.eventId) {
      const ev = this.data.allEvents.find((e) => e.eventId === options.eventId);
      if (ev) {
        this.setData({ currentEvent: ev, showDetail: true });
        if (ev.date) {
          this.setData({
            currentYear: Number(ev.date.split('-')[0]),
            currentMonth: Number(ev.date.split('-')[1]),
          }, () => this.generateCalendar());
        }
      }
    }
  },

  goBack() {
    wx.navigateBack();
  },

  loadEvents() {
    // 单一数据源：events.ts
    this.setData({ allEvents: getEventViews() });
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
