import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import argparse
from datetime import datetime, timedelta
from matplotlib.widgets import SpanSelector
import gc

mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

def get_timestamp_from_indexed_df(indexed_df, day, end_day, start, end):
    if end == '00:00':
        target_df = indexed_df.loc[f'{day} {start}':f'{end_day} {end}'].copy()
    else:
        target_df = indexed_df.loc[f'{day} {start}':f'{end_day} {end}'].copy()
    
    target_df.index.name = 'timestamp'
    return target_df

class HourlyPlotter:
    def __init__(self, day_df, window_size=1):
        self.day_df = day_df
        self.keys = self.day_df.keys()[1:3]
        self.day_str = day_df.index.min().strftime('%Y-%m-%d')
        self.window_size = window_size
        self.current_idx = 0 
        self.span = None
        self.current_plot_df = None
        self.target_time = {"start":[], "end":[]}
        self.time_splits = [f"{h:02d}:00" for h in range(0, 24, self.window_size)]
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        print("플롯 생성 완료. '→' (다음), '←' (이전) 키로 탐색하세요.")

    def on_key_press(self, event):
        if event.key == 'right':
            self.current_idx = min(self.current_idx + 1, len(self.time_splits) - 1)
        elif event.key == 'left':
            self.current_idx = max(self.current_idx - 1, 0)
        else:
            return 
        
        print(f"Loading plot {self.current_idx + 1}/{len(self.time_splits)}...")
        self.plot_current_chunk()

    def plot_current_chunk(self):
        """현재 인덱스(current_idx)에 해당하는 데이터를 플로팅"""
        start = self.time_splits[self.current_idx]
        end_idx = (self.current_idx + 1) % len(self.time_splits)
        end = self.time_splits[end_idx]
        day = self.day_str
        end_day = self.day_str
        if end == '00:00':
            end_day_dt = datetime.strptime(self.day_str, '%Y-%m-%d') + timedelta(days=1)
            end_day = end_day_dt.strftime('%Y-%m-%d')
        data_found_and_valid = False
        title = ""
        try:
            if end == '00:00':
                self.current_plot_df = self.day_df.loc[f'{day} {start}':f'{end_day} {end}'].copy()
            else:
                self.current_plot_df = self.day_df.loc[f'{day} {start}':f'{day} {end}'].copy()
                
            if len(self.current_plot_df) < 2:
                title = f"{day} {start} ~ {end} (데이터 부족: {len(self.current_plot_df)}개)"
            else:
                data_found_and_valid = True # 2개 이상일 때만 True
                
        except Exception as e:
            title = f"{day} {start} ~ {end} (데이터 없음: {e})"

        self.ax.clear()
        if data_found_and_valid:
            self.ax.plot(self.current_plot_df.index, self.current_plot_df[self.keys[0]], label=f'{self.keys[0]}')
            self.ax.plot(self.current_plot_df.index, self.current_plot_df[self.keys[1]], label=f'{self.keys[1]}')

            self.ax.set_title(f'{day} ({start} ~ {end}) 유량 압력 데이터')
            self.ax.set_ylabel('Value')
            self.ax.set_xlabel('Time')
            self.ax.legend()
            self.ax.grid(True)
            
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            self.span = SpanSelector(
                self.ax,
                self.onselect,
                'horizontal',
                useblit=True
            )
        
        else:
            self.ax.set_title(title)
            self.ax.xaxis.set_major_formatter(plt.NullFormatter())
            self.ax.xaxis.set_major_locator(plt.NullLocator())
            self.ax.yaxis.set_major_formatter(plt.NullFormatter())
            self.ax.yaxis.set_major_locator(plt.NullLocator())

        try:
            self.fig.canvas.draw()
        except Exception as e:
            print(f"Canvas draw 실패: {e}")

    def onselect(self, xmin_float, xmax_float):
        """SpanSelector가 호출하는 콜백 함수"""
        start_time_aware = mdates.num2date(xmin_float)
        end_time_aware = mdates.num2date(xmax_float)
        start_time = start_time_aware.replace(tzinfo=None)
        end_time = end_time_aware.replace(tzinfo=None)

        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        self.target_time['start'].append(start_str)
        self.target_time['end'].append(end_str)
        print(f"\n[선택 구간] {start_str} 부터 {end_str} 까지")
        self.ax.axvspan(xmin_float, xmax_float, facecolor='gray', alpha=0.3)
        self.fig.canvas.draw_idle() 
        df_selected = self.current_plot_df.loc[start_time:end_time]
        
        if df_selected.empty:
            print("  -> 선택된 구간에 데이터가 없습니다.")
            return
        print(f"  -> 선택된 데이터 개수: {len(df_selected)}")

    def show(self):
        self.plot_current_chunk()
        plt.tight_layout()
        plt.show()

def main(args):    
    print(f"Loading data from: {args.csv_path}")
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"오류: {args.csv_path} 파일을 찾을 수 없습니다.")
        return

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    except KeyError:
        print("오류: 'timestamp' 컬럼을 찾을 수 없습니다.")
        return

    start_day_str = args.day
    end_day_dt = datetime.strptime(args.day, '%Y-%m-%d') + timedelta(days=1)
    end_day_str = end_day_dt.strftime('%Y-%m-%d')
    
    try:
        day_df = get_timestamp_from_indexed_df(df, start_day_str, end_day_str, '00:00', '00:00')
    except Exception as e:
         print(f"오류: {start_day_str} 00:00 ~ {end_day_str} 00:00 범위의 데이터 추출 실패: {e}")
         return

    print(f"원본 DataFrame (행 {len(df)}개) 삭제 중...")
    del df
    gc.collect() # 가비지 컬렉터 호출
    print("삭제 완료. 24시간 데이터만 메모리에 남음.")
    plotter = HourlyPlotter(day_df, args.window_size)
    plotter.show()
    pd.DataFrame.from_dict(plotter.target_time).to_csv(args.result_path)
    print("플롯 창을 닫았습니다. 프로그램을 종료합니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="/Users/ahha/image-processing/1010_1100_with_label.csv", help='target path .csv')
    parser.add_argument('--result_path', type=str, default="./result.csv", help='result path .csv')
    parser.add_argument('--day', type=str, default='2021-10-11', help='target day')
    parser.add_argument('--window_size', type=int, default=1, help='time window size')
    args = parser.parse_args()
    main(args)