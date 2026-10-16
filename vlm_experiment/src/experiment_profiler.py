"""任意代码区间的计时、allocator测量和算子采集；不拥有模型或实验循环。"""
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time


def memory_stats(torch):
    """当前CUDA设备的PyTorch allocator状态，单位bytes，不含其他进程/非PyTorch分配。"""
    return {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def _record(name, metadata):
    if any(not isinstance(v, (str, int, float, bool, type(None))) for v in metadata.values()):
        raise TypeError('Profiler metadata must contain scalars only')
    return {'phase': name, 'metadata': metadata, 'status': 'complete'}


def _summary(records, time_fields=(), memory_fields=()):
    result = {}
    for name in dict.fromkeys(r['phase'] for r in records):
        rows = [r for r in records if r['phase'] == name and r['status'] == 'complete']
        item = {'completed_calls': len(rows),
                'failed_calls': sum(r['phase'] == name and r['status'] == 'failed' for r in records)}
        for field in time_fields:
            values = [r[field] for r in rows if field in r]
            if values:
                item.update({f'total_{field}': sum(values), f'mean_{field}': statistics.mean(values)})
        for field in memory_fields:
            values = [r[field] for r in rows if field in r]
            if values:
                item['max_' + field] = max(values)
        result[name] = item
    return result


class TimingProfiler:
    """CPU区间计时及可选的当前stream CUDA事件计时，不读取allocator。

    measure不同步GPU。resolve在所有区间结束后统一同步，可重复调用；report
    只整理已解析结果，不隐式等待GPU。CPU耗时不代表GPU完成耗时；CUDA事件
    区间包含当前stream调度空隙，多stream工作需调用者显式建立依赖。
    """
    def __init__(self, torch, cuda_events=True):
        self.torch = torch
        self.cuda_events = cuda_events
        self.records = []
        self._pending = []
        self._active = 0

    @contextmanager
    def measure(self, name, **metadata):
        record = _record(name, metadata)
        start = end = None
        if self.cuda_events:
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
        self._active += 1
        tick = time.perf_counter()
        try:
            yield
        except BaseException:
            record['status'] = 'failed'
            raise
        finally:
            record['host_elapsed_us'] = (time.perf_counter() - tick) * 1e6
            try:
                if end is not None:
                    end.record()
                    self._pending.append((record, start, end))
            finally:
                self.records.append(record)
                self._active -= 1

    def resolve(self):
        if self._active:
            raise RuntimeError('Cannot resolve active timing intervals')
        if self._pending:
            self.torch.cuda.synchronize()
            for record, start, end in self._pending:
                record['cuda_elapsed_us'] = start.elapsed_time(end) * 1000
            self._pending.clear()

    def report(self):
        if self._active or self._pending:
            raise RuntimeError('Finish timing intervals and call resolve() before report()')
        return {'measurement': 'timing', 'calls': self.records,
                'summary': _summary(self.records, time_fields=('host_elapsed_us', 'cuda_elapsed_us'))}


class MemoryProfiler:
    """同步采样allocator入口/结束状态并测量峰值；不计时、不启用torch.profiler。

    峰值重置影响设备级计数，禁止嵌套包括不同实例在同设备上的测峰值区间。
    各区间峰值不能相加；增量不等于activation，reserved不等于活跃tensor。
    本类只保存标量，完整分配历史snapshot不属于此接口。
    """
    _active_devices = set()  # 单线程实验内按设备保护峰值重置，覆盖不同实例的嵌套。

    def __init__(self, torch):
        self.torch = torch
        self.records = []
        self._active = False

    @contextmanager
    def measure(self, name, **metadata):
        device = (id(self.torch.cuda), self.torch.cuda.current_device())
        if self._active or device in self._active_devices:
            raise RuntimeError('Memory intervals must not overlap or nest on the same device')
        record = _record(name, metadata)
        self.torch.cuda.synchronize()
        before = memory_stats(self.torch)
        self.torch.cuda.reset_peak_memory_stats()
        self._active = True
        self._active_devices.add(device)
        record.update(allocated_before_bytes=before['allocated_bytes'], reserved_before_bytes=before['reserved_bytes'])
        try:
            yield
        except BaseException:
            record['status'] = 'failed'
            raise
        finally:
            try:
                self.torch.cuda.synchronize()
                after = memory_stats(self.torch)
                record.update(allocated_after_bytes=after['allocated_bytes'], reserved_after_bytes=after['reserved_bytes'],
                              peak_allocated_bytes=after['peak_allocated_bytes'], peak_reserved_bytes=after['peak_reserved_bytes'],
                              peak_allocated_increase_bytes=max(0, after['peak_allocated_bytes'] - before['allocated_bytes']),
                              peak_reserved_increase_bytes=max(0, after['peak_reserved_bytes'] - before['reserved_bytes']))
            finally:
                self.records.append(record)
                self._active = False
                self._active_devices.discard(device)

    def report(self):
        if self._active:
            raise RuntimeError('Cannot report an active memory interval')
        return {'measurement': 'memory', 'calls': self.records,
                'summary': _summary(self.records, memory_fields=('peak_allocated_bytes', 'peak_reserved_bytes',
                                   'peak_allocated_increase_bytes', 'peak_reserved_increase_bytes'))}


class OperatorProfiler:
    """一个短窗口torch.profiler采集器：标记区间、推进采集步、显式导出。

    capture仅可运行一次；region和step仅在窗口内有效，region支持嵌套。采集
    设置由调用者明确传入，退出capture后才能export；算子内存是净变化，不是
    区间峰值。调用者负责预热和失败窗口的导出，不用export异常覆盖实验原异常。
    """
    def __init__(self, torch, settings):
        self.torch = torch
        self.settings = settings
        self._profile = None
        self._active = False
        self._closed = False
        self._regions = 0

    @contextmanager
    def capture(self):
        if self._profile is not None:
            raise RuntimeError('Operator capture may only run once per instance')
        self._profile = self.torch.profiler.profile(
            activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
            record_shapes=self.settings['record_shapes'], profile_memory=self.settings['profile_memory'],
            with_stack=self.settings['with_stack'])
        try:
            with self._profile:
                self._active = True
                try:
                    yield self
                finally:
                    self._active = False
        finally:
            self._closed = True

    @contextmanager
    def region(self, name):
        if not self._active:
            raise RuntimeError('Operator region requires an active capture')
        self._regions += 1
        try:
            with self.torch.profiler.record_function(name):
                yield
        finally:
            self._regions -= 1

    def step(self):
        if not self._active or self._regions:
            raise RuntimeError('Operator step requires capture and completed regions')
        self._profile.step()

    def export(self, output):
        if not self._closed or self._active or self._regions:
            raise RuntimeError('Finish operator capture before export()')
        output = Path(output)
        self._profile.export_chrome_trace(str(output / 'trace.json'))
        operators = []
        for event in self._profile.key_averages():
            operators.append({
                'name': event.key, 'count': event.count,
                'cpu_total_us': event.cpu_time_total, 'self_cpu_us': event.self_cpu_time_total,
                'device_total_us': getattr(event, 'device_time_total', getattr(event, 'cuda_time_total', 0)),
                'self_device_us': getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0)),
                'cpu_memory_bytes': event.cpu_memory_usage,
                'device_memory_bytes': getattr(event, 'device_memory_usage', getattr(event, 'cuda_memory_usage', 0)),
            })
        path = output / 'operators.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(sorted(operators, key=lambda op: op['self_device_us'], reverse=True),
                                        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
