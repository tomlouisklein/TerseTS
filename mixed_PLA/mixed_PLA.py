import math
from typing import List, Optional, Tuple
import struct
from dataclasses import dataclass
from typing import List

# Mixed_PLA.py

# Constants
class Space:
    UPPER_CHAIN = True
    LOWER_CHAIN = False
    CLOSED_FROM_ABOVE = False
    CLOSED_FROM_BELOW = True
    PIECES_CONNECTED = 'c'
    PIECES_DISJOINT = 'd'
    PIECES_MIXEDLINK = 'm'
    RIGHT_MOST = True
    LEFT_MOST = False
    REVERSE = True
    SIGN_VALUE_DIFF = 0.0000001
    SIGN_TIME_DIFF = 0.0000001
    MAX_UNI = 222222222

    @staticmethod
    def judge_value_by_two_sides(val):
        if val > Space.SIGN_VALUE_DIFF:
            return 1
        elif val < -Space.SIGN_VALUE_DIFF:
            return -1
        else:
            return 0

class HPlane:
    POINT_TO_ABOVE = True
    POINT_TO_BELOW = False
    CONTAIN_ALL_CHAIN = 2
    CONTAIN_SOME_CHAIN = 1
    CONTAIN_NONE_CHAIN = 0
    NON_EXIST_CHAIN = -1
    INCLUDE_POINT = 1
    TOUCH_POINT = 0
    EXCLUDE_POINT = -1

@dataclass
class DataPoint:
    t: float
    m: float

    def copy(self, other):
        self.t = other.t
        self.m = other.m

    def assign(self, t, m):
        self.t = t
        self.m = m


@dataclass
class Line:
    k: float = 0.0  # slope
    b: float = 0.0  # intercept

    def y(self, x):
        return self.k * x + self.b

    def link_two_points(self, p1: DataPoint, p2: DataPoint):
        if abs(p1.t - p2.t) < 1e-10:
            self.k = 0
            self.b = p1.m
        else:
            self.k = (p1.m - p2.m) / (p1.t - p2.t)
            self.b = (p1.t * p2.m - p1.m * p2.t) / (p1.t - p2.t)

    def transform(self, para_point, shift):
        self.k = para_point.x
        self.b = para_point.y
        self.b = self.b - self.k * shift

    def cross_point_with_slope(self, p: DataPoint, slope: float):
        self.k = slope
        self.b = p.m - slope * p.t

    def is_inner(self, p: DataPoint, direction):
        diff = self.y(p.t)
        if direction == HPlane.POINT_TO_ABOVE:
            diff = p.m - diff
        else:
            diff = diff - p.m
        return Space.judge_value_by_two_sides(diff)


@dataclass
class DataSegment:
    upper: DataPoint
    lower: DataPoint

    def __init__(self, dp: DataPoint = None, delta: float = None):
        if dp is not None and delta is not None:
            self.upper = DataPoint(dp.t, dp.m + delta)
            self.lower = DataPoint(dp.t, dp.m - delta)
        else:
            self.upper = DataPoint(0, 0)
            self.lower = DataPoint(0, 0)

    def copy(self, other):
        self.upper.copy(other.upper)
        self.lower.copy(other.lower)

    def value(self):
        return (self.upper.m + self.lower.m) / 2.0

    def is_vertical(self):
        diff = abs(self.upper.t - self.lower.t)
        return diff < Space.SIGN_TIME_DIFF

    def hitting_line(self, dp: DataPoint, exl: Line):
        if self.is_vertical():
            dp.assign(self.upper.t, exl.y(self.upper.t))
        else:
            lw = Line()
            lw.link_two_points(self.upper, self.lower)
            if abs(lw.k - exl.k) > 1e-10:
                # Cross two lines
                t = (exl.b - lw.b) / (lw.k - exl.k)
                m = (lw.b * exl.k - exl.b * lw.k) / (exl.k - lw.k)
                dp.assign(t, m)
            else:
                dp.assign((self.upper.t + self.lower.t) / 2.0,
                          (self.upper.m + self.lower.m) / 2.0)


@dataclass
class ParaPoint:
    x: float
    y: float

    def copy(self, other):
        self.x = other.x
        self.y = other.y

    def assign(self, x, y):
        self.x = x
        self.y = y

    def cross_two_lines(self, l1: Line, l2: Line):
        if abs(l1.k - l2.k) < 1e-10:
            return False
        self.x = (l2.b - l1.b) / (l1.k - l2.k)
        self.y = (l1.b * l2.k - l2.b * l1.k) / (l2.k - l1.k)
        return True


class Edge:
    def __init__(self, p: ParaPoint, color=False):
        self.p = ParaPoint(p.x, p.y)
        self.color = color


class Halfplane:
    def __init__(self, slope, intercept, direction):
        self.sep = Line(slope, intercept)
        self.direction = direction

    def is_inner(self, p: ParaPoint):
        diff = self.sep.y(p.x)
        if self.direction == HPlane.POINT_TO_ABOVE:
            diff = p.y - diff
        else:
            diff = diff - p.y
        return Space.judge_value_by_two_sides(diff)


@dataclass
class ChainMeters:
    size: int
    thr: float

    def __init__(self, delta, eps):
        if eps > 0:
            self.size = int(math.ceil(4 / eps))
        else:
            self.size = Space.MAX_UNI
        self.thr = eps * delta


class ConvexList:
    def __init__(self, convex_type):
        self.convex_type = convex_type
        self.edges: List[Edge] = []
        self.end_most = ParaPoint(0, 0)
        self.end_exist = False

    def init(self, p: Optional[ParaPoint], q: Optional[ParaPoint], e: ParaPoint):
        self.edges.clear()
        self.edges.append(Edge(p))
        if q is not None:
            self.edges.append(Edge(q))
        self.end_most.copy(e)
        self.end_exist = True

    def reset_endmost(self, temp: ParaPoint):
        self.end_most.copy(temp)

    def return_endmost(self):
        return self.end_most

    def return_size(self):
        if self.end_exist:
            return int(len(self.edges) * 1.5 + 1 + 0.5)
        else:
            return int(len(self.edges) * 1.5)

    def re_push_front(self, fr: ParaPoint):
        self.edges.insert(0, Edge(fr, True))

    def pad(self, h: Halfplane, reverse=False):
        if not self.end_exist:
            return HPlane.NON_EXIST_CHAIN

        if not reverse:  # Normal case
            end_p = self.end_most
            inner = h.is_inner(end_p)

            if inner != HPlane.EXCLUDE_POINT:
                return HPlane.CONTAIN_ALL_CHAIN

            while self.edges:
                beg_p = self.edges[-1].p
                inner = h.is_inner(beg_p)

                if inner == HPlane.INCLUDE_POINT:
                    cur_edge = Line()
                    cur_edge.link_two_points(DataPoint(beg_p.x, beg_p.y),
                                             DataPoint(end_p.x, end_p.y))
                    new_p = ParaPoint(0, 0)
                    new_p.cross_two_lines(cur_edge, h.sep)
                    self.edges.append(Edge(new_p))
                    return HPlane.CONTAIN_SOME_CHAIN
                elif inner == HPlane.TOUCH_POINT:
                    self.edges[-1].color = True
                    return HPlane.CONTAIN_SOME_CHAIN
                else:
                    end_p.copy(beg_p)
                    self.edges.pop()

            self.end_exist = False
            return HPlane.CONTAIN_NONE_CHAIN
        else:  # Reverse case
            if not self.edges:
                beg_p = self.end_most
            else:
                beg_p = self.edges[0].p

            inner = h.is_inner(beg_p)
            if inner != HPlane.EXCLUDE_POINT:
                return HPlane.CONTAIN_ALL_CHAIN

            while self.edges:
                if len(self.edges) > 1:
                    end_p = self.edges[1].p
                else:
                    end_p = self.end_most

                inner = h.is_inner(end_p)
                if inner == HPlane.INCLUDE_POINT:
                    cur_edge = Line()
                    cur_edge.link_two_points(DataPoint(beg_p.x, beg_p.y),
                                             DataPoint(end_p.x, end_p.y))
                    beg_p.cross_two_lines(cur_edge, h.sep)
                    return HPlane.CONTAIN_SOME_CHAIN
                elif inner == HPlane.TOUCH_POINT:
                    self.edges.pop(0)
                    return HPlane.CONTAIN_SOME_CHAIN
                else:
                    self.edges.pop(0)
                    beg_p = end_p

            self.end_exist = False
            return HPlane.CONTAIN_NONE_CHAIN

    def cut(self, h: Halfplane, reverse=False):
        if not self.end_exist:
            return None

        if not reverse:  # Normal case
            if self.edges:
                beg_p = self.edges[0].p
            else:
                beg_p = self.end_most

            inner = h.is_inner(beg_p)
            if inner != HPlane.EXCLUDE_POINT:
                return None

            while self.edges:
                if len(self.edges) > 1:
                    end_p = self.edges[1].p
                else:
                    end_p = self.end_most

                inner = h.is_inner(end_p)
                if inner == HPlane.EXCLUDE_POINT:
                    self.edges.pop(0)
                    beg_p = end_p
                elif inner == HPlane.TOUCH_POINT:
                    self.edges.pop(0)
                    return end_p
                else:  # INCLUDE_POINT
                    cur_edge = Line()
                    cur_edge.link_two_points(DataPoint(beg_p.x, beg_p.y),
                                             DataPoint(end_p.x, end_p.y))
                    beg_p.cross_two_lines(cur_edge, h.sep)
                    return beg_p

            self.end_exist = False
            return None
        else:  # Reverse case
            end_p = self.end_most
            inner = h.is_inner(end_p)

            if inner != HPlane.EXCLUDE_POINT:
                return None

            while self.edges:
                beg_p = self.edges[-1].p
                inner = h.is_inner(beg_p)

                if inner == HPlane.INCLUDE_POINT:
                    cur_edge = Line()
                    cur_edge.link_two_points(DataPoint(beg_p.x, beg_p.y),
                                             DataPoint(end_p.x, end_p.y))
                    end_p.cross_two_lines(cur_edge, h.sep)
                    return end_p
                elif inner == HPlane.TOUCH_POINT:
                    end_p.copy(beg_p)
                    self.edges.pop()
                    return end_p
                else:
                    end_p.copy(beg_p)
                    self.edges.pop()

            self.end_exist = False
            return None


@dataclass
class Segment:
    start_time: float
    end_time: float
    line: Line

    def evaluate(self, t: float) -> float:
        """Evaluate the segment at time t"""
        if self.start_time <= t <= self.end_time:
            return self.line.y(t)
        else:
            raise ValueError(f"Time {t} is outside segment range [{self.start_time}, {self.end_time}]")


class ConvexPoly:
    def __init__(self, delta, eps):
        self.cmt = ChainMeters(delta, eps)
        self.upper_edges = ConvexList(Space.UPPER_CHAIN)
        self.lower_edges = ConvexList(Space.LOWER_CHAIN)
        self.instantiated = False

    def init(self, p: DataPoint, q: DataPoint, delta):
        self.instantiated = True

        pu = Line(0, p.m + delta)
        pu.k = 0 - p.t
        pl = Line(0, p.m - delta)
        pl.k = 0 - p.t
        qu = Line(0, q.m + delta)
        qu.k = 0 - q.t
        ql = Line(0, q.m - delta)
        ql.k = 0 - q.t

        lm = ParaPoint(0, 0)
        lm.cross_two_lines(pu, ql)
        rm = ParaPoint(0, 0)
        rm.cross_two_lines(pl, qu)
        mt = ParaPoint(0, 0)
        mt.cross_two_lines(pu, qu)
        mb = ParaPoint(0, 0)
        mb.cross_two_lines(pl, ql)

        self.upper_edges.init(lm, mt, rm)
        self.lower_edges.init(rm, mb, lm)

    def re_init(self, lseg: DataSegment, c: DataSegment, time_base, closed_direction=False):
        self.instantiated = True

        uc = Halfplane(time_base - c.upper.t, c.upper.m, HPlane.POINT_TO_BELOW)
        lc = Halfplane(time_base - c.lower.t, c.lower.m, HPlane.POINT_TO_ABOVE)
        us = Halfplane(time_base - lseg.upper.t, lseg.upper.m, HPlane.POINT_TO_BELOW)
        ls = Halfplane(time_base - lseg.lower.t, lseg.lower.m, HPlane.POINT_TO_ABOVE)

        lt = ParaPoint(0, 0)
        lb = ParaPoint(0, 0)
        rt = ParaPoint(0, 0)
        rb = ParaPoint(0, 0)

        max_val = 100000000

        if abs(lseg.upper.t - c.upper.t) < 1e-10:
            lt.assign(-max_val, c.upper.m)
            lb.assign(-10 * max_val, c.lower.m)
            rt.cross_two_lines(ls.sep, uc.sep)
            rb.cross_two_lines(ls.sep, lc.sep)
        elif abs(lseg.lower.t - c.upper.t) < 1e-10:
            lt.cross_two_lines(us.sep, uc.sep)
            lb.cross_two_lines(us.sep, lc.sep)
            rt.assign(10 * max_val, c.upper.m)
            rb.assign(max_val, c.lower.m)
        else:
            lt.cross_two_lines(us.sep, uc.sep)
            lb.cross_two_lines(us.sep, lc.sep)
            rt.cross_two_lines(ls.sep, uc.sep)
            rb.cross_two_lines(ls.sep, lc.sep)

            is_conv_null = False

            if lseg.upper.t > lseg.lower.t:
                if lt.x >= rt.x and lb.x <= rb.x:
                    rt.cross_two_lines(us.sep, ls.sep)
                    self.upper_edges.init(lb, None, rt)
                    self.lower_edges.init(rt, rb, lb)
                    return
                elif lb.x > rb.x:
                    is_conv_null = True
            elif lseg.upper.t < lseg.lower.t:
                if lb.x >= rb.x and lt.x <= rt.x:
                    lb.cross_two_lines(us.sep, ls.sep)
                    self.upper_edges.init(lb, lt, rt)
                    self.lower_edges.init(rt, None, lb)
                    return
                elif lt.x > rt.x:
                    is_conv_null = True

            if is_conv_null:
                if closed_direction == Space.CLOSED_FROM_ABOVE:
                    rt.copy(lt)
                    rb.copy(lb)
                else:
                    lt.copy(rt)
                    lb.copy(rb)

        self.upper_edges.init(lb, lt, rt)
        self.lower_edges.init(rt, rb, lb)

    def return_size(self):
        if self.instantiated:
            return self.upper_edges.return_size() + self.lower_edges.return_size()
        return 0

    def return_endmost(self, right_or_left):
        if right_or_left:
            return self.upper_edges.return_endmost()
        else:
            return self.lower_edges.return_endmost()

    def reset_endmost(self, up_p: ParaPoint, low_p: ParaPoint):
        self.upper_edges.reset_endmost(up_p)
        self.lower_edges.reset_endmost(low_p)

    def is_instantiated(self):
        return self.instantiated

    def set_ins_none(self):
        self.instantiated = False

    def select_sol(self, shift, curb):
        s = Line()
        if not self.is_instantiated():
            s.k = 0
            s.b = curb
        else:
            up = self.upper_edges.return_endmost()
            lp = self.lower_edges.return_endmost()
            if abs(up.x) > abs(lp.x):
                s.transform(lp, shift)
            else:
                s.transform(up, shift)
        return s

    def intersect(self, h: Halfplane):
        if h.direction == HPlane.POINT_TO_BELOW:
            relationship = self.upper_edges.pad(h)
            end = self.lower_edges.cut(h)
            if end is not None:
                self.upper_edges.reset_endmost(end)
        else:
            relationship = self.lower_edges.pad(h)
            end = self.upper_edges.cut(h)
            if end is not None:
                self.lower_edges.reset_endmost(end)
        return relationship

    def load_arc_plane(self, h: Halfplane):
        if h.direction == HPlane.POINT_TO_BELOW:
            relationship = self.upper_edges.pad(h, Space.REVERSE)
            st = self.lower_edges.cut(h, Space.REVERSE)
            if st is not None:
                self.upper_edges.re_push_front(st)
        else:
            relationship = self.lower_edges.pad(h, Space.REVERSE)
            st = self.upper_edges.cut(h, Space.REVERSE)
            if st is not None:
                self.lower_edges.re_push_front(st)
        return relationship


class OrigConvexList:
    def __init__(self, convex_type, delta, eps):
        self.convex_type = convex_type
        self.cmt = ChainMeters(delta, eps)
        self.vex: List[DataPoint] = []
        self.ln = Line()

    def re_set(self, sp: ParaPoint, st: DataPoint, shift_time):
        self.vex.clear()
        self.ln.transform(sp, shift_time)
        self.pad(st)

    def pad(self, r: DataPoint):
        if self.convex_type:
            r_copy = DataPoint(r.t, r.m - self.cmt.thr)
        else:
            r_copy = DataPoint(r.t, r.m + self.cmt.thr)

        if not self.vex:
            self.vex.append(r_copy)
            return

        while len(self.vex) >= 2:
            q = self.vex[-1]
            p = self.vex[-2]

            hp = Halfplane(0, 0, self.convex_type)
            hp.sep.link_two_points(p, q)

            if hp.is_inner(ParaPoint(r_copy.t, r_copy.m)) == HPlane.INCLUDE_POINT:
                self.vex.append(r_copy)
                return
            else:
                self.vex.pop()

        ln_copy = Line(self.ln.k, self.ln.b)
        if self.convex_type == Space.UPPER_CHAIN:
            ln_copy.b -= self.cmt.thr
        else:
            ln_copy.b += self.cmt.thr

        rel = ln_copy.is_inner(r_copy, self.convex_type)
        if rel == HPlane.INCLUDE_POINT:
            self.vex.append(r_copy)
        elif rel == HPlane.TOUCH_POINT:
            self.vex.append(r_copy)
            if self.vex:
                self.vex.pop(0)
        elif rel == HPlane.EXCLUDE_POINT:
            print("Fatal error in origConvexList.pad()")

    def update(self, r: DataPoint):
        self.pad(r)
        return True

    def pop_back(self, delete=True):
        if not self.vex:
            return None
        dp = self.vex.pop()
        return dp

    def front(self):
        if not self.vex:
            return None
        return self.vex[0]

    def return_ex_light(self):
        return Line(self.ln.k, self.ln.b)

    def return_size(self):
        if not self.vex:
            return 1
        return int(len(self.vex) * 1.5 + 1)

    def clear(self):
        self.vex.clear()

    def point_to_direction(self):
        if self.convex_type:
            return HPlane.POINT_TO_BELOW
        else:
            return HPlane.POINT_TO_ABOVE


@dataclass
class FittingWindow:
    tu: float = -1
    tg: float = -1

    def copy(self, f):
        self.tu = f.tu
        self.tg = f.tg

    def assign(self, u, g):
        self.tu = u
        self.tg = g


@dataclass
class Ck:
    k: int
    knotype: bool  # True = connected, False = disconnected
    refn: int
    prev: Optional['Ck']
    lastknot: DataPoint
    end_point: DataPoint
    fw: FittingWindow

    def __init__(self, k, knotype, prev=None):
        self.k = k
        self.knotype = knotype
        self.refn = 1
        self.prev = prev
        if prev is not None:
            prev.inc_ref()
        self.lastknot = DataPoint(0, 0)
        self.end_point = DataPoint(0, 0)
        self.fw = FittingWindow()

    def dec_ref(self):
        self.refn -= 1

    def inc_ref(self):
        self.refn += 1


class Apr:
    def __init__(self, delta, eps, pieces_type):
        self.mtr = {'delta': delta, 'eps': eps}
        self.pieces_type = pieces_type
        self.segs: List[DataPoint] = []
        self.cur_time = 0
        self.time_base = 0
        self._buffer_dp = None
        self.delay_info = 0
        self.apx_type = 0

    def fitting_cost(self):
        return -1.0

    def return_size(self):
        return 0


class ContApr(Apr):
    def __init__(self, delta, eps):
        super().__init__(delta, eps, Space.PIECES_CONNECTED)
        self.apx_type = 2
        self.closed_direction = False
        self.real_bndy: List[DataSegment] = []
        self.conv = ConvexPoly(delta, eps)
        self.light_window = DataSegment()
        self.ceil_arc = OrigConvexList(True, delta, eps)
        self.flor_arc = OrigConvexList(False, delta, eps)

    def buffer(self, dp: DataPoint):
        self.cur_time = dp.t
        size = len(self.real_bndy)

        if size == 0:
            self.time_base = dp.t
            self._buffer_dp = DataPoint(dp.t, dp.m)
            self.init_light_window(dp, self.mtr['delta'])
            self.real_bndy.append(DataSegment(dp, self.mtr['delta']))
        elif size == 1:
            self.conv.init(self._buffer_dp, dp, self.mtr['delta'])

            sec = DataSegment(dp, self.mtr['delta'])
            self.ceil_arc.re_set(self.conv.return_endmost(Space.RIGHT_MOST),
                                 sec.upper, self.time_base)
            self.flor_arc.re_set(self.conv.return_endmost(Space.LEFT_MOST),
                                 sec.lower, self.time_base)

            self._buffer_dp = None
            self.real_bndy.append(DataSegment(dp, self.mtr['delta']))
        elif size in [2, 3]:
            self.real_bndy.append(DataSegment(dp, self.mtr['delta']))

    def init_light_window(self, dp: DataPoint, delta):
        self.light_window.upper.assign(dp.t, dp.m + delta)
        self.light_window.lower.assign(dp.t, dp.m - delta)

    def update_immediately(self):
        if self.real_bndy:
            a = self.real_bndy.pop(0)

        if len(self.real_bndy) < 2:
            return True

        c = self.real_bndy[1]

        uh = Halfplane(self.time_base - c.upper.t, c.upper.m, HPlane.POINT_TO_BELOW)
        rst_uh = self.conv.intersect(uh)
        if rst_uh == HPlane.CONTAIN_SOME_CHAIN:
            self.ceil_arc.re_set(self.conv.return_endmost(Space.RIGHT_MOST),
                                 c.upper, self.time_base)

        lh = Halfplane(self.time_base - c.lower.t, c.lower.m, HPlane.POINT_TO_ABOVE)
        rst_lh = self.conv.intersect(lh)
        if rst_lh == HPlane.CONTAIN_SOME_CHAIN:
            self.flor_arc.re_set(self.conv.return_endmost(Space.LEFT_MOST),
                                 c.lower, self.time_base)

        if rst_uh == HPlane.CONTAIN_NONE_CHAIN or rst_lh == HPlane.CONTAIN_NONE_CHAIN:
            if rst_uh == HPlane.CONTAIN_NONE_CHAIN:
                self.closed_direction = Space.UPPER_CHAIN
                self.ceil_arc.clear()
                chain = self.flor_arc
            else:
                self.closed_direction = Space.LOWER_CHAIN
                self.flor_arc.clear()
                chain = self.ceil_arc

            exl = chain.return_ex_light()
            self.record_last_knot(exl)

            lseg = self.compute_window_lseg(c, self.closed_direction)
            self.restart_new_round(lseg, chain)

            return False
        else:
            if rst_uh == HPlane.CONTAIN_ALL_CHAIN:
                self.ceil_arc.update(c.upper)
            if rst_lh == HPlane.CONTAIN_ALL_CHAIN:
                self.flor_arc.update(c.lower)
            return True

    def update_last(self):
        if self.real_bndy:
            self.real_bndy.pop(0)

        if not self.real_bndy:
            return True

        c = self.real_bndy[-1]

        uh = Halfplane(self.time_base - c.upper.t, c.upper.m, HPlane.POINT_TO_BELOW)
        rst_uh = self.conv.intersect(uh)
        if rst_uh == HPlane.CONTAIN_SOME_CHAIN:
            self.ceil_arc.re_set(self.conv.return_endmost(Space.RIGHT_MOST),
                                 c.upper, self.time_base)

        lh = Halfplane(self.time_base - c.lower.t, c.lower.m, HPlane.POINT_TO_ABOVE)
        rst_lh = self.conv.intersect(lh)
        if rst_lh == HPlane.CONTAIN_SOME_CHAIN:
            self.flor_arc.re_set(self.conv.return_endmost(Space.LEFT_MOST),
                                 c.lower, self.time_base)

        if rst_uh == HPlane.CONTAIN_NONE_CHAIN or rst_lh == HPlane.CONTAIN_NONE_CHAIN:
            if rst_uh == HPlane.CONTAIN_NONE_CHAIN:
                self.closed_direction = Space.UPPER_CHAIN
                self.ceil_arc.clear()
                chain = self.flor_arc
            else:
                self.closed_direction = Space.LOWER_CHAIN
                self.flor_arc.clear()
                chain = self.ceil_arc

            exl = chain.return_ex_light()
            self.record_last_knot(exl)

            lseg = self.compute_window_lseg(c, self.closed_direction)

            return False
        else:
            if rst_uh == HPlane.CONTAIN_ALL_CHAIN:
                self.ceil_arc.update(c.upper)
            if rst_lh == HPlane.CONTAIN_ALL_CHAIN:
                self.flor_arc.update(c.lower)
            return True

    def compute_window_lseg(self, c: DataSegment, up_or_low):
        lseg = DataSegment()

        if up_or_low == Space.UPPER_CHAIN:
            extreme_light = Line()
            extreme_light.transform(self.conv.return_endmost(Space.LEFT_MOST),
                                    self.time_base)

            self.light_window.upper.assign(c.upper.t,
                                           extreme_light.y(c.upper.t))
            ft = self.flor_arc.front().t if self.flor_arc.front() else c.upper.t
            self.light_window.lower.assign(ft, extreme_light.y(ft))

            lseg.upper.copy(self.light_window.upper)
            last = self.flor_arc.pop_back(False)
            if last:
                lseg.lower.copy(last)
            else:
                lseg.lower.copy(self.light_window.lower)
        else:
            extreme_light = Line()
            extreme_light.transform(self.conv.return_endmost(Space.RIGHT_MOST),
                                    self.time_base)

            self.light_window.lower.assign(c.lower.t,
                                           extreme_light.y(c.lower.t))
            ft = self.ceil_arc.front().t if self.ceil_arc.front() else c.lower.t
            self.light_window.upper.assign(ft, extreme_light.y(ft))

            lseg.lower.copy(self.light_window.lower)
            last = self.ceil_arc.pop_back(False)
            if last:
                lseg.upper.copy(last)
            else:
                lseg.upper.copy(self.light_window.upper)

        return lseg

    def restart_new_round(self, lseg: DataSegment, chain: OrigConvexList):
        if len(self.real_bndy) < 2:
            return True

        c = self.real_bndy[1]
        self.time_base = c.upper.t

        self.conv.re_init(lseg, c, self.time_base, self.closed_direction)

        while True:
            dp = chain.pop_back(False)
            if dp is None:
                break
            h = Halfplane(self.time_base - dp.t, dp.m, chain.point_to_direction())
            self.conv.load_arc_plane(h)

        self.ceil_arc.re_set(self.conv.return_endmost(Space.RIGHT_MOST),
                             c.upper, self.time_base)
        self.flor_arc.re_set(self.conv.return_endmost(Space.LEFT_MOST),
                             c.lower, self.time_base)

        return True

    def record_last_knot(self, rsep: Line):
        dp = DataPoint(0, 0)

        if self.light_window.is_vertical():
            x = self.light_window.upper.t
            dp.assign(x, rsep.y(x))
        else:
            lw = Line()
            lw.link_two_points(self.light_window.upper, self.light_window.lower)
            if abs(lw.k - rsep.k) > 1e-10:
                t = (rsep.b - lw.b) / (lw.k - rsep.k)
                m = (lw.b * rsep.k - rsep.b * lw.k) / (rsep.k - lw.k)
                dp.assign(t, m)
            else:
                dp.assign((self.light_window.upper.t + self.light_window.lower.t) / 2,
                          (self.light_window.upper.m + self.light_window.lower.m) / 2)

        self.segs.append(dp)
        self.delay_info += int(self.cur_time - self.time_base)

    def return_size(self):
        count = len(self.real_bndy)
        return (count + self.conv.return_size() +
                self.ceil_arc.return_size() + self.flor_arc.return_size())

    def fitting_cost(self):
        pieces = len(self.segs) - 1
        return 2.0 * pieces


class Fittable(ContApr):
    def __init__(self, delta, eps, k, connected):
        super().__init__(delta, eps)
        self.apx_type = 4
        self.knot_type = connected
        self.fw = FittingWindow()
        self.beg_point = DataPoint(0, 0)
        self.end_point = DataPoint(0, 0)
        self.bias_lseg = None
        self.bias_chain = None

    def update(self, dp: DataPoint):
        self.buffer(dp)

        if len(self.real_bndy) <= 3:
            return True

        proceed = self.update_immediately()
        return proceed

    def update_immediately(self):
        """Override the base class method with fittable-specific logic"""
        if self.real_bndy:
            a = self.real_bndy.pop(0)

        if len(self.real_bndy) < 2:
            return True

        c = self.real_bndy[1]

        uh = Halfplane(self.time_base - c.upper.t, c.upper.m, HPlane.POINT_TO_BELOW)
        rst_uh = self.conv.intersect(uh)
        if rst_uh == HPlane.CONTAIN_SOME_CHAIN:
            self.ceil_arc.re_set(self.conv.return_endmost(Space.RIGHT_MOST),
                                 c.upper, self.time_base)

        lh = Halfplane(self.time_base - c.lower.t, c.lower.m, HPlane.POINT_TO_ABOVE)
        rst_lh = self.conv.intersect(lh)
        if rst_lh == HPlane.CONTAIN_SOME_CHAIN:
            self.flor_arc.re_set(self.conv.return_endmost(Space.LEFT_MOST),
                                 c.lower, self.time_base)

        if rst_uh == HPlane.CONTAIN_NONE_CHAIN or rst_lh == HPlane.CONTAIN_NONE_CHAIN:
            if rst_uh == HPlane.CONTAIN_NONE_CHAIN:
                self.closed_direction = Space.UPPER_CHAIN
                self.ceil_arc.clear()
                self.bias_chain = self.flor_arc
            else:
                self.closed_direction = Space.LOWER_CHAIN
                self.flor_arc.clear()
                self.bias_chain = self.ceil_arc

            exl = self.bias_chain.return_ex_light()
            self.compute_end_point(exl, c.upper.t)

            self.bias_lseg = self.compute_window_lseg(c, self.closed_direction)

            if self.closed_direction == Space.UPPER_CHAIN:
                self.fw.tu = c.upper.t
                self.fw.tg = self.light_window.lower.t
            else:
                self.fw.tu = self.light_window.upper.t
                self.fw.tg = c.lower.t

            return False
        else:
            if rst_uh == HPlane.CONTAIN_ALL_CHAIN:
                self.ceil_arc.update(c.upper)
            if rst_lh == HPlane.CONTAIN_ALL_CHAIN:
                self.flor_arc.update(c.lower)
            return True

    def update_last(self):
        """Override the base class method with fittable-specific logic"""
        if self.real_bndy:
            self.real_bndy.pop(0)

        if not self.real_bndy:
            return True

        c = self.real_bndy[-1]

        uh = Halfplane(self.time_base - c.upper.t, c.upper.m, HPlane.POINT_TO_BELOW)
        rst_uh = self.conv.intersect(uh)
        if rst_uh == HPlane.CONTAIN_SOME_CHAIN:
            self.ceil_arc.re_set(self.conv.return_endmost(Space.RIGHT_MOST),
                                 c.upper, self.time_base)

        lh = Halfplane(self.time_base - c.lower.t, c.lower.m, HPlane.POINT_TO_ABOVE)
        rst_lh = self.conv.intersect(lh)
        if rst_lh == HPlane.CONTAIN_SOME_CHAIN:
            self.flor_arc.re_set(self.conv.return_endmost(Space.LEFT_MOST),
                                 c.lower, self.time_base)

        if rst_uh == HPlane.CONTAIN_NONE_CHAIN or rst_lh == HPlane.CONTAIN_NONE_CHAIN:
            if rst_uh == HPlane.CONTAIN_NONE_CHAIN:
                self.closed_direction = Space.UPPER_CHAIN
                self.ceil_arc.clear()
                self.bias_chain = self.flor_arc
            else:
                self.closed_direction = Space.LOWER_CHAIN
                self.flor_arc.clear()
                self.bias_chain = self.ceil_arc

            exl = self.bias_chain.return_ex_light()
            self.compute_end_point(exl, c.upper.t)

            self.bias_lseg = self.compute_window_lseg(c, self.closed_direction)

            if self.closed_direction == Space.UPPER_CHAIN:
                self.fw.tu = c.upper.t
                self.fw.tg = self.light_window.lower.t
            else:
                self.fw.tu = self.light_window.upper.t
                self.fw.tg = c.lower.t

            return False
        else:
            if rst_uh == HPlane.CONTAIN_ALL_CHAIN:
                self.ceil_arc.update(c.upper)
            if rst_lh == HPlane.CONTAIN_ALL_CHAIN:
                self.flor_arc.update(c.lower)
            return True

    def close_fitting(self):
        last = self.real_bndy[-1]
        sol = self.conv.select_sol(self.time_base, last.value())

        last_t = self.cur_time + 0.0001
        self.compute_end_point(sol, last_t)
        self.fw.assign(last_t, last_t)

    def init_new_round_with_type(self, link_type):
        self.conv.set_ins_none()
        if link_type == Space.PIECES_CONNECTED:
            self.knot_type = True
            self.restart_cont_new_round()
        elif link_type == Space.PIECES_DISJOINT:
            self.knot_type = False
            self.restart_uncont_new_round()

    def restart_uncont_new_round(self):
        if self.real_bndy:
            self.real_bndy.pop(0)

        if not self.real_bndy:
            return

        c = self.real_bndy[0]
        self.light_window.copy(c)

        self.time_base = c.upper.t

        if len(self.real_bndy) == 2:
            d = self.real_bndy[1]
            self.conv.re_init(c, d, self.time_base)
            self.ceil_arc.re_set(self.conv.return_endmost(Space.RIGHT_MOST),
                                 d.upper, self.time_base)
            self.flor_arc.re_set(self.conv.return_endmost(Space.LEFT_MOST),
                                 d.lower, self.time_base)

    def restart_cont_new_round(self):
        if self.bias_lseg is not None and self.bias_chain is not None:
            self.restart_new_round(self.bias_lseg, self.bias_chain)
        self.bias_chain = None
        self.bias_lseg = None

    def clone_fittable(self, f: 'Fittable'):
        self.fw.copy(f.fw)
        self.bias_chain = None
        self.bias_lseg = None

        # Clone real_bndy
        self.real_bndy.clear()
        for ds in f.real_bndy:
            new_ds = DataSegment()
            new_ds.copy(ds)
            self.real_bndy.append(new_ds)

        # Clone other properties
        self.light_window.copy(f.light_window)
        self.cur_time = f.cur_time
        self.time_base = f.time_base
        self.apx_type = f.apx_type
        self.pieces_type = f.pieces_type

        # Clone segs
        self.segs.clear()
        for dp in f.segs:
            self.segs.append(DataPoint(dp.t, dp.m))

        if f._buffer_dp:
            self._buffer_dp = DataPoint(f._buffer_dp.t, f._buffer_dp.m)
        else:
            self._buffer_dp = None

    def compute_end_point(self, exl: Line, ctime):
        """Compute the beginning and end points for the current segment"""
        self.light_window.hitting_line(self.beg_point, exl)
        self.end_point.assign(ctime, exl.y(ctime))


class MixContApr(Apr):
    def __init__(self, delta, eps):
        super().__init__(delta, eps, Space.PIECES_MIXEDLINK)
        self.apx_type = 4
        self.k = -1
        self.ck_list: List[Ck] = []
        self.c = [None, None, None]  # C[k], C[k+1], C[k+2]

        # self.debug_points_added = []  # DEBUG: Track all point additions

        # Initialize base array with 5 fittable objects
        self.base = [
            Fittable(delta, eps, 2, False),  # base[k] + d
            Fittable(delta, eps, 2, True),  # base[k+1] + c
            Fittable(delta, eps, 3, False),  # base[k+1] + d
            Fittable(delta, eps, 3, True),  # base[k+2] + c
            Fittable(delta, eps, 4, False)  # base[k+2] + d
        ]

        # Initialize flags
        self.fg = [True, True, True, True, True]
        self.opt = 0
        self.knot_flags = []

    def update(self, p: DataPoint):
        # Update all five candidates
        for i in range(5):
            if self.fg[i]:
                self.fg[i] = self.base[i].update(p)

        self.cur_time = p.t

        # Process when both fg[0] and fg[1] are False
        while not self.fg[0] and not self.fg[1]:
            self.dp_to_bases()

        return True

    # def close_fitting(self):
    #     # Update last segments
    #     for i in range(5):
    #         if self.fg[i]:
    #             self.fg[i] = self.base[i].update_last()
    #
    #     while not self.fg[0] and not self.fg[1]:
    #         self.dp_to_bases()
    #
    #     self.opt = self.is_better(0, 1)
    #     self.base[self.opt].close_fitting()
    #
    #     ck3 = self.create_ck3(self.opt)
    #
    #     # DEBUG: Show what's in the different data structures
    #     print(f"🔍 Before detect_and_output_fixed_pieces:")
    #     print(f"   ck_list length: {len(self.ck_list)}")
    #     for i, ck in enumerate(self.ck_list):
    #         print(f"     ck_list[{i}]: lastknot.t={ck.lastknot.t:.1f}, refn={ck.refn}")
    #
    #     print(f"   c[0]: {'None' if self.c[0] is None else f'{self.c[0].lastknot.t:.1f}'}")
    #     print(f"   c[1]: {'None' if self.c[1] is None else f'{self.c[1].lastknot.t:.1f}'}")
    #     print(f"   c[2]: {'None' if self.c[2] is None else f'{self.c[2].lastknot.t:.1f}'}")
    #     print(f"   ck3: {'None' if ck3 is None else f'{ck3.lastknot.t:.1f}'}")
    #
    #     beg_k = self.push_ck_and_try_to_erase(self.c[0])
    #     self.detect_and_output_fixed_pieces(beg_k)
    #
    #     self.k += 1
    #     self.c[0] = self.c[1]
    #     self.c[1] = self.c[2]
    #
    #     # DEBUG
    #     print(f"🔍 After detect_and_output_fixed_pieces:")
    #     print(f"   ck_list length: {len(self.ck_list)}")
    #     for i, ck in enumerate(self.ck_list):
    #         print(f"     ck_list[{i}]: lastknot.t={ck.lastknot.t:.1f}, refn={ck.refn}")
    #
    #     print(f"   c[0]: {'None' if self.c[0] is None else f'{self.c[0].lastknot.t:.1f}'}")
    #     print(f"   c[1]: {'None' if self.c[1] is None else f'{self.c[1].lastknot.t:.1f}'}")
    #
    #     self.c[2] = ck3
    #
    #     if self.c[2] is None:
    #         self.c[2] = self.c[1]
    #
    #     print(f"🔍 Before output_list:")
    #     if self.c[2] is not None:
    #         print(f"   c[2] chain exists: lastknot.t={self.c[2].lastknot.t:.1f}")
    #         print(f"   Tracing c[2] chain:")
    #         ck = self.c[2]
    #         chain_length = 0
    #         while ck is not None:
    #             print(
    #                 f"     chain[{chain_length}]: lastknot.t={ck.lastknot.t:.1f}, end_point.t={ck.end_point.t:.1f}, knotype={ck.knotype}, refn={ck.refn}")
    #             ck = ck.prev
    #             chain_length += 1
    #             if chain_length > 10:  # Prevent infinite loops
    #                 print(f"     ... (stopping trace at 10 elements)")
    #                 break
    #     else:
    #         print(f"   c[2] is None")
    #
    #     self.output_list(self.c[2])
    #
    #     print(f"🔍 After output_list:")
    #     print(f"   Total segs added: {len(self.segs)}")
    #     print(f"   Total knot_flags: {len(self.knot_flags)}")
    #
    #     self.clear_data()
    #
    #     if self.knot_flags:
    #         self.delay_info = self.delay_info / len(self.knot_flags)

    def close_fitting(self):
        # Update last segments
        for i in range(5):
            if self.fg[i]:
                self.fg[i] = self.base[i].update_last()

        while not self.fg[0] and not self.fg[1]:
            self.dp_to_bases()

        self.opt = self.is_better(0, 1)
        self.base[self.opt].close_fitting()

        ck3 = self.create_ck3(self.opt)

        beg_k = self.push_ck_and_try_to_erase(self.c[0])
        self.detect_and_output_fixed_pieces(beg_k)

        self.k += 1
        self.c[0] = self.c[1]
        self.c[1] = self.c[2]
        self.c[2] = ck3



        if self.c[2] is None:
            self.c[2] = self.c[1]

        self.output_list(self.c[2])
        self.clear_data()

        if self.knot_flags:
            self.delay_info = self.delay_info / len(self.knot_flags)

    def dp_to_bases(self):
        # 1. Prepare C[k+3]
        self.opt = self.is_better(0, 1)
        ck3 = self.create_ck3(self.opt)

        beg_k = self.push_ck_and_try_to_erase(self.c[0])

        # 2. Prepare two new bases
        if ck3 is not None:
            self.base[1 - self.opt].clone_fittable(self.base[self.opt])

            ck3c = self.base[self.opt]
            ck3d = self.base[1 - self.opt]

            ck3c.init_new_round_with_type(Space.PIECES_CONNECTED)
            ck3d.init_new_round_with_type(Space.PIECES_DISJOINT)
        else:
            ck3c = self.base[self.opt]
            ck3d = self.base[1 - self.opt]
            ck3c.fw.assign(-1, -1)
            ck3d.fw.assign(-1, -1)

        # 3. Translate by 2
        self.k += 1

        # Move flags
        self.fg[0] = self.fg[2]
        self.fg[1] = self.fg[3]
        self.fg[2] = self.fg[4]
        if ck3 is not None:
            self.fg[3] = True
            self.fg[4] = True
        else:
            self.fg[3] = False
            self.fg[4] = False

        # Move bases
        self.base[0] = self.base[2]
        self.base[1] = self.base[3]
        self.base[2] = self.base[4]
        self.base[3] = ck3c
        self.base[4] = ck3d

        # Move C[k]
        self.c[0] = self.c[1]
        self.c[1] = self.c[2]

        self.c[2] = ck3
        if ck3 is not None:
            ck3.inc_ref()  # FIX: reference ck3


        # Try to output fixed pieces
        self.detect_and_output_fixed_pieces(beg_k)

    def create_ck3(self, win):
        ck3 = None

        f3 = FittingWindow()
        f3.copy(self.base[win].fw)

        f2 = FittingWindow()
        if self.c[2] is not None:
            f2.copy(self.c[2].fw)

        if f2.tg >= f3.tg and f2.tu >= f3.tu:
            return None
        else:
            if win == 0:
                ck3 = Ck(self.k + 3, False, self.c[0])

                #ck3 = Ck(self.k + 3, False, self.c[0]) # THIS IS THE MAGICAL FIX


                    # DEBUG
                    # prev_info = f"t={self.c[0].lastknot.t:.1f}" if self.c[0] is not None else "None"
                    # print(f"🔍 create_ck3: Created ck3 with prev=c[0] ({prev_info})")
                    # if self.c[0] is not None:
                    #     print(f"    c[0] refn after inc_ref: {self.c[0].refn}")



            elif win == 1:

                # DEBUG
                # prev_info = f"t={self.c[1].lastknot.t:.1f}" if self.c[1] is not None else "None"
                # print(f"🔍 create_ck3: Created ck3 with prev=c[1] ({prev_info})")
                # if self.c[1] is not None:
                #     print(f"    c[1] refn after inc_ref: {self.c[1].refn}")

                ck3 = Ck(self.k + 3, True, self.c[1])

            ck3.lastknot.copy(self.base[win].beg_point)
            ck3.end_point.copy(self.base[win].end_point)
            ck3.fw.copy(self.base[win].fw)

            return ck3

    def push_ck_and_try_to_erase(self, out: Optional[Ck]):
        if out is None:
            return -1

        self.ck_list.append(out)

        ck = out
        while ck is not None:
            # Find ck in list
            idx = -1
            for i in range(len(self.ck_list) - 1, -1, -1):
                if self.ck_list[i] == ck:
                    idx = i
                    break

            if idx == -1:
                print("Locating error...")
                return -2

            ck.dec_ref()

            if ck.refn < 0:
                print("Technical error.")
                return ck.k
            elif ck.refn > 0:
                return ck.refn

            # Can erase current ck
            prev = ck.prev
            self.ck_list.pop(idx)
            ck = prev

        return -1

    def detect_and_output_fixed_pieces(self, beg_k):
        while self.ck_list:
            fir = self.ck_list[0]
            if fir.refn != 1:
                return

            if len(self.ck_list) > 1:
                sec = self.ck_list[1]
            else:
                if self.c[0] is not None:
                    sec = self.c[0]
                elif self.c[1] is not None:
                    sec = self.c[1]
                elif self.c[2] is not None:
                    sec = self.c[2]
                else:
                    print("Error in fix!")
                    return

            if sec.prev == fir:
                # Record this segment
                self.knot_flags.append(fir.knotype)

                # DEBUG: Track point addition
                # point_to_add = DataPoint(fir.lastknot.t, fir.lastknot.m)
                # # print(f"🔍 detect_and_output_fixed_pieces: Adding knot point t={point_to_add.t:.1f}, m={point_to_add.m:.6f}")
                # # self.debug_points_added.append(('detect_fixed_knot', point_to_add.t, point_to_add.m))

                self.segs.append(DataPoint(fir.lastknot.t, fir.lastknot.m))

                if not sec.knotype:  # Disconnected

                    # DEBUG: Track point addition
                    # end_point_to_add = DataPoint(fir.end_point.t, fir.end_point.m)
                    # print(f"🔍 detect_and_output_fixed_pieces: Adding end point (disconnected) t={end_point_to_add.t:.1f}, m={end_point_to_add.m:.6f}")
                    # self.debug_points_added.append(('detect_fixed_end', end_point_to_add.t, end_point_to_add.m))

                    self.segs.append(DataPoint(fir.end_point.t, fir.end_point.m))

                # Record delay
                self.delay_info += int(self.cur_time - fir.lastknot.t)

                # Delete fir
                self.ck_list.pop(0)
                sec.prev = None
            else:
                return

    def output_list(self, ck: Optional[Ck]):
        if ck is None:
            return

        flags = []
        dps = []

        while ck is not None:
            ck.dec_ref()

            if not flags:

                # DEBUG
                # end_point = DataPoint(ck.end_point.t, ck.end_point.m)
                # print(f"🔍 output_list: Adding initial end point t={end_point.t:.1f}, m={end_point.m:.6f}")
                # self.debug_points_added.append(('output_initial_end', end_point.t, end_point.m))

                dps.append(DataPoint(ck.end_point.t, ck.end_point.m)) # keep
            elif not flags[-1]:  # Last was disconnected

                # DEBUG
                # end_point = DataPoint(ck.end_point.t, ck.end_point.m)
                # print(f"🔍 output_list: Adding disconnected end point t={end_point.t:.1f}, m={end_point.m:.6f}")
                # self.debug_points_added.append(('output_disconnected_end', end_point.t, end_point.m))

                dps.append(DataPoint(ck.end_point.t, ck.end_point.m)) #keep

            dps.append(DataPoint(ck.lastknot.t, ck.lastknot.m))

            # Record delay
            self.delay_info += int(self.cur_time - ck.lastknot.t)

            flags.append(ck.knotype)
            prev = ck.prev
            ck = prev

        # Reverse and add to main lists
        while flags:
            self.knot_flags.append(flags.pop())
        while dps:

            #point = dps.pop()

            # DEBUG
            # print(f"🔍 output_list: Final append t={point.t:.1f}, m={point.m:.6f}")
            # self.debug_points_added.append(('output_final', point.t, point.m))

            #self.segs.append(point) # keep
            self.segs.append(dps.pop())

    def is_better(self, i, j):
        win = -1

        if not self.fg[i] and self.fg[j]:
            win = j
        elif self.fg[i] and not self.fg[j]:
            win = i
        elif not self.fg[i] and not self.fg[j]:
            if self.base[i].fw.tu > self.base[j].fw.tu:
                if self.base[i].fw.tg >= self.base[j].fw.tg:
                    win = i
                else:
                    print("FW crossing error 1.")
                    win = 0
            elif self.base[i].fw.tu < self.base[j].fw.tu:
                if self.base[i].fw.tg <= self.base[j].fw.tg:
                    win = j
                else:
                    print("FW crossing error 2.")
                    win = 0
            else:  # Equal tu
                if self.base[i].fw.tg > self.base[j].fw.tg:
                    win = i
                else:
                    win = j
        else:
            return 1

        return win

    def clear_data(self):
        self.c = [None, None, None]
        self.ck_list.clear()

    def fitting_cost(self):
        return self.k + 2

    def return_size(self):
        sum_size = 3 * (3 + len(self.ck_list))
        for i in range(5):
            if self.fg[i]:
                sum_size += self.base[i].return_size()
        return sum_size + 1

    def run(self, data_points: List[DataPoint], return_segments=False):
        """Run the algorithm on a list of data points"""
        for dp in data_points:
            self.update(dp)
        self.close_fitting()

        # Return segments and knot_flags
        return self.segs, self.knot_flags

def read_segments(segment_data, knot_flags):
    """Convert knot points and flags into actual line segments

    Following the C++ algorithm logic:
    - Skip first knot flag (like C++ code does with knot++)
    - knot_flags[i] indicates if there's a connection after segment i
    - If disconnected, consume two points: end of current segment and start of next
    """
    if len(segment_data) < 2:
        return []


    segments = []
    it = 0  # Iterator for segment_data
    knot_idx = 1  # Skip first knot flag like C++ code

    # Start with first point
    start_point = segment_data[it]
    it += 1

    while knot_idx < len(knot_flags) and it < len(segment_data):
        # Get end point of current segment
        end_point = segment_data[it]
        it += 1

        # Create line segment
        line = Line()
        line.link_two_points(start_point, end_point)

        # The knot flag indicates if this segment is connected to the next
        connected = knot_flags[knot_idx]

        segment = Segment(
            start_time=start_point.t,
            end_time=end_point.t,
            line=line,
        )
        segments.append(segment)

        if connected:
            # Next segment starts where this one ends
            start_point = end_point
        else:
            # Disconnected: next segment starts at next point
            if it < len(segment_data):
                start_point = segment_data[it]
                it += 1

        knot_idx += 1

    # Handle last segment if there are remaining points
    if it < len(segment_data):
        end_point = segment_data[it]
        line = Line()
        line.link_two_points(start_point, end_point)

        segment = Segment(
            start_time=start_point.t,
            end_time=end_point.t,
            line=line,
        )
        segments.append(segment)

    return segments


def compress(uncompressed_values: List[float], error_bound: float) -> bytes:
    """
    Compress time series data using Mixed-PLA algorithm.

    Args:
        uncompressed_values: List of float values (timestamps are implicit: 0, 1, 2, ...)
        error_bound: Maximum allowed error (delta parameter)

    Returns:
        Compressed data as bytes
    """
    if not uncompressed_values:
        raise ValueError("Empty input")
    if error_bound <= 0:
        raise ValueError("Invalid error bound")

    # Adjust the error bound to avoid exceeding it during decompression
    adjusted_error_bound = error_bound - 1e-7


    # Convert implicit timestamps to DataPoint format
    data_points = [DataPoint(float(i), val) for i, val in enumerate(uncompressed_values)]

    # Run Mixed-PLA algorithm (delta = error_bound, eps = 0.0)
    mixed_pla = MixContApr(adjusted_error_bound, 0.0)
    segments, knot_flags = mixed_pla.run(data_points)

    # Serialize to bytes
    return serialize_segments(segments, knot_flags, len(uncompressed_values))


def decompress(compressed_bytes: bytes) -> List[float]:
    """
    Decompress time series data from Mixed-PLA compressed format.

    Args:
        compressed_bytes: Compressed data as bytes

    Returns:
        Reconstructed time series values
    """
    # Deserialize from bytes
    segment_data, knot_flags, original_length = deserialize_segments(compressed_bytes)

    # print(f"Printing all DataPoints")
    # for s in segment_data:
    #     print(s)

    # Read segment_data ant transform into segments
    segments = read_segments(segment_data, knot_flags)

    # Reconstruct full time series
    return reconstruct_time_series(segments, knot_flags, original_length)

def serialize_segments(segments: List[DataPoint], knot_flags: List[bool], original_length: int) -> bytes:
    """
    Serialize segments and connection flags to compact binary format.

    Format: [num_segments][segment_data...][num_flags][knot_flags...][original_length]
    """
    data = bytearray()

    # Number of segments
    data.extend(struct.pack('<I', len(segments)))  # 4 bytes, little-endian

    # Segment data (time, value pairs)
    for seg in segments:
        data.extend(struct.pack('<d', seg.t))  # 8 bytes for time
        data.extend(struct.pack('<d', seg.m))  # 8 bytes for value

    # Number of knot flags
    data.extend(struct.pack('<I', len(knot_flags)))  # 4 bytes

    # Connection flags (packed as bits into bytes for efficiency)
    flag_bytes = pack_flags_to_bytes(knot_flags)
    data.extend(struct.pack('<I', len(flag_bytes)))  # 4 bytes for flag byte length
    data.extend(flag_bytes)

    # Original length for reconstruction
    data.extend(struct.pack('<I', original_length))  # 4 bytes

    return bytes(data)


def pack_flags_to_bytes(flags: List[bool]) -> bytes:
    """Pack boolean flags into bytes (8 flags per byte)."""
    byte_array = bytearray()

    for i in range(0, len(flags), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(flags) and flags[i + j]:
                byte_val |= (1 << j)
        byte_array.append(byte_val)

    return bytes(byte_array)


def deserialize_segments(data: bytes) -> Tuple[List[DataPoint], List[bool], int]:
    """
    Deserialize segments and connection flags from binary format.
    """
    offset = 0

    # Number of segments
    num_segments = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    # Segment data
    segments = []
    for _ in range(num_segments):
        t = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        m = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        segments.append(DataPoint(t, m))

    # Number of knot flags
    num_flags = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    # Flag bytes length
    flag_bytes_len = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    # Connection flags
    flag_bytes = data[offset:offset + flag_bytes_len]
    offset += flag_bytes_len
    knot_flags = unpack_flags_from_bytes(flag_bytes, num_flags)

    # Original length
    original_length = struct.unpack_from('<I', data, offset)[0]

    return segments, knot_flags, original_length


def unpack_flags_from_bytes(flag_bytes: bytes, num_flags: int) -> List[bool]:
    """Unpack boolean flags from bytes."""
    flags = []

    for byte_idx, byte_val in enumerate(flag_bytes):
        for bit_idx in range(8):
            if len(flags) < num_flags:
                flags.append(bool(byte_val & (1 << bit_idx)))

    return flags

# def reconstruct_time_series(segments: List[Segment], knot_flags: List[bool], original_length: int) -> List[float]:
#     """
#     Reconstruct the full time series from segments.
#
#     For each integer timestamp, find the segment that contains it and evaluate
#     the segment's line at that timestamp.
#     """
#     if not segments:
#         return [0.0] * original_length
#
#     decompressed_values = [0.0] * original_length
#
#     # For each integer timestamp
#     for decompressed_values_idx in range(original_length):
#         t = float(decompressed_values_idx)
#         for segment_idx in range(0, len(segments)):
#             if knot_flags[decompressed_values_idx]: # connected segment, use the segment_start_time and segment_end_time
#                 current_segment = segments[segment_idx]
#                 if current_segment.start_time <= t <= current_segment.end_time:
#                     decompressed_values[decompressed_values_idx] = current_segment.line.y(t)
#                 segment_idx += 1
#
#
#
#         #     for segment in segments:
#         #         if segment.start_time <= t <= segment.end_time:
#         #             result[i] = segment.line.y(t)
#         #             break
#         # else: # disonnected segment, use
#
#
#     return result

# Exprimental function that gets the second instance.  prioritize the segment that starts at that timestamp rather than the one that ends at it.
def reconstruct_time_series(segments: List[Segment], knot_flags: List[bool], original_length: int) -> List[float]:
    """
    Reconstruct the full time series from segments.
    When a timestamp fits in multiple segments, prefer the segment that starts at that timestamp.
    """
    if not segments:
        return [0.0] * original_length

    result = [0.0] * original_length

    # For each integer timestamp
    for i in range(original_length):
        t = float(i)
        segment_found = False # DEBUG
        # Find all segments that contain this timestamp
        matching_segments = []
        for segment in segments:
            if segment.start_time <= t <= segment.end_time:
                matching_segments.append(segment)

        if matching_segments:
            # Prefer the segment that starts at this timestamp
            preferred_segment = None
            for segment in matching_segments:
                if abs(segment.start_time - t) < 1e-10:  # segment starts at this timestamp
                    preferred_segment = segment
                    segment_found = True # DEBUG
                    break

            # If no segment starts at this timestamp, use the first matching segment
            if preferred_segment is None:
                preferred_segment = matching_segments[0]
                segment_found = True  # DEBUG

            result[i] = preferred_segment.line.y(t)

            if not segment_found:
                print(f"WARNING: No segment covers position {i} (t={t})")
                # Keep as 0.0 but log the issue

    return result

#
# # NOTE: This reconstruct function gets the fist instance where the point fits.
# def reconstruct_time_series(segments: List[Segment], knot_flags: List[bool], original_length: int) -> List[float]:
#     """
#     Reconstruct the full time series from segments.
#
#     For each integer timestamp, find the segment that contains it and evaluate
#     the segment's line at that timestamp.
#     """
#     if not segments:
#         return [0.0] * original_length
#
#     result = [0.0] * original_length
#
#
#     # For each integer timestamp
#     for i in range(original_length):
#         t = float(i)
#
#         knot_flag = False
#
#         # Find the segment that contains this timestamp
#         for segment in segments:
#             if segment.start_time <= t <= segment.end_time:
#                 result[i] = segment.line.y(t)
#
#     return result
# #
# def reconstruct_time_series(segments: List[Segment], knot_flags: List[bool], original_length: int) -> List[float]:
#     """
#     Reconstruct the full time series from segments.
#
#     Since segments are already sorted, we can directly iterate through each segment's
#     time range and fill the result array using the rounded times as indices.
#     """
#     if not segments:
#         return [0.0] * original_length
#
#     result = [0.0] * original_length
#
#     # Process each segment in order
#     for segment in segments:
#         start_idx = round(segment.start_time)
#         end_idx = min(round(segment.end_time), original_length - 1)
#
#         # Fill all time points covered by this segment
#         for i in range(start_idx, end_idx + 1):
#             if i < original_length:
#                 result[i] = segment.line.y(float(i))
#
#     return result
# def create_line_segments(segments: List[DataPoint], knot_flags: List[bool]) -> List[Segment]:
#     """
#     Convert knot points and connection flags into line segments.
#     """
#     if len(segments) < 2:
#         return []
#
#     line_segments = []
#     i = 0
#     knot_idx = 0
#
#     while i < len(segments) - 1:
#         start_point = segments[i]
#
#         # For connected segments, the next point is simply segments[i+1]
#         # For disconnected segments, we need to handle the gap
#         if knot_idx < len(knot_flags) and knot_flags[knot_idx]:
#             # Connected segment
#             end_point = segments[i + 1]
#
#             line = Line()
#             line.link_two_points(start_point, end_point)
#
#             segment = Segment(
#                 start_time=start_point.t,
#                 end_time=end_point.t,
#                 line=line,
#                 connected=True
#             )
#             line_segments.append(segment)
#             i += 1
#         else:
#             # Disconnected segment
#             # The segment ends at the same point (vertical line at the knot)
#             # and the next segment starts at segments[i+2]
#             end_point = segments[i + 1]
#
#             line = Line()
#             line.link_two_points(start_point, end_point)
#
#             segment = Segment(
#                 start_time=start_point.t,
#                 end_time=end_point.t,
#                 line=line,
#                 connected=False
#             )
#             line_segments.append(segment)
#
#             # Skip the disconnection point
#             i += 2
#
#         knot_idx += 1
#
#     return line_segments
def create_line_segments(segments: List[DataPoint], knot_flags: List[bool]) -> List[Segment]:
    """
    Convert knot points and connection flags into line segments.
    This function matches the logic used in decompress_segments.
    """
    if len(segments) < 2:
        return []

    line_segments = []
    i = 0

    # Process each consecutive pair of points
    while i < len(segments) - 1:
        start_point = segments[i]
        end_point = segments[i + 1]

        # Create line segment
        line = Line()
        line.link_two_points(start_point, end_point)

        # Determine connection flag for this segment
        connected = True
        if i < len(knot_flags):
            connected = knot_flags[i]

        segment = Segment(
            start_time=start_point.t,
            end_time=end_point.t,
            line=line,
            connected=connected
        )
        line_segments.append(segment)

        # Move to next segment
        if connected:
            # Connected: next segment starts where this one ends
            i += 1
        else:
            # Disconnected: skip the gap, next segment starts 2 points ahead
            i += 2

    return line_segments

import numpy as np

def check_error_bound(original, decompressed, error_bound, epsilon=1e-7):
    original = np.array(original)
    decompressed = np.array(decompressed)
    errors = np.abs(original - decompressed)
    mask = np.greater(errors, error_bound + epsilon)

    if np.any(mask):
        indices = np.where(mask)[0]
        print(f"ERROR: Found {len(indices)} positions truly exceeding error bound:")
        for i in indices:
            print(f"  Position {i}: orig={original[i]:.6f}, decomp={decompressed[i]:.6f}, error={errors[i]:.6f}")
    else:
        print("SUCCESS: All decompressed values are within the error bound.")

# Example usage
if __name__ == "__main__":
    # Create test data
    #test_data = [0.0, 0.24759351997749793, 0.4809905674925789, 0.6871836614750813, 0.8554378852922637, 0.9781730319062303, 1.051566288834133, 1.075822370562128, 1.0550876811750982, 0.9970168497443985, 0.9120309888378301, 0.8123345129898113, 0.7107788643198003, 0.6196751058252061, 0.5496618437816662, 0.5087299277715059, 0.5014912986235848, 0.5287574692886023, 0.5874653871002014, 0.6709573163885957, 0.7695896777262188, 0.8716163168910486, 0.964267085576279, 1.0349250943708197, 1.072297112475727, 1.0674721324913259, 1.0147730445087666, 0.912324785656373, 0.7622876042000448, 0.5707339442550736, 0.3471792506807455, 0.10380787418682222, -0.1455375340390699, -0.38651364103090935, -0.6054784458236029, -0.7906311076754549, -0.9329885583280981, -1.0271124350104879, -1.071522153775057, -1.0687579511434908, -1.0250889543081456, -0.9498930067473121, -0.8547642488030198, -0.7524287171806083, -0.6555653165359046, -0.5756379043892865, -0.5218431689114125, -0.500268563190142, -0.5133357205612016, -0.5595791970528654]
    #test_data = [0, 1, 2, 10, 11, 12]
    # test_data = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 10.0, 10.3, 10.6, 10.9, 11.2, 11.5, 12.3, 13.1, 13.9,
    #           14.7, 15.5, 16.3, 17.1, 17.9, 18.7]
    test_data = [0.0, 0.9092974268256817, -0.7568024953079282, -0.27941549819892586, 0.9893582466233818,
              -0.5440211108893698, -0.5365729180004349, 0.9906073556948704, -0.2879033166650653, -0.750987246771676,
              0.9129452507276277, -0.008851309290403876, -0.9055783620066239, 0.7625584504796027, 0.27090578830786904,
              -0.9880316240928618, 0.5514266812416906, 0.5290826861200238, -0.9917788534431158, 0.2963685787093853,
              0.7451131604793488, -0.9165215479156338, 0.017701925105413577, 0.9017883476488092, -0.7682546613236668,
              -0.26237485370392877, 0.9866275920404853, -0.5587890488516163, -0.5215510020869119, 0.9928726480845371,
              -0.3048106211022167, -0.7391806966492228, 0.9200260381967906, -0.026551154023966794, -0.8979276806892913,
              0.7738906815578891, 0.25382336276203626, -0.9851462604682474, 0.5661076368981803, 0.5139784559875352,
              -0.9938886539233752, 0.31322878243308516, 0.7331903200732922, -0.9234584470040598, 0.03539830273366068,
              0.8939966636005579, -0.7794660696158047, -0.24525198546765434, 0.9835877454343449, -0.5733818719904229]

    error_bound = 0.01

    serialized_segments = compress(test_data, error_bound)

    decompressed_values = decompress(serialized_segments)


    print(f"Uncompressed values: {test_data}")
    print(f"Decompressed values: {decompressed_values}")
    print()

    check_error_bound(test_data, decompressed_values, error_bound)
