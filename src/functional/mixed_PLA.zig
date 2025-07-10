// Copyright 2025 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Implementation of the Mixed-PLA algorithm from the paper
//! "G. Luo, K. Yi, S.-W. Cheng, Z. Li, W. Fan, C. He, and Y. Mu.
//! Piecewise Linear Approximation of Streaming Time Series Data with Max‑Error Guarantees.
//! Proc. IEEE 31st Int'l Conf. Data Engineering (ICDE)*, Seoul, South Korea 2015, pp. 173–184.
//! https://ieeexplore.ieee.org/document/7113282".

const std = @import("std");
const math = std.math;
const ArrayList = std.ArrayList;
const mem = std.mem;

const tersets = @import("../tersets.zig");
const shared = @import("../utilities/shared_structs.zig");
const tester = @import("../tester.zig");
const ch = @import("../utilities/convex_hull.zig");

const Error = tersets.Error;

// Enhanced line representation that properly handles vertical lines
pub const Line = struct {
    is_vertical: bool,
    x_value: f64, // For vertical lines: the x-coordinate
    slope: f64, // For non-vertical lines
    intercept: f64, // For non-vertical lines

    // Create a line from two points
    pub fn fromPoints(p1: shared.DiscretePoint, p2: shared.DiscretePoint) Line {
        if (p1.time == p2.time) {
            // Vertical line
            return .{
                .is_vertical = true,
                .x_value = @as(f64, @floatFromInt(p1.time)),
                .slope = 0,
                .intercept = 0,
            };
        }

        // Non-vertical line
        const x1 = @as(f64, @floatFromInt(p1.time));
        const x2 = @as(f64, @floatFromInt(p2.time));
        const slope = (p2.value - p1.value) / (x2 - x1);
        const intercept = p1.value - slope * x1;

        return .{
            .is_vertical = false,
            .x_value = 0,
            .slope = slope,
            .intercept = intercept,
        };
    }

    // Evaluate the line at a given x-coordinate
    pub fn evaluate(self: Line, x: f64) ?f64 {
        if (self.is_vertical) {
            // Vertical line only has a value at its x-coordinate
            if (@abs(x - self.x_value) < 1e-10) {
                // At the vertical line, any y-value is valid
                return null; // Indicates undefined
            }
            return null;
        }
        return self.slope * x + self.intercept;
    }

    // Check if a point is above the line
    pub fn isPointAbove(self: Line, point: shared.DiscretePoint) bool {
        const x = @as(f64, @floatFromInt(point.time));
        if (self.is_vertical) {
            // For vertical lines, "above" doesn't make sense unless at the same x
            return false;
        }
        const line_y = self.evaluate(x) orelse return false;
        return point.value > line_y + 1e-10;
    }

    // Check if a point is below the line
    pub fn isPointBelow(self: Line, point: shared.DiscretePoint) bool {
        const x = @as(f64, @floatFromInt(point.time));
        if (self.is_vertical) {
            // For vertical lines, "below" doesn't make sense unless at the same x
            return false;
        }
        const line_y = self.evaluate(x) orelse return false;
        return point.value < line_y - 1e-10;
    }
};

pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    // Apply error bound margin for numerical stability.
    const adjusted_error = error_bound - shared.ErrorBoundMargin;
    if (adjusted_error <= 0) {
        return Error.UnsupportedErrorBound;
    }

    // Check if we can fit a single line through all points
    var single_line_possible = true;
    var min_slope: f64 = -math.inf(f64);
    var max_slope: f64 = math.inf(f64);

    for (uncompressed_values[1..], 1..) |value, i| {
        const upper_slope = (value + adjusted_error - (uncompressed_values[0] - adjusted_error)) / @as(f64, @floatFromInt(i));
        const lower_slope = (value - adjusted_error - (uncompressed_values[0] + adjusted_error)) / @as(f64, @floatFromInt(i));

        max_slope = @min(max_slope, upper_slope);
        min_slope = @max(min_slope, lower_slope);

        if (min_slope > max_slope) {
            single_line_possible = false;
            break;
        }
    }

    if (single_line_possible) {
        // Use the middle slope for best average error
        const slope = (min_slope + max_slope) / 2.0;
        const intercept = uncompressed_values[0] - slope * 0.0;

        // Encode as special case: [1][slope][intercept][original_length]
        try compressed_values.append(1);
        try compressed_values.appendSlice(mem.asBytes(&slope));
        try compressed_values.appendSlice(mem.asBytes(&intercept));
        const orig_len = uncompressed_values.len;
        try compressed_values.appendSlice(mem.asBytes(&orig_len));
        return;
    }

    // Step 1: Create extended polygon.
    var polygon = try ExtendedPolygon.init(allocator, adjusted_error);
    defer polygon.deinit();

    // Add all data points to create the extended polygon.
    for (uncompressed_values, 0..) |value, i| {
        try polygon.addDataPoint(i, value);
    }

    // Get all windows that connect the upper and lower chains.
    const windows = try polygon.getAllWindows();
    defer windows.deinit();

    // Get initial and final windows from the extended polygon.
    const initial_window = polygon.getInitialWindow();
    const final_window = polygon.getFinalWindow();

    // Step 2: Initialize DP arrays.
    var C = ArrayList(Window).init(allocator);
    defer C.deinit();
    var pred = ArrayList(?usize).init(allocator);
    defer pred.deinit();

    // Step 3: Compute C[0] = cw(w0) where w0 is the initial window.
    var initial_vr = try VisibleRegion.init(allocator, initial_window);
    defer initial_vr.deinit();

    // Process windows to find the closing window of the initial window.
    for (windows.items[1..]) |window| {
        try initial_vr.updateWithNewWindow(window);
        if (initial_vr.is_closed) {
            try C.append(initial_vr.closing_window.?);
            try pred.append(null);
            break;
        }
    }

    // If initial window can see to the end, we're done
    if (!initial_vr.is_closed) {
        // Direct line from start to end
        const knot_time = final_window.upper_point.time;
        const knot_y = (final_window.upper_point.value + final_window.lower_point.value) / 2.0;

        const num_knots: u8 = 1;
        try compressed_values.append(num_knots);

        const time_bytes = mem.asBytes(&knot_time);
        const y_bytes = mem.asBytes(&knot_y);
        try compressed_values.appendSlice(time_bytes);
        try compressed_values.appendSlice(y_bytes);
        return;
    }

    // Step 4: DP loop following Algorithm 1 from the paper
    var k: usize = 0;
    while (true) {
        // Base cases for small k values
        if (k == 0) {
            // Already handled above
            k = 1;
            continue;
        } else if (k == 1) {
            // C[2] = cw(C[0])
            if (C.items.len > 0) {
                const c_0 = C.items[0];
                var vr_c_0 = try VisibleRegion.init(allocator, c_0);
                defer vr_c_0.deinit();

                // Find the starting point for processing
                var start_idx: usize = 0;
                for (windows.items, 0..) |w, idx| {
                    if (w.upper_point.time >= c_0.upper_point.time) {
                        start_idx = idx;
                        break;
                    }
                }

                for (windows.items[start_idx..]) |window| {
                    try vr_c_0.updateWithNewWindow(window);
                    if (vr_c_0.is_closed) {
                        break;
                    }
                }

                if (vr_c_0.is_closed) {
                    try C.append(vr_c_0.closing_window.?);
                    try pred.append(0);
                } else {
                    // Can reach the end
                    break;
                }
            }
        } else if (k == 2) {
            // C[3] = cw(nw(C[0]))
            if (C.items.len > 0) {
                const c_0 = C.items[0];
                const nw_c_0 = nextWindow(windows.items, c_0);
                if (nw_c_0) |next_win| {
                    var vr_nw_c_0 = try VisibleRegion.init(allocator, next_win);
                    defer vr_nw_c_0.deinit();

                    // Find the starting point for processing
                    var start_idx: usize = 0;
                    for (windows.items, 0..) |w, idx| {
                        if (w.upper_point.time >= next_win.upper_point.time) {
                            start_idx = idx;
                            break;
                        }
                    }

                    for (windows.items[start_idx..]) |window| {
                        try vr_nw_c_0.updateWithNewWindow(window);
                        if (vr_nw_c_0.is_closed) {
                            break;
                        }
                    }

                    if (vr_nw_c_0.is_closed) {
                        try C.append(vr_nw_c_0.closing_window.?);
                        try pred.append(0);
                    } else {
                        // Can reach the end
                        break;
                    }
                }
            }
        } else {
            // General case: C[k] = max(cw(C[k-2]), cw(nw(C[k-3])))
            const c_k_minus_2_idx = if (k >= 2) k - 2 else null;
            const c_k_minus_3_idx = if (k >= 3) k - 3 else null;

            var cw_c_k_minus_2: ?Window = null;
            var cw_nw_c_k_minus_3: ?Window = null;
            var can_reach_end = false;

            // Compute cw(C[k-2])
            if (c_k_minus_2_idx) |idx| {
                if (idx < C.items.len) {
                    const c = C.items[idx];
                    var vr = try VisibleRegion.init(allocator, c);
                    defer vr.deinit();

                    var start_idx: usize = 0;
                    for (windows.items, 0..) |w, i| {
                        if (w.upper_point.time >= c.upper_point.time) {
                            start_idx = i;
                            break;
                        }
                    }

                    for (windows.items[start_idx..]) |window| {
                        try vr.updateWithNewWindow(window);
                        if (vr.is_closed) {
                            break;
                        }
                    }

                    if (vr.is_closed) {
                        cw_c_k_minus_2 = vr.closing_window.?;
                    } else {
                        can_reach_end = true;
                    }
                }
            }

            // Compute cw(nw(C[k-3]))
            if (c_k_minus_3_idx) |idx| {
                if (idx < C.items.len) {
                    const c = C.items[idx];
                    const nw_c = nextWindow(windows.items, c);
                    if (nw_c) |next_win| {
                        var vr = try VisibleRegion.init(allocator, next_win);
                        defer vr.deinit();

                        var start_idx: usize = 0;
                        for (windows.items, 0..) |w, i| {
                            if (w.upper_point.time >= next_win.upper_point.time) {
                                start_idx = i;
                                break;
                            }
                        }

                        for (windows.items[start_idx..]) |window| {
                            try vr.updateWithNewWindow(window);
                            if (vr.is_closed) {
                                break;
                            }
                        }

                        if (vr.is_closed) {
                            cw_nw_c_k_minus_3 = vr.closing_window.?;
                        } else {
                            can_reach_end = true;
                        }
                    }
                }
            }

            if (can_reach_end) {
                // One of the paths can reach the end
                break;
            }

            // Choose the rightmost window according to the recurrence
            if (cw_c_k_minus_2 != null and cw_nw_c_k_minus_3 != null) {
                if (isWindowCompletelyRightOf(cw_nw_c_k_minus_3.?, cw_c_k_minus_2.?)) {
                    try C.append(cw_nw_c_k_minus_3.?);
                    try pred.append(c_k_minus_3_idx.?);
                } else {
                    try C.append(cw_c_k_minus_2.?);
                    try pred.append(c_k_minus_2_idx.?);
                }
            } else if (cw_c_k_minus_2 != null) {
                try C.append(cw_c_k_minus_2.?);
                try pred.append(c_k_minus_2_idx.?);
            } else if (cw_nw_c_k_minus_3 != null) {
                try C.append(cw_nw_c_k_minus_3.?);
                try pred.append(c_k_minus_3_idx.?);
            } else {
                // No valid next window found
                break;
            }
        }

        k += 1;

        // Safety check to prevent infinite loops.
        if (k > uncompressed_values.len) {
            break;
        }
    }

    // Step 5: Reconstruct the optimal PLA by walking backwards through pred[].
    var knots = ArrayList(KnotInfo).init(allocator);
    defer knots.deinit();

    // Trace back from the optimal k value
    var current_k: usize = C.items.len - 1;

    // Add the final knot
    try knots.append(KnotInfo{
        .time = uncompressed_values.len - 1,
        .is_joint = true,
        .y1 = uncompressed_values[uncompressed_values.len - 1],
        .y2 = uncompressed_values[uncompressed_values.len - 1],
    });

    // Trace backwards through the DP solution
    while (current_k > 0 and current_k < pred.items.len) {
        if (pred.items[current_k]) |prev_k| {
            const knot_window = C.items[current_k];
            const knot_time = knot_window.upper_point.time;

            if (current_k == prev_k + 2) {
                // Joint knot
                const knot_y = (knot_window.upper_point.value + knot_window.lower_point.value) / 2.0;
                try knots.append(KnotInfo{
                    .time = knot_time,
                    .is_joint = true,
                    .y1 = knot_y,
                    .y2 = knot_y,
                });
            } else if (current_k == prev_k + 3) {
                // Disjoint knot - need to compute both y-values
                // This is more complex and requires analyzing the segments
                // For now, use the window bounds
                try knots.append(KnotInfo{
                    .time = knot_time,
                    .is_joint = false,
                    .y1 = knot_window.lower_point.value,
                    .y2 = knot_window.upper_point.value,
                });
            }

            current_k = prev_k;
        } else {
            break;
        }
    }

    // Add the initial knot
    try knots.append(KnotInfo{
        .time = 0,
        .is_joint = true,
        .y1 = uncompressed_values[0],
        .y2 = uncompressed_values[0],
    });

    // Reverse to get correct order
    mem.reverse(KnotInfo, knots.items);

    // Step 6: Encode the result
    const num_knots = knots.items.len;
    try compressed_values.append(@intCast(num_knots));

    for (knots.items) |knot_info| {
        if (knot_info.is_joint) {
            // Joint knot: store as [x][y]
            const x_bytes = std.mem.asBytes(&knot_info.time);
            const y_bytes = std.mem.asBytes(&knot_info.y1);
            try compressed_values.appendSlice(x_bytes);
            try compressed_values.appendSlice(y_bytes);
        } else {
            // Disjoint knot: store as [-x][y1][y2]
            const neg_x = @as(i64, @intCast(knot_info.time)) * -1;
            const neg_x_bytes = std.mem.asBytes(&neg_x);
            const y1_bytes = std.mem.asBytes(&knot_info.y1);
            const y2_bytes = std.mem.asBytes(&knot_info.y2);
            try compressed_values.appendSlice(neg_x_bytes);
            try compressed_values.appendSlice(y1_bytes);
            try compressed_values.appendSlice(y2_bytes);
        }
    }
}

// Your existing decompress function is mostly fine, just ensure it handles the special case

// ExtendedPolygon implementation remains the same
const ExtendedPolygon = struct {
    data_points: ArrayList(shared.DiscretePoint),
    allocator: mem.Allocator,
    error_bound: f32,

    pub fn init(allocator: mem.Allocator, error_bound: f32) !ExtendedPolygon {
        return .{
            .data_points = ArrayList(shared.DiscretePoint).init(allocator),
            .allocator = allocator,
            .error_bound = error_bound,
        };
    }

    pub fn deinit(self: *ExtendedPolygon) void {
        self.data_points.deinit();
    }

    pub fn addDataPoint(self: *ExtendedPolygon, time: usize, value: f64) !void {
        try self.data_points.append(shared.DiscretePoint{ .time = time, .value = value });
    }

    pub fn getAllWindows(self: *ExtendedPolygon) !ArrayList(Window) {
        var windows = ArrayList(Window).init(self.allocator);

        for (self.data_points.items) |point| {
            const window = Window.initVertical(
                point.time,
                point.value + self.error_bound,
                point.value - self.error_bound,
            );
            try windows.append(window);
        }

        return windows;
    }

    pub fn getInitialWindow(self: *ExtendedPolygon) Window {
        const first_point = self.data_points.items[0];
        return Window.initVertical(
            first_point.time,
            first_point.value + self.error_bound,
            first_point.value - self.error_bound,
        );
    }

    pub fn getFinalWindow(self: *ExtendedPolygon) Window {
        const last_point = self.data_points.items[self.data_points.items.len - 1];
        return Window.initVertical(
            last_point.time,
            last_point.value + self.error_bound,
            last_point.value - self.error_bound,
        );
    }
};

// KnotInfo structure
const KnotInfo = struct {
    time: usize,
    is_joint: bool,
    y1: f64,
    y2: f64,
};

// Window structure
pub const Window = struct {
    upper_point: shared.DiscretePoint,
    lower_point: shared.DiscretePoint,

    pub fn init(upper_point: shared.DiscretePoint, lower_point: shared.DiscretePoint) Window {
        return .{
            .upper_point = upper_point,
            .lower_point = lower_point,
        };
    }

    pub fn initVertical(time: usize, upper_value: f64, lower_value: f64) Window {
        return .{
            .upper_point = .{ .time = time, .value = upper_value },
            .lower_point = .{ .time = time, .value = lower_value },
        };
    }

    pub fn isValid(self: Window) bool {
        if (self.upper_point.time == self.lower_point.time) {
            return self.upper_point.value >= self.lower_point.value;
        }
        return true;
    }
};

// Improved VisibleRegion with better vertical line handling
pub const VisibleRegion = struct {
    source_window: Window,
    upper_boundary_hull: ArrayList(shared.DiscretePoint),
    lower_boundary_hull: ArrayList(shared.DiscretePoint),
    z_plus: ?Line,
    z_minus: ?Line,
    l_plus: ?shared.DiscretePoint,
    r_plus: ?shared.DiscretePoint,
    l_minus: ?shared.DiscretePoint,
    r_minus: ?shared.DiscretePoint,
    closing_window: ?Window,
    is_closed: bool,
    allocator: mem.Allocator,

    pub fn init(allocator: mem.Allocator, window: Window) !VisibleRegion {
        var result = VisibleRegion{
            .source_window = window,
            .upper_boundary_hull = ArrayList(shared.DiscretePoint).init(allocator),
            .lower_boundary_hull = ArrayList(shared.DiscretePoint).init(allocator),
            .z_plus = null,
            .z_minus = null,
            .l_plus = null,
            .r_plus = null,
            .l_minus = null,
            .r_minus = null,
            .closing_window = null,
            .is_closed = false,
            .allocator = allocator,
        };

        // Initialize with the source window endpoints
        try result.upper_boundary_hull.append(window.upper_point);
        try result.lower_boundary_hull.append(window.lower_point);

        // Don't initialize supporting lines for vertical windows
        // They will be computed when we have enough non-collinear points

        return result;
    }

    pub fn deinit(self: *VisibleRegion) void {
        self.upper_boundary_hull.deinit();
        self.lower_boundary_hull.deinit();
    }

    pub fn updateWithNewWindow(self: *VisibleRegion, new_window: Window) !void {
        if (self.is_closed) return;

        const window_time = new_window.upper_point.time;
        const source_time = self.source_window.upper_point.time;

        if (window_time <= source_time) return;

        // Add points to hulls
        try self.addToUpperBoundary(new_window.upper_point);
        try self.addToLowerBoundary(new_window.lower_point);

        // Update supporting lines after we have enough points
        if (self.upper_boundary_hull.items.len >= 2 and self.lower_boundary_hull.items.len >= 2) {
            try self.updateSupportingLines();

            // Check for closure after updating supporting lines
            if (self.z_plus != null and self.z_minus != null) {
                const z_plus = self.z_plus.?;
                const z_minus = self.z_minus.?;

                // Check if the boundaries have crossed
                if (!z_plus.is_vertical and !z_minus.is_vertical) {
                    // Check at the new window's time
                    const time_f64 = @as(f64, @floatFromInt(window_time));
                    const z_plus_val = z_plus.evaluate(time_f64) orelse 0;
                    const z_minus_val = z_minus.evaluate(time_f64) orelse 0;

                    // If z_minus is above z_plus, the region closes
                    if (z_minus_val > z_plus_val + 1e-10) {
                        self.is_closed = true;
                        // The closing window is where the lines intersect
                        const intersect_x = (z_plus.intercept - z_minus.intercept) / (z_minus.slope - z_plus.slope);
                        const intersect_y = z_plus.slope * intersect_x + z_plus.intercept;

                        self.closing_window = Window{
                            .upper_point = .{ .time = @intFromFloat(intersect_x), .value = intersect_y },
                            .lower_point = .{ .time = @intFromFloat(intersect_x), .value = intersect_y },
                        };
                        return;
                    }

                    // Check if new window violates visibility
                    if (new_window.lower_point.value > z_plus_val + 1e-10) {
                        self.is_closed = true;
                        self.closing_window = Window{
                            .upper_point = .{ .time = window_time, .value = z_plus_val },
                            .lower_point = .{ .time = window_time, .value = z_plus_val },
                        };
                        return;
                    }

                    if (new_window.upper_point.value < z_minus_val - 1e-10) {
                        self.is_closed = true;
                        self.closing_window = Window{
                            .upper_point = .{ .time = window_time, .value = z_minus_val },
                            .lower_point = .{ .time = window_time, .value = z_minus_val },
                        };
                        return;
                    }
                }
            }
        }
    }

    fn addToUpperBoundary(self: *VisibleRegion, point: shared.DiscretePoint) !void {
        try addToHullWithTurn(&self.upper_boundary_hull, .right, point);
    }

    fn addToLowerBoundary(self: *VisibleRegion, point: shared.DiscretePoint) !void {
        try addToHullWithTurn(&self.lower_boundary_hull, .left, point);
    }

    pub fn addToHullWithTurn(hull: *ArrayList(shared.DiscretePoint), turn: ch.Turn, point: shared.DiscretePoint) !void {
        if (hull.items.len < 2) {
            try hull.append(point);
        } else {
            var top: usize = hull.items.len - 1;
            while ((top > 0) and (ch.computeTurn(
                hull.items[top - 1],
                hull.items[top],
                point,
            ) != turn)) : (top -= 1) {
                _ = hull.pop();
            }
            try hull.append(point);
        }
    }

    pub fn updateSupportingLines(self: *VisibleRegion) !void {
        if (self.upper_boundary_hull.items.len < 2 or self.lower_boundary_hull.items.len < 2) {
            return;
        }

        var max_slope: f64 = -std.math.inf(f64);
        var min_slope: f64 = std.math.inf(f64);
        var max_slope_line: ?Line = null;
        var min_slope_line: ?Line = null;
        var max_upper_idx: usize = 0;
        var max_lower_idx: usize = 0;
        var min_upper_idx: usize = 0;
        var min_lower_idx: usize = 0;

        // Try all combinations of points from upper and lower hulls
        for (self.upper_boundary_hull.items, 0..) |upper_point, u_idx| {
            for (self.lower_boundary_hull.items, 0..) |lower_point, l_idx| {
                // Skip if points are at the same x-coordinate
                if (upper_point.time == lower_point.time) continue;

                const line = Line.fromPoints(upper_point, lower_point);

                if (line.is_vertical) continue;

                // Check if this line validly separates the hulls
                if (self.isValidSeparatingLine(line)) {
                    // Update max slope line
                    if (line.slope > max_slope) {
                        max_slope = line.slope;
                        max_slope_line = line;
                        max_upper_idx = u_idx;
                        max_lower_idx = l_idx;
                    }

                    // Update min slope line
                    if (line.slope < min_slope) {
                        min_slope = line.slope;
                        min_slope_line = line;
                        min_upper_idx = u_idx;
                        min_lower_idx = l_idx;
                    }
                }
            }
        }

        // Update z+ (max slope)
        if (max_slope_line) |line| {
            self.z_plus = line;
            // Determine which hull each tangent point belongs to
            if (self.upper_boundary_hull.items[max_upper_idx].time < self.lower_boundary_hull.items[max_lower_idx].time) {
                self.l_plus = self.upper_boundary_hull.items[max_upper_idx];
                self.r_plus = self.lower_boundary_hull.items[max_lower_idx];
            } else {
                self.l_plus = self.lower_boundary_hull.items[max_lower_idx];
                self.r_plus = self.upper_boundary_hull.items[max_upper_idx];
            }
        }

        // Update z- (min slope)
        if (min_slope_line) |line| {
            self.z_minus = line;
            // Determine which hull each tangent point belongs to
            if (self.upper_boundary_hull.items[min_upper_idx].time < self.lower_boundary_hull.items[min_lower_idx].time) {
                self.l_minus = self.upper_boundary_hull.items[min_upper_idx];
                self.r_minus = self.lower_boundary_hull.items[min_lower_idx];
            } else {
                self.l_minus = self.lower_boundary_hull.items[min_lower_idx];
                self.r_minus = self.upper_boundary_hull.items[min_upper_idx];
            }
        }
    }

    fn isValidSeparatingLine(self: *VisibleRegion, line: Line) bool {

        // Check all upper hull points are above or on the line
        for (self.upper_boundary_hull.items) |point| {
            if (line.isPointBelow(point)) {
                return false;
            }
        }

        // Check all lower hull points are below or on the line
        for (self.lower_boundary_hull.items) |point| {
            if (line.isPointAbove(point)) {
                return false;
            }
        }

        return true;
    }
};

// Helper functions remain mostly the same
fn nextWindow(windows: []const Window, w: Window) ?Window {
    var min_time: usize = std.math.maxInt(usize);
    const w_time = w.upper_point.time;
    var res: ?Window = null;

    for (windows) |win| {
        const win_time = win.upper_point.time;
        if (win_time > w_time and win_time < min_time) {
            min_time = win_time;
            res = win;
        }
    }
    return res;
}

pub fn isWindowCompletelyRightOf(w1: Window, w2: Window) bool {
    const w1_time = w1.upper_point.time;
    const w2_time = w2.upper_point.time;
    return w1_time > w2_time;
}

// Decompress function remains the same as your original
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
    allocator: mem.Allocator,
) Error!void {
    if (compressed_values.len < 1) {
        return Error.InvalidData;
    }

    var offset: usize = 0;
    const size_f64 = @sizeOf(f64);
    const size_usize = @sizeOf(usize);

    // Read number of knots
    const num_knots = compressed_values[offset];
    offset += 1;

    // Handle special case: direct line (slope, intercept, original_length)
    if (num_knots == 1 and compressed_values.len == 1 + size_f64 + size_f64 + size_usize) {
        // This is the special case: [1][slope][intercept][original_length]
        if (offset + 2 * size_f64 + size_usize > compressed_values.len) {
            return Error.InvalidData;
        }

        const slope = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
        offset += size_f64;

        const intercept = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
        offset += size_f64;

        // Read the stored original length
        const orig_len = mem.bytesAsSlice(usize, compressed_values[offset .. offset + size_usize])[0];

        // Use the correct range [0..orig_len].
        for (0..orig_len) |t| {
            const value = slope * @as(f64, @floatFromInt(t)) + intercept;
            try decompressed_values.append(value);
        }
        return;
    }

    // Handle general case: multiple knots
    var knots = ArrayList(KnotInfo).init(allocator);
    defer knots.deinit();

    for (0..num_knots) |_| {
        // Read the first field (8 bytes) and determine if it's joint or disjoint based on sign
        if (offset + 8 > compressed_values.len) {
            return Error.InvalidData;
        }

        // Read as i64 to check sign, but be prepared to interpret as usize if positive
        const time_raw_bytes = compressed_values[offset .. offset + 8];
        const time_i64 = mem.bytesAsSlice(i64, time_raw_bytes)[0];
        offset += 8;

        if (time_i64 < 0) {
            // Disjoint knot: [-x][y1][y2]
            if (offset + 2 * size_f64 > compressed_values.len) {
                return Error.InvalidData;
            }

            const y1 = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;

            const y2 = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;

            try knots.append(KnotInfo{
                .time = @as(usize, @intCast(-time_i64)),
                .is_joint = false,
                .y1 = y1,
                .y2 = y2,
            });
        } else {
            // Joint knot: [x][y] - reinterpret as usize
            if (offset + size_f64 > compressed_values.len) {
                return Error.InvalidData;
            }

            const y = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;

            // For joint knots, the time was encoded as usize, so reinterpret the bytes
            const time_usize = mem.bytesAsSlice(usize, time_raw_bytes)[0];

            try knots.append(KnotInfo{
                .time = time_usize,
                .is_joint = true,
                .y1 = y,
                .y2 = y,
            });
        }
    }

    if (knots.items.len == 0) {
        return; // No knots, nothing to decompress
    }

    // Sort knots by time
    mem.sort(KnotInfo, knots.items, {}, struct {
        pub fn lessThan(context: void, a: KnotInfo, b: KnotInfo) bool {
            _ = context;
            return a.time < b.time;
        }
    }.lessThan);

    // Determine the time range
    const max_time = knots.items[knots.items.len - 1].time;

    // Reconstruct the piecewise linear function and evaluate at each time point
    for (0..max_time + 1) |t| {
        const value = evaluatePLAAtTime(knots.items, t);
        try decompressed_values.append(value);
    }
}

fn evaluatePLAAtTime(knots: []const KnotInfo, t: usize) f64 {
    if (knots.len == 0) {
        return 0.0;
    }

    if (knots.len == 1) {
        return knots[0].y1;
    }

    // Handle time before first knot
    if (t < knots[0].time) {
        // Extrapolate backwards from first segment
        if (knots.len > 1) {
            const dt = @as(f64, @floatFromInt(knots[1].time - knots[0].time));
            if (dt > 0) {
                const slope = (knots[1].y1 - knots[0].y1) / dt;
                const dt_back = @as(f64, @floatFromInt(knots[0].time - t));
                return knots[0].y1 - slope * dt_back;
            }
        }
        return knots[0].y1;
    }

    // Find the segment that contains time t
    for (0..knots.len) |i| {
        if (t <= knots[i].time) {
            if (i == 0) {
                return knots[0].y1;
            } else {
                // Linear interpolation between knots[i-1] and knots[i]
                const prev_knot = knots[i - 1];
                const curr_knot = knots[i];

                const t1 = prev_knot.time;
                const t2 = curr_knot.time;

                // For disjoint knots, use y2 from previous and y1 from current
                // For joint knots, use y1 from both
                const y1 = if (prev_knot.is_joint) prev_knot.y1 else prev_knot.y2;
                const y2 = curr_knot.y1;

                if (t2 == t1) {
                    return y2;
                }

                const fraction = @as(f64, @floatFromInt(t - t1)) / @as(f64, @floatFromInt(t2 - t1));
                return y1 + fraction * (y2 - y1);
            }
        }
    }

    // Time is beyond the last knot - extrapolate forward
    const last_knot = knots[knots.len - 1];
    if (knots.len > 1) {
        const prev_knot = knots[knots.len - 2];
        const dt = @as(f64, @floatFromInt(last_knot.time - prev_knot.time));
        if (dt > 0) {
            const y1 = if (prev_knot.is_joint) prev_knot.y1 else prev_knot.y2;
            const y2 = last_knot.y1;
            const slope = (y2 - y1) / dt;
            const dt_forward = @as(f64, @floatFromInt(t - last_knot.time));
            return last_knot.y1 + slope * dt_forward;
        }
    }

    return if (last_knot.is_joint) last_knot.y1 else last_knot.y2;
}
