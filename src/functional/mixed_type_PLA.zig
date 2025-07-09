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

//! Implementation of the Mix-Piece algorithm from the paper
//! “G. Luo, K. Yi, S.-W. Cheng, Z. Li, W. Fan, C. He, and Y. Mu.
//! Piecewise Linear Approximation of Streaming Time Series Data with Max‑Error Guarantees.
//! Proc. IEEE 31st Int’l Conf. Data Engineering (ICDE)*, Seoul, South Korea 2015, pp. 173–184.
//! https://ieeexplore.ieee.org/document/7113282”.
//!
//! The implementation is partially based on the author's implementation generously provided
//! by Prof. Ke Yi of Hong Kong University of Science and Technology.

const std = @import("std");
const stdout = std.io.getStdOut().writer();
const math = std.math;
const ArrayList = std.ArrayList;
const mem = std.mem;

const tersets = @import("../tersets.zig");
const shared = @import("../utilities/shared_structs.zig");
const tester = @import("../tester.zig");
const ch = @import("../utilities/convex_hull.zig");

const Error = tersets.Error;

pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    std.debug.print("=== COMPRESS START ===\n", .{});
    std.debug.print("Input length: {}, error_bound: {}\n", .{ uncompressed_values.len, error_bound });

    // Apply error bound margin for numerical stability.
    const adjusted_error = error_bound - shared.ErrorBoundMargin;
    std.debug.print("Adjusted error: {}\n", .{adjusted_error});

    if (adjusted_error <= 0) {
        std.debug.print("ERROR: Adjusted error <= 0\n", .{});
        return Error.UnsupportedErrorBound;
    }

    // Step 1: Create extended polygon.
    var polygon = try ExtendedPolygon.init(allocator, adjusted_error);
    defer polygon.deinit();

    std.debug.print("Created polygon with {} data points\n", .{polygon.data_points.items.len});

    // Add all data points to create the extended polygon.
    for (uncompressed_values, 0..) |value, i| {
        try polygon.addDataPoint(i, value);
    }

    std.debug.print("Added {} data points to polygon\n", .{polygon.data_points.items.len});

    // Get all windows that connect the upper and lower chains.
    const windows = try polygon.getAllWindows();
    defer windows.deinit();

    std.debug.print("Generated {} windows\n", .{windows.items.len});
    for (windows.items, 0..) |window, i| {
        std.debug.print("Window {}: upper=({}, {}), lower=({}, {})\n", .{ i, window.upper_point.time, window.upper_point.value, window.lower_point.time, window.lower_point.value });
    }

    // Get initial and final windows from the extended polygon.
    const initial_window = polygon.getInitialWindow();
    const final_window = polygon.getFinalWindow();

    std.debug.print("Initial window: upper=({}, {}), lower=({}, {})\n", .{ initial_window.upper_point.time, initial_window.upper_point.value, initial_window.lower_point.time, initial_window.lower_point.value });
    std.debug.print("Final window: upper=({}, {}), lower=({}, {})\n", .{ final_window.upper_point.time, final_window.upper_point.value, final_window.lower_point.time, final_window.lower_point.value });

    // Step 2: Initialize DP arrays.
    var C = ArrayList(Window).init(allocator);
    defer C.deinit();
    var pred = ArrayList(?usize).init(allocator);
    defer pred.deinit();

    // Step 3: Compute C[0] = cw(w0) where w0 is the initial window.
    var initial_vr = try VisibleRegion.init(allocator, initial_window);
    defer initial_vr.deinit();

    std.debug.print("Created initial visible region\n", .{});

    // Process windows to find the closing window of the initial window.
    var found_closing = false;
    for (windows.items[1..], 1..) |window, i| {
        std.debug.print("Processing window {} for initial VR\n", .{i});
        try initial_vr.updateWithNewWindow(window);

        std.debug.print("VR is_closed: {}\n", .{initial_vr.is_closed});

        if (initial_vr.is_closed) {
            try C.append(initial_vr.closing_window.?);
            try pred.append(null);
            found_closing = true;
            std.debug.print("Found closing window at index {}\n", .{i});
            break;
        }
    }

    printVisibleRegion(&initial_vr);

    if (!found_closing) {
        std.debug.print("SPECIAL CASE: Initial window can see final window directly\n", .{});

        // Try to create direct line from initial to final window
        if (canReachDirectly(initial_window, final_window)) {
            std.debug.print("Creating direct line segment\n", .{});

            const segment_slope = (final_window.upper_point.value - initial_window.upper_point.value) /
                @as(f64, @floatFromInt(final_window.upper_point.time - initial_window.upper_point.time));
            const segment_intercept = initial_window.upper_point.value -
                segment_slope * @as(f64, @floatFromInt(initial_window.upper_point.time));

            std.debug.print("Direct segment: slope={}, intercept={}\n", .{ segment_slope, segment_intercept });

            try compressed_values.append(1); // Number of segments
            try compressed_values.appendSlice(std.mem.asBytes(&segment_slope));
            try compressed_values.appendSlice(std.mem.asBytes(&segment_intercept));
            const orig_len: usize = uncompressed_values.len;
            try compressed_values.appendSlice(std.mem.asBytes(&orig_len));

            std.debug.print("Compressed to {} bytes (direct line)\n", .{compressed_values.items.len});
            return;
        } else {
            std.debug.print("ERROR: Cannot create direct line but VR not closed\n", .{});
            return Error.InvalidData;
        }
    }

    std.debug.print("Starting DP with C[0] computed\n", .{});

    // Step 3: Maintain list of visible regions for DP frontier
    var visible_regions = ArrayList(VisibleRegion).init(allocator);
    defer {
        for (visible_regions.items) |*vr| {
            vr.deinit();
        }
        visible_regions.deinit();
    }

    // Step 4: DP loop following Algorithm 1 from the paper
    var k: usize = 0;
    while (true) {
        // Base cases for small k values
        if (k == 0) {
            // C[1] is undefined, skip to k=1
            k = 1;
            continue;
        } else if (k == 1) {
            // C[2] = cw(C[0])
            if (C.items.len > 0) {
                const c_0 = C.items[0];
                var vr_c_0 = try VisibleRegion.init(allocator, c_0);
                defer vr_c_0.deinit();

                for (windows.items) |window| {
                    try vr_c_0.updateWithNewWindow(window);
                    if (vr_c_0.is_closed) {
                        break;
                    }
                }

                if (vr_c_0.is_closed) {
                    try C.append(vr_c_0.closing_window.?);
                    try pred.append(0);
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

                    for (windows.items) |window| {
                        try vr_nw_c_0.updateWithNewWindow(window);
                        if (vr_nw_c_0.is_closed) {
                            break;
                        }
                    }

                    if (vr_nw_c_0.is_closed) {
                        try C.append(vr_nw_c_0.closing_window.?);
                        try pred.append(0);
                    }
                }
            }
        } else {
            // General case: C[k] = max(cw(C[k-2]), cw(nw(C[k-3])))
            const c_k_minus_2_idx = if (k >= 2) k - 2 else null;
            const c_k_minus_3_idx = if (k >= 3) k - 3 else null;

            var cw_c_k_minus_2: ?Window = null;
            var cw_nw_c_k_minus_3: ?Window = null;

            // Compute cw(C[k-2])
            if (c_k_minus_2_idx) |idx| {
                if (idx < C.items.len) {
                    const c = C.items[idx];
                    var vr = try VisibleRegion.init(allocator, c);
                    defer vr.deinit();

                    for (windows.items) |window| {
                        try vr.updateWithNewWindow(window);
                        if (vr.is_closed) {
                            break;
                        }
                    }

                    if (vr.is_closed) {
                        cw_c_k_minus_2 = vr.closing_window.?;
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

                        for (windows.items) |window| {
                            try vr.updateWithNewWindow(window);
                            if (vr.is_closed) {
                                break;
                            }
                        }

                        if (vr.is_closed) {
                            cw_nw_c_k_minus_3 = vr.closing_window.?;
                        }
                    }
                }
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

        // Check if we've reached the final window
        if (k > 0 and k < C.items.len) {
            const current_window = C.items[k];
            var vr_current = try VisibleRegion.init(allocator, current_window);
            defer vr_current.deinit();

            try vr_current.updateWithNewWindow(final_window);
            if (!vr_current.is_closed) {
                // We can reach the final window, so k is the optimal size.
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
    var segments = ArrayList(shared.LinearFunction).init(allocator);
    defer segments.deinit();

    // Start from the final optimal window and trace backwards
    var current_k = k;
    var previous_window: ?Window = null;

    while (current_k > 0 and current_k < C.items.len and current_k < pred.items.len) {
        const current_window = C.items[current_k];

        if (pred.items[current_k]) |prev_k| {
            const prev_window = C.items[prev_k];

            if (current_k == prev_k + 2) {
                // Joint knot (2-step jump)
                // The segments meet at a single point on the closing window
                const knot_time = current_window.upper_point.time;
                const knot_y = current_window.upper_point.value;

                try knots.append(KnotInfo{
                    .time = knot_time,
                    .is_joint = true,
                    .y1 = knot_y,
                    .y2 = knot_y, // Same value for joint knots
                });

                // Add the segment from prev_window to current_window
                const segment = computeLine(prev_window.upper_point, current_window.upper_point);
                try segments.append(segment);
            } else if (current_k == prev_k + 3) {
                // Disjoint knot (3-step jump)
                // Need to compute both y-values for the discontinuity
                const knot_time = current_window.upper_point.time;

                // Find the next window after prev_window
                const next_window = nextWindow(windows.items, prev_window);

                if (next_window) |nw| {
                    // y1: where the left segment (ending at prev_window) intersects the vertical line at knot_time
                    // y2: where the right segment (starting at nw) intersects the vertical line at knot_time

                    // Compute left segment line
                    const left_segment = computeLine(if (previous_window) |pw| pw.upper_point else initial_window.upper_point, prev_window.upper_point);

                    // Compute right segment line
                    const right_segment = computeLine(nw.upper_point, current_window.upper_point);

                    // Evaluate both segments at knot_time
                    const knot_time_f64 = @as(f64, @floatFromInt(knot_time));
                    const y1 = left_segment.slope * knot_time_f64 + left_segment.intercept;
                    const y2 = right_segment.slope * knot_time_f64 + right_segment.intercept;

                    try knots.append(KnotInfo{
                        .time = knot_time,
                        .is_joint = false,
                        .y1 = y1,
                        .y2 = y2,
                    });

                    // Add both segments
                    try segments.append(left_segment);
                    try segments.append(right_segment);
                }
            }

            previous_window = current_window;
            current_k = prev_k;
        } else {
            break;
        }
    }

    // Reverse to get correct order
    mem.reverse(KnotInfo, knots.items);
    mem.reverse(shared.LinearFunction, segments.items);

    // Step 6: Encode the result into compressed_values
    // Format: [num_knots][knot_data...]
    // For joint knots: [x][y] (2 parameters)
    // For disjoint knots: [-x][y1][y2] (3 parameters, negative x indicates disjoint)

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
        std.debug.print(">>> decompress: segment_count = {d}\n", .{compressed_values.items.len});
    }
    std.debug.print("=== COMPRESS END ===\n", .{});
}

// Helper function to check if we can reach directly
fn canReachDirectly(initial: Window, final: Window) bool {
    // A very simple check - you might need to make this more sophisticated
    return initial.upper_point.time < final.upper_point.time;
}

pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
    allocator: mem.Allocator,
) Error!void {
    std.debug.print("=== DECOMPRESS START ===\n", .{});
    std.debug.print("Compressed size: {} bytes\n", .{compressed_values.len});

    if (compressed_values.len < 1) {
        std.debug.print("ERROR: Compressed data too short\n", .{});
        return Error.InvalidData;
    }

    var offset: usize = 0;
    const size_f64 = @sizeOf(f64);
    const size_usize = @sizeOf(usize);

    // Read number of knots
    const num_knots = compressed_values[offset];
    offset += 1;

    std.debug.print("Number of knots: {}\n", .{num_knots});

    // Handle special case: direct line (slope, intercept, original_length)
    if (num_knots == 1 and compressed_values.len == 1 + size_f64 + size_f64 + size_usize) {
        std.debug.print("Special case: direct line\n", .{});

        if (offset + 2 * size_f64 + size_usize > compressed_values.len) {
            std.debug.print("ERROR: Not enough data for direct line\n", .{});
            return Error.InvalidData;
        }

        const slope = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
        offset += size_f64;
        std.debug.print("Slope: {}\n", .{slope});

        const intercept = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
        offset += size_f64;
        std.debug.print("Intercept: {}\n", .{intercept});

        const orig_len = mem.bytesAsSlice(usize, compressed_values[offset .. offset + size_usize])[0];
        std.debug.print("Original length: {}\n", .{orig_len});

        // Generate decompressed values
        for (0..orig_len) |t| {
            const value = slope * @as(f64, @floatFromInt(t)) + intercept;
            try decompressed_values.append(value);
        }

        std.debug.print("Decompressed {} values\n", .{decompressed_values.items.len});
        return;
    }

    // Handle general case: multiple knots
    std.debug.print("General case: multiple knots\n", .{});

    var knots = ArrayList(KnotInfo).init(allocator);
    defer knots.deinit();

    for (0..num_knots) |knot_idx| {
        std.debug.print("Processing knot {}\n", .{knot_idx});

        if (offset + 8 > compressed_values.len) {
            std.debug.print("ERROR: Not enough data for knot {}\n", .{knot_idx});
            return Error.InvalidData;
        }

        const time_raw_bytes = compressed_values[offset .. offset + 8];
        const time_i64 = mem.bytesAsSlice(i64, time_raw_bytes)[0];
        offset += 8;

        std.debug.print("Time raw: {}\n", .{time_i64});

        if (time_i64 < 0) {
            std.debug.print("Disjoint knot\n", .{});

            if (offset + 2 * size_f64 > compressed_values.len) {
                std.debug.print("ERROR: Not enough data for disjoint knot\n", .{});
                return Error.InvalidData;
            }

            const y1 = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;
            const y2 = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;

            std.debug.print("Disjoint knot: time={}, y1={}, y2={}\n", .{ -time_i64, y1, y2 });

            try knots.append(KnotInfo{
                .time = @as(usize, @intCast(-time_i64)),
                .is_joint = false,
                .y1 = y1,
                .y2 = y2,
            });
        } else {
            std.debug.print("Joint knot\n", .{});

            if (offset + size_f64 > compressed_values.len) {
                std.debug.print("ERROR: Not enough data for joint knot\n", .{});
                return Error.InvalidData;
            }

            const y = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;

            const time_usize = mem.bytesAsSlice(usize, time_raw_bytes)[0];

            std.debug.print("Joint knot: time={}, y={}\n", .{ time_usize, y });

            try knots.append(KnotInfo{
                .time = time_usize,
                .is_joint = true,
                .y1 = y,
                .y2 = y,
            });
        }
    }

    if (knots.items.len == 0) {
        std.debug.print("No knots found, returning empty\n", .{});
        return;
    }

    // Sort knots by time
    std.mem.sort(KnotInfo, knots.items, {}, struct {
        pub fn lessThan(context: void, a: KnotInfo, b: KnotInfo) bool {
            _ = context;
            return a.time < b.time;
        }
    }.lessThan);

    std.debug.print("Sorted knots:\n", .{});
    for (knots.items, 0..) |knot, i| {
        std.debug.print("  {}: time={}, joint={}, y1={}, y2={}\n", .{ i, knot.time, knot.is_joint, knot.y1, knot.y2 });
    }

    // Determine the time range
    const max_time = knots.items[knots.items.len - 1].time;
    std.debug.print("Max time: {}\n", .{max_time});

    // Reconstruct the piecewise linear function
    for (0..max_time + 1) |t| {
        const value = evaluatePLAAtTime(knots.items, t);
        try decompressed_values.append(value);
        if (t < 5 or t > max_time - 5) {
            std.debug.print("  t={}: value={}\n", .{ t, value });
        }
    }

    std.debug.print("Decompressed {} values\n", .{decompressed_values.items.len});
    std.debug.print("=== DECOMPRESS END ===\n", .{});
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

// ****************************************************************************************

// Structure to represent the extended polygon
// Structure to represent the extended polygon
const ExtendedPolygon = struct {
    data_points: ArrayList(shared.DiscretePoint), // Original data points
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

    // Get the upper boundary value at a given time coordinate for segment i
    pub fn getUpperBoundary(self: *ExtendedPolygon, time: usize, segment_i: usize) f64 {
        if (segment_i >= self.data_points.items.len - 1) return std.math.nan(f64);

        if (segment_i == 0) {
            // For first segment, use line through pu_0 and pu_1
            const pu_0 = shared.DiscretePoint{ .time = self.data_points.items[0].time, .value = self.data_points.items[0].value + self.error_bound };
            const pu_1 = shared.DiscretePoint{ .time = self.data_points.items[1].time, .value = self.data_points.items[1].value + self.error_bound };
            const line = computeLine(pu_0, pu_1);
            return line.slope * @as(f64, @floatFromInt(time)) + line.intercept;
        } else {
            // For other segments, take max of two lines
            // Line 1: pl_{i-1} to pu_i
            const pl_i_minus_1 = shared.DiscretePoint{ .time = self.data_points.items[segment_i - 1].time, .value = self.data_points.items[segment_i - 1].value - self.error_bound };
            const pu_i = shared.DiscretePoint{ .time = self.data_points.items[segment_i].time, .value = self.data_points.items[segment_i].value + self.error_bound };
            const line1 = computeLine(pl_i_minus_1, pu_i);
            const val1 = line1.slope * @as(f64, @floatFromInt(time)) + line1.intercept;

            // Line 2: pu_i to pu_{i+1}
            const pu_i_plus_1 = shared.DiscretePoint{ .time = self.data_points.items[segment_i + 1].time, .value = self.data_points.items[segment_i + 1].value + self.error_bound };
            const line2 = computeLine(pu_i, pu_i_plus_1);
            const val2 = line2.slope * @as(f64, @floatFromInt(time)) + line2.intercept;

            return @max(val1, val2);
        }
    }

    // Get the lower boundary value at a given time coordinate for segment i
    pub fn getLowerBoundary(self: *ExtendedPolygon, time: usize, segment_i: usize) f64 {
        if (segment_i >= self.data_points.items.len - 1) return std.math.nan(f64);

        if (segment_i == 0) {
            // For first segment, use line through pl_0 and pl_1
            const pl_0 = shared.DiscretePoint{ .time = self.data_points.items[0].time, .value = self.data_points.items[0].value - self.error_bound };
            const pl_1 = shared.DiscretePoint{ .time = self.data_points.items[1].time, .value = self.data_points.items[1].value - self.error_bound };
            const line = computeLine(pl_0, pl_1);
            return line.slope * @as(f64, @floatFromInt(time)) + line.intercept;
        } else {
            // For other segments, take min of two lines
            // Line 1: pu_{i-1} to pl_i
            const pu_i_minus_1 = shared.DiscretePoint{ .time = self.data_points.items[segment_i - 1].time, .value = self.data_points.items[segment_i - 1].value + self.error_bound };
            const pl_i = shared.DiscretePoint{ .time = self.data_points.items[segment_i].time, .value = self.data_points.items[segment_i].value - self.error_bound };
            const line1 = computeLine(pu_i_minus_1, pl_i);
            const val1 = line1.slope * @as(f64, @floatFromInt(time)) + line1.intercept;

            // Line 2: pl_i to pl_{i+1}
            const pl_i_plus_1 = shared.DiscretePoint{ .time = self.data_points.items[segment_i + 1].time, .value = self.data_points.items[segment_i + 1].value - self.error_bound };
            const line2 = computeLine(pl_i, pl_i_plus_1);
            const val2 = line2.slope * @as(f64, @floatFromInt(time)) + line2.intercept;

            return @min(val1, val2);
        }
    }

    // Get all windows that connect the upper and lower chains at data points
    pub fn getAllWindows(self: *ExtendedPolygon) !ArrayList(Window) {
        var windows = ArrayList(Window).init(self.allocator);

        for (self.data_points.items, 0..) |point, i| {
            if (i == 0) {
                // Initial window is vertical at first data point
                const window = Window.initVertical(
                    point.time,
                    point.value + self.error_bound,
                    point.value - self.error_bound,
                );
                try windows.append(window);
            } else if (i == self.data_points.items.len - 1) {
                // Final window is vertical at last data point
                const window = Window.initVertical(
                    point.time,
                    point.value + self.error_bound,
                    point.value - self.error_bound,
                );
                try windows.append(window);
            } else {
                // For intermediate points, we need to consider boundaries from both adjacent segments
                // to properly handle corners in the extended polygon

                // Get boundary values from the left (segment i-1)
                const upper_left = self.getUpperBoundary(point.time, i - 1);
                const lower_left = self.getLowerBoundary(point.time, i - 1);

                // Get boundary values from the right (segment i)
                const upper_right = self.getUpperBoundary(point.time, i);
                const lower_right = self.getLowerBoundary(point.time, i);

                // At a corner, the window should span the full range
                // Upper bound: maximum of both sides (least restrictive)
                // Lower bound: minimum of both sides (least restrictive)
                const upper_value = @max(upper_left, upper_right);
                const lower_value = @min(lower_left, lower_right);

                const window = Window.init(shared.DiscretePoint{ .time = point.time, .value = upper_value }, shared.DiscretePoint{ .time = point.time, .value = lower_value });
                try windows.append(window);
            }
        }

        return windows;
    }

    // Get the initial window (vertical segment at first data point)
    pub fn getInitialWindow(self: *ExtendedPolygon) Window {
        const first_point = self.data_points.items[0];
        return Window.initVertical(
            first_point.time,
            first_point.value + self.error_bound,
            first_point.value - self.error_bound,
        );
    }

    // Get the final window (vertical segment at last data point)
    pub fn getFinalWindow(self: *ExtendedPolygon) Window {
        const last_point = self.data_points.items[self.data_points.items.len - 1];
        return Window.initVertical(
            last_point.time,
            last_point.value + self.error_bound,
            last_point.value - self.error_bound,
        );
    }
};

// ****************************************************************************************

// Define structures for different knot types
const KnotInfo = struct {
    time: usize,
    is_joint: bool,
    y1: f64, // For joint: the y-value; For disjoint: left segment end value.
    y2: f64, // For disjoint: right segment start value (unused for joint).
};
// ****************************************************************************************

// ****************************************************************************************

//*********************
// CORE DATA STRUCTURES
//*********************

// Represents a window in the extended polygon.A window is a segment with one endpoint on the upper
// boundary U and another endpoint on the lower boundary L.
pub const Window = struct {
    upper_point: shared.DiscretePoint,
    lower_point: shared.DiscretePoint,

    // Initialize window with two complete points.
    pub fn init(upper_point: shared.DiscretePoint, lower_point: shared.DiscretePoint) Window {
        return .{
            .upper_point = upper_point,
            .lower_point = lower_point,
        };
    }

    // Convenience function for creating vertical windows (special case).
    pub fn initVertical(time: usize, upper_value: f64, lower_value: f64) Window {
        return .{
            .upper_point = .{ .time = time, .value = upper_value },
            .lower_point = .{ .time = time, .value = lower_value },
        };
    }

    // Check if window is valid (upper point should be on upper boundary, lower on lower).
    pub fn isValid(self: Window) bool {
        // For vertical windows, upper value should be >= lower value.
        if (self.upper_point.time == self.lower_point.time) {
            return self.upper_point.value >= self.lower_point.value;
        }
        // For non-vertical windows, we'd need the polygon boundaries to validate properly.
        // For now, we just check that the window exists.
        return true;
    }
};

// Represents a segment of the extended polygon between two windows.
pub const ExtendedSegment = struct {
    start_window: Window,
    end_window: Window,
    upper_line: shared.LinearFunction,
    lower_line: shared.LinearFunction,

    pub fn init(w1: Window, w2: Window) ExtendedSegment {
        // Compute the extended boundaries.
        const upper_line = computeLine(w1.upper_point, w2.upper_point);
        const lower_line = computeLine(w1.lower_point, w2.lower_point);

        return .{
            .start_window = w1,
            .end_window = w2,
            .upper_line = upper_line,
            .lower_line = lower_line,
        };
    }
};

// Computes a line between two DiscretePoint instances.
pub fn computeLine(p1: shared.DiscretePoint, p2: shared.DiscretePoint) shared.LinearFunction {
    // Handle vertical line case
    if (p1.time == p2.time) {
        // For vertical lines, we can't represent them properly with y = mx + b
        // Return a line with very high slope as approximation
        return .{ .slope = if (p2.value > p1.value) 1e10 else -1e10, .intercept = p1.value - 1e10 * @as(f64, @floatFromInt(p1.time)) };
    }

    // Ensure temporal consistency: always compute line from earlier to later time.
    const earlier_point = if (p1.time <= p2.time) p1 else p2;
    const later_point = if (p1.time <= p2.time) p2 else p1;

    const time_diff = @as(f64, @floatFromInt(later_point.time - earlier_point.time));
    const slope = (later_point.value - earlier_point.value) / time_diff;
    const intercept = earlier_point.value - slope * @as(f64, @floatFromInt(earlier_point.time));

    return .{ .slope = slope, .intercept = intercept };
}

// Basic version of the PLA state.
pub const MixedPLAState = struct {
    allocator: std.mem.Allocator,
    error_bound: f32,
    windows: std.ArrayList(Window),
    current_polygon: ?ExtendedSegment,

    pub fn init(allocator: std.mem.Allocator, error_bound: f32) !MixedPLAState {
        return .{
            .allocator = allocator,
            .error_bound = error_bound,
            .windows = std.ArrayList(Window).init(allocator),
            .current_polygon = null,
        };
    }

    pub fn deinit(self: *MixedPLAState) void {
        self.windows.deinit();
    }

    // Process a new data point.
    pub fn addPoint(self: *MixedPLAState, point: shared.DiscretePoint) !void {
        // Create window around the point.
        const window = Window.init(
            point.time,
            point.value + self.error_bound,
            point.value - self.error_bound,
        );

        try self.windows.append(window);

        // Update the extended polygon.
        if (self.windows.items.len >= 2) {
            const prev_window = self.windows.items[self.windows.items.len - 2];
            self.current_polygon = ExtendedSegment.init(prev_window, window);
        }
    }
};

pub const VisibleRegion = struct {
    // The window from which we're computing visibility.
    source_window: Window,

    // Two separate convex hulls,
    upper_boundary_hull: ArrayList(shared.DiscretePoint), // Convex hull of upper boundary.
    lower_boundary_hull: ArrayList(shared.DiscretePoint), // Convex hull of lower boundary.

    // Supporting lines (z+ has max slope, z- has min slope).
    z_plus: ?shared.LinearFunction,
    z_minus: ?shared.LinearFunction,

    // Points where supporting lines touch the hulls.
    l_plus: ?shared.DiscretePoint, // Left tangent for upper separating line.
    r_plus: ?shared.DiscretePoint, // Right tangent for upper separating line.
    l_minus: ?shared.DiscretePoint, // Left tangent for lower separating line.
    r_minus: ?shared.DiscretePoint, // Right tangent for lower separating line.

    // The closing window (when found).
    closing_window: ?Window,

    // Is this visible region closed?.
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
        // The upper endpoint goes to upper hull, lower endpoint to lower hull
        try result.upper_boundary_hull.append(window.upper_point);
        try result.lower_boundary_hull.append(window.lower_point);

        // Initialize supporting lines if both endpoints are at the same x-coordinate
        if (window.upper_point.time == window.lower_point.time) {
            // For a vertical window, initialize with a vertical supporting line
            const vertical_line = shared.LinearFunction{
                .slope = std.math.inf(f64),
                .intercept = @as(f64, @floatFromInt(window.upper_point.time)),
            };
            result.z_plus = vertical_line;
            result.z_minus = vertical_line;
            result.l_plus = window.lower_point;
            result.r_plus = window.upper_point;
            result.l_minus = window.upper_point;
            result.r_minus = window.lower_point;
        }

        return result;
    }

    pub fn deinit(self: *VisibleRegion) void {
        self.upper_boundary_hull.deinit();
        self.lower_boundary_hull.deinit();
    }

    // Add debug prints to VisibleRegion.updateWithNewWindow
    pub fn updateWithNewWindow(self: *VisibleRegion, new_window: Window) !void {
        std.debug.print("  VR.updateWithNewWindow: new_window=({},{})->({},{})\n", .{ new_window.upper_point.time, new_window.upper_point.value, new_window.lower_point.time, new_window.lower_point.value });

        if (self.is_closed) {
            std.debug.print("  VR already closed, returning\n", .{});
            return;
        }

        // Get the time of the new window
        const window_time = @min(new_window.upper_point.time, new_window.lower_point.time);
        std.debug.print("  Window time: {}\n", .{window_time});

        // Only process windows that are to the right of our source
        const source_time = @max(self.source_window.upper_point.time, self.source_window.lower_point.time);
        std.debug.print("  Source time: {}\n", .{source_time});

        if (window_time <= source_time) {
            std.debug.print("  Window not to the right of source, skipping\n", .{});
            return;
        }

        // Check visibility constraints
        if (self.z_plus != null and self.z_minus != null) {
            const z_plus = self.z_plus.?;
            const z_minus = self.z_minus.?;

            std.debug.print("  z_plus: slope={}, intercept={}\n", .{ z_plus.slope, z_plus.intercept });
            std.debug.print("  z_minus: slope={}, intercept={}\n", .{ z_minus.slope, z_minus.intercept });

            const time_f64 = @as(f64, @floatFromInt(window_time));
            const z_plus_at_window = z_plus.slope * time_f64 + z_plus.intercept;
            const z_minus_at_window = z_minus.slope * time_f64 + z_minus.intercept;

            std.debug.print("  z_plus at window: {}\n", .{z_plus_at_window});
            std.debug.print("  z_minus at window: {}\n", .{z_minus_at_window});
            std.debug.print("  new_window lower: {}\n", .{new_window.lower_point.value});
            std.debug.print("  new_window upper: {}\n", .{new_window.upper_point.value});

            // Check if visibility constraints are violated
            if (new_window.lower_point.value > z_plus_at_window + 1e-10) {
                std.debug.print("  Lower boundary crossed above z_plus - closing VR\n", .{});
                self.is_closed = true;
                const closing_upper = shared.DiscretePoint{
                    .time = window_time,
                    .value = z_plus_at_window,
                };
                self.closing_window = Window{
                    .upper_point = closing_upper,
                    .lower_point = closing_upper,
                };
                return;
            }

            if (new_window.upper_point.value < z_minus_at_window - 1e-10) {
                std.debug.print("  Upper boundary crossed below z_minus - closing VR\n", .{});
                self.is_closed = true;
                const closing_lower = shared.DiscretePoint{
                    .time = window_time,
                    .value = z_minus_at_window,
                };
                self.closing_window = Window{
                    .upper_point = closing_lower,
                    .lower_point = closing_lower,
                };
                return;
            }
        }

        // Add points to hulls
        std.debug.print("  Adding points to hulls\n", .{});
        try self.addToUpperBoundary(new_window.upper_point);
        try self.addToLowerBoundary(new_window.lower_point);

        std.debug.print("  Upper hull size: {}\n", .{self.upper_boundary_hull.items.len});
        std.debug.print("  Lower hull size: {}\n", .{self.lower_boundary_hull.items.len});

        // Update supporting lines
        try self.updateSupportingLines();

        std.debug.print("  Updated supporting lines\n", .{});
    }

    // Reuse the addToHull logic.
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
    // Adapt the addToHull function for upper boundary.
    pub fn addToUpperBoundary(self: *VisibleRegion, point: shared.DiscretePoint) !void {
        // For upper boundary, we maintain LOWER convexity (turns right).
        try addToHullWithTurn(&self.upper_boundary_hull, .right, point);
    }

    // Adapt the addToHull function for lower boundary.
    pub fn addToLowerBoundary(self: *VisibleRegion, point: shared.DiscretePoint) !void {
        // For lower boundary, we maintain UPPER convexity (turns left).
        try addToHullWithTurn(&self.lower_boundary_hull, .left, point);
    }

    // Updates the supporting lines (z+ and z- in the paper).
    pub fn updateSupportingLines(self: *VisibleRegion) !void {
        // Need at least one point in each hull to compute supporting lines.
        if (self.upper_boundary_hull.items.len == 0 or self.lower_boundary_hull.items.len == 0) {
            return;
        }

        // Find all valid separating lines and choose the ones with max/min slope
        var max_slope: f64 = -std.math.inf(f64);
        var min_slope: f64 = std.math.inf(f64);
        var max_slope_line: ?shared.LinearFunction = null;
        var min_slope_line: ?shared.LinearFunction = null;
        var max_upper_idx: usize = 0;
        var max_lower_idx: usize = 0;
        var min_upper_idx: usize = 0;
        var min_lower_idx: usize = 0;

        // Try all combinations of points from upper and lower hulls
        for (self.upper_boundary_hull.items, 0..) |upper_point, u_idx| {
            for (self.lower_boundary_hull.items, 0..) |lower_point, l_idx| {
                // Skip if points are at the same x-coordinate (would create vertical line)
                if (upper_point.time == lower_point.time) continue;

                const line = computeLine(upper_point, lower_point);

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
            self.r_plus = self.upper_boundary_hull.items[max_upper_idx];
            self.l_plus = self.lower_boundary_hull.items[max_lower_idx];
        }

        // Update z- (min slope)
        if (min_slope_line) |line| {
            self.z_minus = line;
            self.l_minus = self.upper_boundary_hull.items[min_upper_idx];
            self.r_minus = self.lower_boundary_hull.items[min_lower_idx];
        }
    }

    // Check if a line validly separates the two hulls.
    fn isValidSeparatingLine(self: *VisibleRegion, line: shared.LinearFunction) bool {
        // A valid separating line must have:
        // 1. All upper hull points on or above the line
        // 2. All lower hull points on or below the line

        const epsilon = 1e-10;

        // Check all upper hull points are above or on the line
        for (self.upper_boundary_hull.items) |point| {
            const line_value = line.slope * @as(f64, @floatFromInt(point.time)) + line.intercept;
            if (point.value < line_value - epsilon) {
                return false;
            }
        }

        // Check all lower hull points are below or on the line
        for (self.lower_boundary_hull.items) |point| {
            const line_value = line.slope * @as(f64, @floatFromInt(point.time)) + line.intercept;
            if (point.value > line_value + epsilon) {
                return false;
            }
        }

        return true;
    }

    fn checkForClosure(self: *VisibleRegion, new_window: Window) !void {
        // If already closed, nothing to do.
        if (self.is_closed) {
            std.debug.print("Window already closed.", .{});
            return;
        }
        // Need supporting lines to check for closure.
        if (self.z_plus == null or self.z_minus == null) {
            std.debug.print("We don't have both supporting lines.", .{});
            return;
        }

        const z_plus = self.z_plus.?;
        const z_minus = self.z_minus.?;

        // Case 3 from the paper: Check if the new window crosses outside the visible wedge.
        // Check upper point against z_minus (Case 3 for upper chain).
        const upper_z_minus_value = z_minus.slope * @as(f64, @floatFromInt(
            new_window.upper_point.time,
        )) + z_minus.intercept;
        const upper_below_z_minus = new_window.upper_point.value < upper_z_minus_value - 1e-10;

        // Check lower point against z_plus (Case 3 for lower chain).
        const lower_z_plus_value = z_plus.slope * @as(f64, @floatFromInt(
            new_window.lower_point.time,
        )) + z_plus.intercept;
        const lower_above_z_plus = new_window.lower_point.value > lower_z_plus_value + 1e-10;

        // If either point crosses outside the wedge, close the visible region.
        if (upper_below_z_minus or lower_above_z_plus) {
            self.is_closed = true;

            std.debug.print("Either point crosses outside the wedge, close the visible region.", .{});

            if (upper_below_z_minus and self.r_minus != null) {
                // Upper boundary crossed z_minus.
                // Closing window connects r_minus (on lower hull) to intersection point.
                const intersection_value = z_minus.slope * @as(f64, @floatFromInt(
                    new_window.upper_point.time,
                )) + z_minus.intercept;
                const intersection_point = shared.DiscretePoint{
                    .time = new_window.upper_point.time,
                    .value = intersection_value,
                };
                self.closing_window = Window{
                    .upper_point = intersection_point,
                    .lower_point = self.r_minus.?,
                };
                std.debug.print("Closing window connects r_minus (on lower hull) to intersection point.\n .", .{});
            } else if (lower_above_z_plus and self.r_plus != null) {
                // Lower boundary crossed z_plus.
                // Closing window connects r_plus (on upper hull) to intersection point.
                const intersection_value = z_plus.slope * @as(f64, @floatFromInt(
                    new_window.lower_point.time,
                )) + z_plus.intercept;
                const intersection_point = shared.DiscretePoint{
                    .time = new_window.lower_point.time,
                    .value = intersection_value,
                };
                self.closing_window = Window{
                    .upper_point = self.r_plus.?,
                    .lower_point = intersection_point,
                };
                std.debug.print("Closing window connects r_plus (on upper hull) to intersection point.\n .", .{});
            }
        }
    }
};

// Structure to hold the dynamic programming state.
pub const MixedPLAOptimizer = struct {
    allocator: mem.Allocator,
    windows: ArrayList(Window),
    C: ArrayList(?Window), // C[k] array - nullable because C[1] is undefined.
    vr_cache: std.hash_map.HashMap(
        usize,
        VisibleRegion,
        std.hash_map.AutoContext(usize),
        std.hash_map.default_max_load_percentage,
    ),

    pub fn init(allocator: mem.Allocator) !MixedPLAOptimizer {
        return .{
            .allocator = allocator,
            .windows = ArrayList(Window).init(allocator),
            .C = ArrayList(?Window).init(allocator),
            .vr_cache = std.hash_map.HashMap(
                usize,
                VisibleRegion,
                std.hash_map.AutoContext(usize),
                std.hash_map.default_max_load_percentage,
            ).init(allocator),
        };
    }

    pub fn deinit(self: *MixedPLAOptimizer) void {
        self.windows.deinit();
        self.C.deinit();

        var it = self.vr_cache.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.vr_cache.deinit();
    }

    // Get the next window nw(w) - the window immediately to the right.
    pub fn nextWindow(self: *MixedPLAOptimizer, w: Window) ?Window {
        // Find the window with the smallest time > w's time.
        var next: ?Window = null;
        var min_time: usize = std.math.maxInt(usize);

        for (self.windows.items) |window| {
            const window_time = @min(window.upper_point.time, window.lower_point.time);
            const w_time = @max(w.upper_point.time, w.lower_point.time);

            if (window_time > w_time and window_time < min_time) {
                min_time = window_time;
                next = window;
            }
        }

        return next;
    }

    // Compute C[k] recursively.
    pub fn computeC(self: *MixedPLAOptimizer, k: usize) anyerror!Window {
        // Base cases
        if (k == 0) {
            const w0 = self.windows.items[0]; // Initial window.
            return try self.computeClosingWindow(w0);
        }

        if (k == 1) {
            return Error.InvalidK; // C[1] is undefined.
        }

        if (k == 2) {
            const c0 = try self.getC(0);
            return try self.computeClosingWindow(c0);
        }

        if (k == 3) {
            const c0 = try self.getC(0);
            const nw_c0 = self.nextWindow(c0) orelse return Error.NoNextWindow;
            return try self.computeClosingWindow(nw_c0);
        }

        // General case (k >= 4).
        if (k >= 5) {
            const c_k_minus_3 = try self.getC(k - 3);
            const nw_c_k_minus_3 = self.nextWindow(c_k_minus_3) orelse return Error.NoNextWindow;
            const cw_nw_c_k_minus_3 = try self.computeClosingWindow(nw_c_k_minus_3);

            const c_k_minus_2 = try self.getC(k - 2);
            const cw_c_k_minus_2 = try self.computeClosingWindow(c_k_minus_2);

            // Choose the rightmost window.
            if (isWindowCompletelyRightOf(cw_c_k_minus_2, cw_nw_c_k_minus_3)) {
                return cw_nw_c_k_minus_3;
            } else {
                return cw_c_k_minus_2;
            }
        } else { // k == 4.
            const c_k_minus_2 = try self.getC(k - 2);
            return try self.computeClosingWindow(c_k_minus_2);
        }
    }

    // Helper to get C[k] with memoization.
    pub fn getC(self: *MixedPLAOptimizer, k: usize) anyerror!Window {
        // Check if already computed.
        if (k < self.C.items.len) {
            if (self.C.items[k]) |window| {
                return window;
            }
        }

        // Ensure array is large enough.
        while (self.C.items.len <= k) {
            try self.C.append(null);
        }

        // Compute and store.
        const result = try self.computeC(k);
        self.C.items[k] = result;
        return result;
    }

    fn computeClosingWindow(self: *MixedPLAOptimizer, source_window: Window) anyerror!Window {
        var vr = try VisibleRegion.init(self.allocator, source_window);
        defer vr.deinit();

        // Find the starting index of windows to process.
        // We need to process windows that come after source_window.
        const source_time = @max(source_window.upper_point.time, source_window.lower_point.time);

        // Process all windows that come after the source window.
        for (self.windows.items) |window| {
            const window_time = @min(window.upper_point.time, window.lower_point.time);

            // Only process windows that are temporally after the source window.
            if (window_time > source_time) {
                try vr.updateWithNewWindow(window);

                // If the visible region closed, return the closing window.
                if (vr.is_closed) {
                    return vr.closing_window.?;
                }
            }
        }

        // If we reach here without closing, it means the visible region reaches the final window.
        return Error.VisibleRegionReachesFinalWindow;
    }
};

// Helper function to find the original window immediately after w in time.
fn nextWindow(windows: []const Window, w: Window) ?Window {
    var min_time: usize = std.math.maxInt(usize);
    const w_time = @max(w.upper_point.time, w.lower_point.time);
    var res: ?Window = null;
    for (windows) |win| {
        const win_time = @min(win.upper_point.time, win.lower_point.time);
        if (win_time > w_time and win_time < min_time) {
            min_time = win_time;
            res = win;
        }
    }
    return res;
}

// Check if w1 is strictly to the right of w2 (no temporal overlap).
fn isWindowRight(w1: Window, w2: Window) bool {
    const w1_left = std.min(w1.upper_point.time, w1.lower_point.time);
    //    const w1_right = std.max(w1.upper_point.time, w1.lower_point.time);
    //    const w2_left = std.min(w2.upper_point.time, w2.lower_point.time);
    const w2_right = std.max(w2.upper_point.time, w2.lower_point.time);
    if (w1_left > w2_right) return true;
    if (w1_left == w2_right) {
        // If they share a boundary point, we say w1 is not "to the right"
        return false;
    }
    return false;
}

// Helper function to compute slope between two points.
fn computeSlope(p1: shared.DiscretePoint, p2: shared.DiscretePoint) f64 {
    const dx = @as(f64, @floatFromInt(p2.time - p1.time));
    if (dx == 0) {
        // Handle vertical line case.
        return if (p2.value > p1.value) math.inf(f64) else -math.inf(f64);
    }
    return (p2.value - p1.value) / dx;
}

pub fn isWindowCompletelyRightOf(w1: Window, w2: Window) bool {
    // Get the temporal bounds.
    const w1_left_time = @min(w1.upper_point.time, w1.lower_point.time);
    const w1_right_time = @max(w1.upper_point.time, w1.lower_point.time);
    const w2_left_time = @min(w2.upper_point.time, w2.lower_point.time);
    const w2_right_time = @max(w2.upper_point.time, w2.lower_point.time);

    // Check for temporal overlap.
    if (w2_left_time < w1_right_time) {
        // w2 starts before w1 ends.
        return false;
    }

    if (w1_left_time == w1_right_time and w2_left_time == w2_right_time) {
        // Both windows are vertical (which they should be in extended polygons).

        if (w1_right_time < w2_left_time) {
            // Clear temporal separation.
            return true;
        } else if (w1_right_time == w2_left_time) {
            // Windows meet at a single time point
            // In this case, they could be considered ordered if they don't overlap in value.
            // But for identical windows or overlapping windows at the same time,
            // neither is "to the right" of the other.

            // Check if they're identical
            if (w1.upper_point.time == w2.upper_point.time and
                w1.lower_point.time == w2.lower_point.time and
                @abs(w1.upper_point.value - w2.upper_point.value) < 1e-10 and
                @abs(w1.lower_point.value - w2.lower_point.value) < 1e-10)
            {
                return false; // Identical windows.
            }

            // For non-identical windows at the same time, we could define an order based on values,
            // but typically windows at the same time are not comparable
            return false;
        }
    }

    return false;
}

// *****************************************************************************************
// DEBUGGING CODE
// *****************************************************************************************
fn printVisibleRegion(vr: *VisibleRegion) void {
    std.debug.print("VisibleRegion:\n", .{});
    std.debug.print(
        "  Source Window: ({d}, {d}) -> ({d}, {d})\n",
        .{
            vr.source_window.upper_point.time,
            vr.source_window.upper_point.value,
            vr.source_window.lower_point.time,
            vr.source_window.lower_point.value,
        },
    );

    std.debug.print("  Upper Hull Points:\n", .{});
    for (vr.upper_boundary_hull.items) |p| {
        std.debug.print("    ({d}, {d})\n", .{ p.time, p.value });
    }

    std.debug.print("  Lower Hull Points:\n", .{});
    for (vr.lower_boundary_hull.items) |p| {
        std.debug.print("    ({d}, {d})\n", .{ p.time, p.value });
    }

    if (vr.z_plus) |z| {
        std.debug.print("  z_plus: y = {d}x + {d}\n", .{ z.slope, z.intercept });
    } else {
        std.debug.print("  z_plus: null\n", .{});
    }

    if (vr.z_minus) |z| {
        std.debug.print("  z_minus: y = {d}x + {d}\n", .{ z.slope, z.intercept });
    } else {
        std.debug.print("  z_minus: null\n", .{});
    }

    if (vr.l_plus) |p| {
        std.debug.print("  l_plus: ({d}, {d})\n", .{ p.time, p.value });
    } else {
        std.debug.print("  l_plus: null\n", .{});
    }

    if (vr.r_plus) |p| {
        std.debug.print("  r_plus: ({d}, {d})\n", .{ p.time, p.value });
    } else {
        std.debug.print("  r_plus: null\n", .{});
    }

    if (vr.l_minus) |p| {
        std.debug.print("  l_minus: ({d}, {d})\n", .{ p.time, p.value });
    } else {
        std.debug.print("  l_minus: null\n", .{});
    }

    if (vr.r_minus) |p| {
        std.debug.print("  r_minus: ({d}, {d})\n", .{ p.time, p.value });
    } else {
        std.debug.print("  r_minus: null\n", .{});
    }

    std.debug.print("  is_closed: {}\n", .{vr.is_closed});
}
