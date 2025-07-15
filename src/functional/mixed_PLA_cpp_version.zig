const std = @import("std");
const math = std.math;
const ArrayList = std.ArrayList;
const mem = std.mem;
const Error = tersets.Error;
const tersets = @import("../tersets.zig");
const shared = @import("../utilities/shared_structs.zig");
const CV = @import("convex_polygon.zig");
const ConvexPolygon = CV.ConvexPolygon;
pub const RightMost = true;
pub const LeftMost = false;

// mixed_PLA_cpp_version.zig

// ======================================
// COMPRESSION AND DECOMPRESSION FUNCTIONS
// =======================================
// Additional error types for Mixed-PLA
const MixedPLAError = error{
    ParallelLines,
} || Error;

// Compress using Mixed-PLA algorithm.
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

    // Initialize the Mixed-PLA algorithm.
    var mixed_pla = try MixedContApr.init(allocator, adjusted_error);
    defer mixed_pla.deinit();

    // Process all data points sequentially (simulating streaming).
    for (uncompressed_values, 0..) |value, i| {
        const point = shared.DiscretePoint{ .time = i, .value = value };
        _ = try mixed_pla.update(point);
    }

    // Finalize the fitting to get the complete solution.
    try mixed_pla.closeFitting();

    // Convert the internal representation to the output format.
    // The mixed_pla.segments contains all segment endpoints.
    // The mixed_pla.knot_flags indicates whether knots are joint (true) or disjoint (false).

    // Count the number of knots (excluding the first and last points).
    const num_knots = mixed_pla.knot_flags.items.len;

    // Special case: if there's only one segment (no internal knots).
    if (num_knots == 0 and mixed_pla.segments.items.len >= 2) {
        // Direct line from first to last point.
        const first = mixed_pla.segments.items[0];
        const last = mixed_pla.segments.items[mixed_pla.segments.items.len - 1];

        // Compute slope and intercept.
        const dt = @as(f64, @floatFromInt(last.time - first.time));
        const slope = if (dt != 0) (last.value - first.value) / dt else 0.0;
        const intercept = first.value - slope * @as(f64, @floatFromInt(first.time));

        // Special encoding for single segment: [1][slope][intercept][original_length].
        try compressed_values.append(1);
        try compressed_values.appendSlice(std.mem.asBytes(&slope));
        try compressed_values.appendSlice(std.mem.asBytes(&intercept));
        const orig_len = uncompressed_values.len;
        try compressed_values.appendSlice(std.mem.asBytes(&orig_len));
        return;
    }

    // Write number of knots as first byte.
    try compressed_values.appendSlice(std.mem.asBytes(&num_knots));

    // Process segments and knots.
    var segment_idx: usize = 0;

    // Skip the first point (it's implied as the start).
    if (mixed_pla.segments.items.len > 0) {
        segment_idx = 1;
    }

    // Write each knot.
    for (mixed_pla.knot_flags.items, 0..) |is_joint, knot_idx| {
        _ = knot_idx;

        if (segment_idx >= mixed_pla.segments.items.len) break;

        if (is_joint) {
            // Joint knot: the segment endpoint is shared.
            const point = mixed_pla.segments.items[segment_idx];
            const x = @as(i64, @intCast(point.time));
            const y = point.value;

            // Store as positive x for joint knot.
            try compressed_values.appendSlice(std.mem.asBytes(&x));
            try compressed_values.appendSlice(std.mem.asBytes(&y));

            segment_idx += 1;
        } else {
            // Disjoint knot: two separate points (end of current, start of next).
            const end_point = mixed_pla.segments.items[segment_idx];
            segment_idx += 1;

            const start_point = if (segment_idx < mixed_pla.segments.items.len)
                mixed_pla.segments.items[segment_idx]
            else
                end_point; // Fallback for safety.

            // Store as negative x for disjoint knot.
            const x = -@as(i64, @intCast(end_point.time));
            const y1 = end_point.value;
            const y2 = start_point.value;

            try compressed_values.appendSlice(std.mem.asBytes(&x));
            try compressed_values.appendSlice(std.mem.asBytes(&y1));
            try compressed_values.appendSlice(std.mem.asBytes(&y2));

            segment_idx += 1;
        }
    }
}

pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
    allocator: mem.Allocator,
) Error!void {
    if (compressed_values.len < @sizeOf(usize)) {
        return Error.InvalidData;
    }

    var offset: usize = 0;
    const size_f64 = @sizeOf(f64);
    const size_i64 = @sizeOf(i64);
    const size_usize = @sizeOf(usize);

    // Read number of knots as usize.
    const num_knots = mem.bytesAsSlice(usize, compressed_values[offset .. offset + size_usize])[0];
    offset += size_usize;

    // Handle special case: direct line (slope, intercept, original_length).
    if (num_knots == 1 and compressed_values.len == size_usize + size_f64 + size_f64 + size_usize) {
        // This is the special case: [1][slope][intercept][original_length].
        if (offset + 2 * size_f64 + size_usize > compressed_values.len) {
            return Error.InvalidData;
        }

        const slope = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
        offset += size_f64;

        const intercept = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
        offset += size_f64;

        // Read the stored original length.
        const orig_len = mem.bytesAsSlice(usize, compressed_values[offset .. offset + size_usize])[0];

        // Use the correct range [0..orig_len].
        for (0..orig_len) |t| {
            const value = slope * @as(f64, @floatFromInt(t)) + intercept;
            try decompressed_values.append(value);
        }
        return;
    }

    // Handle general case: multiple knots.
    var knots = ArrayList(KnotInfo).init(allocator);
    defer knots.deinit();

    // Read all knots.
    for (0..num_knots) |_| {
        // Read the x coordinate as i64 to check sign.
        if (offset + size_i64 > compressed_values.len) {
            return Error.InvalidData;
        }

        const x_i64 = mem.bytesAsSlice(i64, compressed_values[offset .. offset + size_i64])[0];
        offset += size_i64;

        if (x_i64 < 0) {
            // Disjoint knot: [-x][y1][y2].
            if (offset + 2 * size_f64 > compressed_values.len) {
                return Error.InvalidData;
            }

            const y1 = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;

            const y2 = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;

            try knots.append(KnotInfo{
                .time = @as(usize, @intCast(-x_i64)),
                .is_joint = false,
                .y1 = y1,
                .y2 = y2,
            });
        } else {
            // Joint knot: [x][y].
            if (offset + size_f64 > compressed_values.len) {
                return Error.InvalidData;
            }

            const y = mem.bytesAsSlice(f64, compressed_values[offset .. offset + size_f64])[0];
            offset += size_f64;

            try knots.append(KnotInfo{
                .time = @as(usize, @intCast(x_i64)),
                .is_joint = true,
                .y1 = y,
                .y2 = y, // Same value for joint knots.
            });
        }
    }

    if (knots.items.len == 0) {
        return; // No knots, nothing to decompress.
    }

    // Sort knots by time (should already be sorted, but ensure correctness).
    mem.sort(KnotInfo, knots.items, {}, struct {
        pub fn lessThan(context: void, a: KnotInfo, b: KnotInfo) bool {
            _ = context;
            return a.time < b.time;
        }
    }.lessThan);

    // Determine the time range.
    const first_time: usize = 0; // Always start from time 0.
    const last_time = knots.items[knots.items.len - 1].time;

    // Build the complete piecewise linear function.
    // We need to handle the implicit first point at time 0.
    var segments = ArrayList(SegmentInfo).init(allocator);
    defer segments.deinit();

    // Create segments from knots.
    var prev_time: usize = 0;
    var prev_value: f64 = 0.0; // Will be computed from first segment.

    // First segment: from start to first knot.
    if (knots.items.len > 0) {
        const first_knot = knots.items[0];

        // For the first segment, we need to extrapolate backwards to find the value at time 0.
        if (knots.items.len > 1) {
            const second_knot = knots.items[1];
            const dt = @as(f64, @floatFromInt(second_knot.time - first_knot.time));
            if (dt > 0) {
                const slope = (second_knot.y1 - first_knot.y1) / dt;
                prev_value = first_knot.y1 - slope * @as(f64, @floatFromInt(first_knot.time));
            } else {
                prev_value = first_knot.y1;
            }
        } else {
            // Only one knot - assume horizontal line before it.
            prev_value = first_knot.y1;
        }
    }

    // Process each knot to create segments.
    for (knots.items) |knot| {
        if (knot.is_joint) {
            // Joint knot: create segment from prev to this knot.
            try segments.append(SegmentInfo{
                .start_time = prev_time,
                .start_value = prev_value,
                .end_time = knot.time,
                .end_value = knot.y1,
            });
            prev_time = knot.time;
            prev_value = knot.y1;
        } else {
            // Disjoint knot: create segment ending at this knot.
            try segments.append(SegmentInfo{
                .start_time = prev_time,
                .start_value = prev_value,
                .end_time = knot.time,
                .end_value = knot.y1,
            });

            // Next segment starts from y2.
            prev_time = knot.time;
            prev_value = knot.y2;
        }
    }

    // Add final segment if needed (from last knot to end of data).
    if (prev_time < last_time) {
        // Extrapolate using the slope of the last segment.
        if (segments.items.len > 0) {
            const last_seg = segments.items[segments.items.len - 1];
            const dt = @as(f64, @floatFromInt(last_seg.end_time - last_seg.start_time));
            if (dt > 0) {
                const slope = (last_seg.end_value - last_seg.start_value) / dt;
                const end_value = prev_value + slope * @as(f64, @floatFromInt(last_time - prev_time));
                try segments.append(SegmentInfo{
                    .start_time = prev_time,
                    .start_value = prev_value,
                    .end_time = last_time,
                    .end_value = end_value,
                });
            }
        }
    }

    // Reconstruct values by evaluating the piecewise linear function.
    var current_segment_idx: usize = 0;

    for (first_time..last_time + 1) |t| {
        // Find the appropriate segment.
        while (current_segment_idx < segments.items.len and
            t > segments.items[current_segment_idx].end_time)
        {
            current_segment_idx += 1;
        }

        if (current_segment_idx >= segments.items.len) {
            // Beyond last segment - extrapolate.
            const last_seg = segments.items[segments.items.len - 1];
            const dt = @as(f64, @floatFromInt(last_seg.end_time - last_seg.start_time));
            if (dt > 0) {
                const slope = (last_seg.end_value - last_seg.start_value) / dt;
                const value = last_seg.end_value + slope * @as(f64, @floatFromInt(t - last_seg.end_time));
                try decompressed_values.append(value);
            } else {
                try decompressed_values.append(last_seg.end_value);
            }
        } else {
            const seg = segments.items[current_segment_idx];
            const dt = @as(f64, @floatFromInt(seg.end_time - seg.start_time));

            if (dt == 0) {
                try decompressed_values.append(seg.end_value);
            } else {
                const slope = (seg.end_value - seg.start_value) / dt;
                const value = seg.start_value + slope * @as(f64, @floatFromInt(t - seg.start_time));
                try decompressed_values.append(value);
            }
        }
    }
}

// Helper structure for segment information.
const SegmentInfo = struct {
    start_time: usize,
    start_value: f64,
    end_time: usize,
    end_value: f64,
};

// ================================
// CORE DATA STRUCTURES
// ================================

pub const KnotInfo = struct {
    time: usize,
    is_joint: bool,
    y1: f64,
    y2: f64,
};

pub const ClosureDirection = enum { UpperChain, LowerChain };

// Represents a point with upper and lower bounds (±epsilon)
pub const DataSegment = struct {
    upper: shared.DiscretePoint, // (time, value + epsilon)
    lower: shared.DiscretePoint, // (time, value - epsilon)

    pub fn init(time: usize, value: f64, epsilon: f64) DataSegment {
        return .{
            .upper = shared.DiscretePoint{ .time = time, .value = value + epsilon },
            .lower = shared.DiscretePoint{ .time = time, .value = value - epsilon },
        };
    }

    pub fn getValue(self: *const DataSegment) f64 {
        return (self.upper.value + self.lower.value) / 2.0;
    }
};

// Point in parameter space (slope, intercept)
pub const ParameterPoint = struct {
    slope: f64,
    intercept: f64,

    // Convert to/from LinearFunction for clarity
    pub fn fromLinearFunction(lf: shared.LinearFunction) ParameterPoint {
        return .{ .slope = lf.slope, .intercept = lf.intercept };
    }

    pub fn toLinearFunction(self: ParameterPoint) shared.LinearFunction {
        return .{ .slope = self.slope, .intercept = self.intercept };
    }
};

// Half-plane in parameter space.
pub const HalfPlane = struct {
    sep_line: shared.LinearFunction, // Separating line in parameter space.
    direction: enum { PointToAbove, PointToBelow },

    pub fn fromDiscretePoint(time: f64, value: f64, is_upper: bool) HalfPlane {
        return .{
            .sep_line = .{
                .slope = time,
                .intercept = value,
            },
            .direction = if (is_upper) .PointToBelow else .PointToAbove,
        };
    }

    // Check if a parameter point satisfies the constraint
    pub fn contains(self: HalfPlane, pt: ParameterPoint) bool {
        // Evaluate the separating line at the point's slope
        const line_value = self.sep_line.slope * pt.slope + self.sep_line.intercept;

        return switch (self.direction) {
            .PointToAbove => pt.intercept >= line_value,
            .PointToBelow => pt.intercept <= line_value,
        };
    }

    // Result of containment check
    pub fn checkContainment(self: HalfPlane, pt: ParameterPoint) enum { Inside, Outside, OnBoundary } {
        const line_value = self.sep_line.slope * pt.slope + self.sep_line.intercept;
        const diff = pt.intercept - line_value;
        const epsilon = 1e-10;

        if (@abs(diff) < epsilon) return .OnBoundary;

        return switch (self.direction) {
            .PointToAbove => if (diff > 0) .Inside else .Outside,
            .PointToBelow => if (diff < 0) .Inside else .Outside,
        };
    }
};

// Edge in the convex hull
const Edge = struct {
    point: ParameterPoint,
    color: enum { Red, Green }, // Red = vertex, Green = intersection point
};

// Convex chain (upper or lower boundary)
pub const ConvexList = struct {
    edges: ArrayList(Edge),
    end_most: ?ParameterPoint,
    is_upper: bool, // true for upper chain, false for lower
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, is_upper: bool) !ConvexList {
        return .{
            .edges = ArrayList(Edge).init(allocator),
            .end_most = null,
            .is_upper = is_upper,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ConvexList) void {
        self.edges.deinit();
    }

    // Extend the chain to satisfy the half-plane constraint
    pub fn pad(self: *ConvexList, hp: *const HalfPlane) !IntersectionResult {
        if (self.end_most == null) return .NonExistChain;

        const end_containment = hp.checkContainment(self.end_most.?);
        if (end_containment != .Outside) {
            return .ContainAllChain;
        }

        // Scan backwards to find where we enter the half-plane
        var i = self.edges.items.len;
        while (i > 0) : (i -= 1) {
            const edge = self.edges.items[i - 1];
            const containment = hp.checkContainment(edge.point);

            if (containment == .Inside) {
                // Compute intersection and add new edge
                const intersection = computeIntersection(edge.point, self.end_most.?, hp);
                try self.edges.append(.{ .point = intersection, .color = .Green });
                return .ContainSomeChain;
            } else if (containment == .OnBoundary) {
                // Point is exactly on the boundary
                self.edges.items[i - 1].color = .Green;
                return .ContainSomeChain;
            }
        }

        return .ContainNoneChain;
    }

    // Remove parts that violate the half-plane constraint
    pub fn cut(self: *ConvexList, hp: *const HalfPlane) ?ParameterPoint {
        var i: usize = 0;
        var last_inside: ?ParameterPoint = null;

        while (i < self.edges.items.len) {
            const edge = self.edges.items[i];
            const containment = hp.checkContainment(edge.point);

            if (containment == .Outside) {
                if (last_inside) |inside_pt| {
                    // Compute intersection
                    const intersection = computeIntersection(inside_pt, edge.point, hp);
                    // Remove everything after this point
                    self.edges.shrinkRetainingCapacity(i);
                    return intersection;
                }
                // Remove this edge
                _ = self.edges.orderedRemove(i);
            } else {
                last_inside = edge.point;
                i += 1;
            }
        }

        // Check end_most
        if (self.end_most) |em| {
            if (hp.checkContainment(em) == .Outside) {
                if (last_inside) |inside_pt| {
                    const intersection = computeIntersection(inside_pt, em, hp);
                    return intersection;
                }
            }
        }

        return null;
    }
    // Clear the convex list
    pub fn clear(self: *ConvexList) void {
        self.edges.clearRetainingCapacity();
        self.end_most = null;
    }

    // Get the size of the convex list
    pub fn getSize(self: *const ConvexList) usize {
        return self.edges.items.len + if (self.end_most != null) @as(usize, 1) else 0;
    }

    // Initialize convex chain with points (matches C++ ConvexList::init)
    pub fn initConvexChain(self: *ConvexList, first_point: ParameterPoint, second_point: ?ParameterPoint, end_point: ParameterPoint) !void {
        // Clear existing edges
        self.edges.clearRetainingCapacity();

        // Add first point as an edge (always required)
        try self.edges.append(.{ .point = first_point, .color = .Red });

        // Add second point as an edge (if provided and different from first)
        if (second_point) |p2| {
            const epsilon = 1e-10;
            const is_different = (@abs(p2.slope - first_point.slope) > epsilon) or
                (@abs(p2.intercept - first_point.intercept) > epsilon);
            if (is_different) {
                try self.edges.append(.{ .point = p2, .color = .Red });
            }
        }

        // Set the endpoint of the chain
        self.end_most = end_point;
    }
};

// Result of intersection operation
pub const IntersectionResult = enum {
    ContainAllChain, // Polygon completely inside half-plane
    ContainSomeChain, // Polygon partially intersects
    ContainNoneChain, // Polygon completely outside
    NonExistChain, // Polygon is empty
};

// Helper function to compute intersection of line segment with half-plane
fn computeIntersection(inside: ParameterPoint, outside: ParameterPoint, hp: *const HalfPlane) ParameterPoint {
    // Line segment: point(t) = inside + t * (outside - inside)
    // Separating line: intercept = sep_line.slope * slope + sep_line.intercept

    // At intersection: inside.intercept + t * (outside.intercept - inside.intercept) =
    //                  sep_line.slope * (inside.slope + t * (outside.slope - inside.slope)) + sep_line.intercept

    const delta_intercept = outside.intercept - inside.intercept;
    const delta_slope = outside.slope - inside.slope;

    // Rearranging: t * (delta_intercept - sep_line.slope * delta_slope) =
    //              sep_line.slope * inside.slope + sep_line.intercept - inside.intercept

    const denominator = delta_intercept - hp.sep_line.slope * delta_slope;
    const numerator = hp.sep_line.slope * inside.slope + hp.sep_line.intercept - inside.intercept;

    // Handle parallel lines case
    const t = if (@abs(denominator) < 1e-10)
        0.5 // Fallback to midpoint if lines are parallel
    else
        numerator / denominator;

    // Clamp t to [0, 1] to ensure we stay within the line segment
    const t_clamped = @max(0.0, @min(1.0, t));

    return .{
        .slope = inside.slope + t_clamped * (outside.slope - inside.slope),
        .intercept = inside.intercept + t_clamped * (outside.intercept - inside.intercept),
    };
}

// Fitting window - represents valid time range for segment endpoints
const FittingWindow = struct {
    tu: f64, // Upper time bound
    tg: f64, // Lower time bound

    pub fn init() FittingWindow {
        return .{ .tu = -1, .tg = -1 };
    }
    pub fn assign(self: *FittingWindow, tu: f64, tg: f64) void {
        self.tu = tu;
        self.tg = tg;
    }
};
// ================================
// MAIN ALGORITHM STRUCTURES
// ================================

// Arc structure - tracks extreme supporting lines
const Arc = struct {
    points: ArrayList(shared.DiscretePoint), // Arc points
    extreme_param_point: ?ParameterPoint, // Extreme point in parameter space
    current_point: ?shared.DiscretePoint,
    extreme_light: ?shared.LinearFunction, // The supporting line
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Arc {
        return .{
            .points = ArrayList(shared.DiscretePoint).init(allocator),
            .extreme_param_point = null,
            .current_point = null,
            .extreme_light = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Arc) void {
        self.points.deinit();
    }

    pub fn update(self: *Arc, point: shared.DiscretePoint) void {
        self.current_point = point;
        // Add to points list
        self.points.append(point) catch {}; // TODO: Handle error properly
    }

    pub fn reset(self: *Arc, extreme_param_pt: ParameterPoint, current_pt: shared.DiscretePoint, time_base: usize) void {
        self.extreme_param_point = extreme_param_pt;
        self.current_point = current_pt;
        self.points.clearRetainingCapacity();
        self.points.append(current_pt) catch {}; // TODO: Handle error properly

        // Convert parameter point to linear function for extreme light
        self.extreme_light = transformParametricPoint(extreme_param_pt, time_base);
    }

    pub fn clear(self: *Arc) void {
        self.extreme_param_point = null;
        self.current_point = null;
        self.extreme_light = null;
        self.points.clearRetainingCapacity();
    }

    pub fn returnExLight(self: *Arc) ?shared.LinearFunction {
        return self.extreme_light;
    }

    // Get the front (first) point of the arc
    pub fn front(self: *const Arc) ?shared.DiscretePoint {
        if (self.points.items.len > 0) {
            return self.points.items[0];
        }
        return null;
    }

    // Pop the back (last) point of the arc
    pub fn popBack(self: *Arc, remove: bool) ?shared.DiscretePoint {
        if (self.points.items.len > 0) {
            const last_point = self.points.items[self.points.items.len - 1];
            if (remove) {
                _ = self.points.pop();
            }
            return last_point;
        }
        return null;
    }
};

// Evaluate a line at a given time
fn evaluateLineAt(line: shared.LinearFunction, time: f64) f64 {
    return line.slope * time + line.intercept;
}

// Transform a parametric point to a line with time base shift
fn transformParametricPoint(param_point: ParameterPoint, time_base: usize) shared.LinearFunction {
    // Based on C++ line::transform() method
    // The parametric point represents (slope, intercept) in parameter space
    // We need to shift the intercept by the time base
    const time_base_f64 = @as(f64, @floatFromInt(time_base));
    return .{
        .slope = param_point.slope,
        .intercept = param_point.intercept - param_point.slope * time_base_f64,
    };
}

// Single fitting instance (manages one potential path)
pub const Fittable = struct {
    // Boundaries and segments
    real_boundary: ArrayList(*const DataSegment),
    light_window: DataSegment,
    time_base: usize,
    current_time: usize,

    // Convex hull in parameter space
    conv: ConvexPolygon,

    // Two arcs for tracking supporting lines
    ceil_arc: Arc, // Upper arc
    flor_arc: Arc, // Lower arc (floor arc)

    // Fitting window and knot information
    fw: FittingWindow,
    knot_type: bool, // true = connected, false = disjoint
    beg_point: shared.DiscretePoint,
    end_point: shared.DiscretePoint,

    // For continuous segments (bias information)
    bias_lseg: ?*const DataSegment,
    bias_chain: ?*Arc,
    closed_direction: ClosureDirection,

    allocator: std.mem.Allocator,
    epsilon: f64,

    pub fn init(allocator: std.mem.Allocator, epsilon: f64, knot_type: bool) !Fittable {
        return .{
            .real_boundary = ArrayList(*const DataSegment).init(allocator),
            .light_window = DataSegment.init(0, 0.0, epsilon),
            .time_base = 0,
            .current_time = 0,
            // Here only one argument is expected. Is the epsilon needed?
            .conv = try ConvexPolygon.init(allocator),
            .ceil_arc = Arc.init(allocator),
            .flor_arc = Arc.init(allocator),
            .fw = FittingWindow.init(),
            .knot_type = knot_type,
            .beg_point = shared.DiscretePoint{ .time = 0, .value = 0.0 },
            .end_point = shared.DiscretePoint{ .time = 0, .value = 0.0 },
            .bias_lseg = null,
            .bias_chain = null,
            .closed_direction = .UpperChain,
            .allocator = allocator,
            .epsilon = epsilon,
        };
    }
    pub fn deinit(self: *Fittable) void {
        // Free all DataSegments before deinitializing the ArrayList
        for (self.real_boundary.items) |data_segment| {
            self.allocator.destroy(data_segment);
        }
        self.real_boundary.deinit();

        // Free bias_lseg if it exists
        if (self.bias_lseg) |lseg| {
            self.allocator.destroy(lseg);
        }

        self.conv.deinit();
        self.ceil_arc.deinit();
        self.flor_arc.deinit();
    }

    // Main update logic - returns true if can continue, false if closed
    pub fn update(self: *Fittable, dp: shared.DiscretePoint) !bool {
        // Convert discrete point to DataSegment

        // Buffer the point
        try self.buffer(dp);

        // If we have <= 3 points, continue buffering
        if (self.real_boundary.items.len <= 3) {
            return true;
        }

        // When we have 4 points, call updateImmediately
        const proceed = try self.updateImmediately();
        return proceed;
    }

    // Buffer a data point - creates DataSegment internally when needed
    fn buffer(self: *Fittable, dp: shared.DiscretePoint) !void {
        // Create data segment from point
        const data_segment = try self.allocator.create(DataSegment);
        data_segment.* = DataSegment.init(dp.time, dp.value, self.epsilon);

        try self.real_boundary.append(data_segment);
        self.current_time = dp.time;

        // Initialize convex polygon when we have 2 points
        if (self.real_boundary.items.len == 2) {
            const first = self.real_boundary.items[0];
            const second = self.real_boundary.items[1];
            self.time_base = first.upper.time;

            // Initialize light window from first segment
            self.light_window = first.*;

            // Initialize convex polygon with the two center points and epsilon
            const p1 = shared.DiscretePoint{ .time = first.upper.time, .value = (first.upper.value + first.lower.value) / 2.0 };
            const p2 = shared.DiscretePoint{ .time = second.upper.time, .value = (second.upper.value + second.lower.value) / 2.0 };

            try self.conv.initWithPoints(p1, p2, self.epsilon);
        }
    }
    // Core update logic when we have enough points
    fn updateImmediately(self: *Fittable) !bool {
        // Step 0: Delete oldest segment A
        const A = self.real_boundary.orderedRemove(0);
        self.allocator.destroy(A); // Free the allocated DataSegment

        // Step 1: Get segments B and C
        const B = self.real_boundary.items[0];
        const C = self.real_boundary.items[1];
        _ = B; // B is used for reference in full implementation

        // Step 1.1: Update with upper point constraint
        const upper_halfplane = HalfPlane.fromDiscretePoint(@as(f64, @floatFromInt(
            self.time_base,
        )) - @as(f64, @floatFromInt(
            C.upper.time,
        )), C.upper.value, true);
        const rst_uh = try self.conv.intersect(&upper_halfplane);

        if (rst_uh == .ContainSomeChain) {
            // Reset upper arc since light changed
            if (self.conv.returnEndmost(RightMost)) |right_pt| {
                self.ceil_arc.reset(right_pt, C.upper, self.time_base);
            }
        }

        // Step 1.2: Update with lower point constraint
        const lower_halfplane = HalfPlane.fromDiscretePoint(@as(f64, @floatFromInt(
            self.time_base,
        )) - @as(f64, @floatFromInt(C.lower.time)), C.lower.value, false // is_lower
        );

        const rst_lh = try self.conv.intersect(&lower_halfplane);

        if (rst_lh == .ContainSomeChain) {
            // Reset lower arc since light changed
            if (self.conv.returnEndmost(LeftMost)) |left_pt| {
                self.flor_arc.reset(left_pt, C.lower, self.time_base);
            }
        }

        // Step 2: Check if we need to restart or can continue
        if (rst_uh == .ContainNoneChain or rst_lh == .ContainNoneChain) {
            // Restart new round - the fitting is closed

            // Step 2.1: Determine closure direction
            if (rst_uh == .ContainNoneChain) {
                self.closed_direction = .UpperChain;
                self.ceil_arc.clear();
                self.bias_chain = &self.flor_arc;
            } else {
                self.closed_direction = .LowerChain;
                self.flor_arc.clear();
                self.bias_chain = &self.ceil_arc;
            }

            // Step 2.2: Compute end points using extreme light
            if (self.bias_chain.?.returnExLight()) |exl| {
                self.computeEndPoint(exl, @as(f64, @floatFromInt(C.upper.time)));
            }

            // Step 2.3: Compute new light window and bias segment
            // Clean up any existing bias_lseg before assigning new one
            if (self.bias_lseg) |old_lseg| {
                self.allocator.destroy(old_lseg);
            }
            self.bias_lseg = try self.computeWindowLseg(C, self.closed_direction);

            // Step 2.4: Prepare fitting window information
            if (self.closed_direction == .UpperChain) {
                self.fw.tu = @as(f64, @floatFromInt(C.upper.time));
                self.fw.tg = @as(f64, @floatFromInt(self.light_window.lower.time));
            } else {
                self.fw.tu = @as(f64, @floatFromInt(self.light_window.upper.time));
                self.fw.tg = @as(f64, @floatFromInt(C.lower.time));
            }

            return false; // Closed
        } else {
            // Step 3: Update arcs if needed
            if (rst_uh == .ContainAllChain) {
                self.ceil_arc.update(C.upper);
            }
            if (rst_lh == .ContainAllChain) {
                self.flor_arc.update(C.lower);
            }

            return true; // Continue
        }
    }

    // Compute end point using extreme light
    fn computeEndPoint(self: *Fittable, exl: shared.LinearFunction, ctime: f64) void {
        // Compute beginning point by intersecting light window with supporting line
        self.beg_point = DataSegmentExt.intersectWithLine(self.light_window, exl);

        // Compute end point
        self.end_point.time = @as(usize, @intFromFloat(ctime));
        self.end_point.value = exl.slope * ctime + exl.intercept;
    }

    // Compute new light window and left segment for continuous fitting
    // This function is called when a fitting round closes and we need to prepare
    // for continuous (connected) fitting in the next round.
    //
    // upOrlow/direction indicates which chain caused the closure:
    // - UpperChain: upper chain crossed lowest extreme light (smallest slope)
    // - LowerChain: lower chain crossed highest extreme light (largest slope)
    //
    // Returns a left segment that will be used for continuous fitting
    fn computeWindowLseg(self: *Fittable, C: *const DataSegment, direction: ClosureDirection) !*const DataSegment {
        // Create the left segment that will be returned
        const lseg = try self.allocator.create(DataSegment);

        // Get extreme light based on closure direction
        var extreme_light: shared.LinearFunction = undefined;
        var flag = true;

        if (direction == .UpperChain) {
            // For upper chain: below the lowest light (smallest slope)
            // Get extreme light from leftmost point of convex polygon
            if (self.conv.returnEndmost(LeftMost)) |leftmost_point| {
                extreme_light = transformParametricPoint(leftmost_point, self.time_base);
            } else {
                flag = false;
            }

            if (flag) {
                // 1. Compute light window
                // Upper point: at C's time, value from extreme light
                self.light_window.upper.time = C.upper.time;
                self.light_window.upper.value = evaluateLineAt(extreme_light, @as(f64, @floatFromInt(C.upper.time)));

                // Lower point: at front of floor arc time, value from extreme light
                if (self.flor_arc.front()) |front_point| {
                    const fT = @as(f64, @floatFromInt(front_point.time));
                    self.light_window.lower.time = front_point.time;
                    self.light_window.lower.value = evaluateLineAt(extreme_light, fT);
                } else {
                    flag = false;
                }

                // 2. Compute the left segment
                if (flag) {
                    lseg.upper = self.light_window.upper;
                    if (self.flor_arc.popBack(false)) |back_point| {
                        lseg.lower = back_point;
                    } else {
                        flag = false;
                    }
                }
            }
        } else { // LowerChain
            // For lower chain: above the highest light (largest slope)
            // Get extreme light from rightmost point of convex polygon
            if (self.conv.returnEndmost(RightMost)) |rightmost_point| {
                extreme_light = transformParametricPoint(rightmost_point, self.time_base);
            } else {
                flag = false;
            }

            if (flag) {
                // 1. Compute light window
                // Lower point: at C's time, value from extreme light
                self.light_window.lower.time = C.lower.time;
                self.light_window.lower.value = evaluateLineAt(extreme_light, @as(f64, @floatFromInt(C.lower.time)));

                // Upper point: at front of ceil arc time, value from extreme light
                if (self.ceil_arc.front()) |front_point| {
                    const fT = @as(f64, @floatFromInt(front_point.time));
                    self.light_window.upper.time = front_point.time;
                    self.light_window.upper.value = evaluateLineAt(extreme_light, fT);
                } else {
                    flag = false;
                }

                // 2. Compute the left segment
                if (flag) {
                    lseg.lower = self.light_window.lower;
                    if (self.ceil_arc.popBack(false)) |back_point| {
                        lseg.upper = back_point;
                    } else {
                        flag = false;
                    }
                }
            }
        }

        if (flag) {
            return lseg;
        } else {
            // Two parallel lines case - cleanup and return error
            self.allocator.destroy(lseg);
            std.debug.print("Two parallel lines: compute light windows\n", .{});
            return MixedPLAError.ParallelLines;
        }
    }

    // Start a new fitting round after closure
    pub fn initNewRoundWithType(self: *Fittable, link_type: enum { Connected, Disjoint }) !void {
        // Reset convex polygon
        self.conv.setEmpty();

        switch (link_type) {
            .Connected => {
                self.knot_type = true;
                try self.restartContNewRound();
            },
            .Disjoint => {
                self.knot_type = false;
                try self.restartUncontNewRound();
            },
        }
    }

    // Fix 3: Update restartUncontNewRound to free the removed DataSegment
    fn restartUncontNewRound(self: *Fittable) !void {
        // Step 1: Delete segment B and free its memory
        if (self.real_boundary.items.len > 0) {
            const B = self.real_boundary.orderedRemove(0);
            self.allocator.destroy(B); // Free the allocated DataSegment
        }

        // Step 2: Reset light window to segment C
        if (self.real_boundary.items.len > 0) {
            const C = self.real_boundary.items[0];
            DataSegmentExt.copy(&self.light_window, C);

            // Step 3: Reset time base
            self.time_base = C.upper.time;

            // Step 4: Reinitialize convex polygon and arcs
            if (self.real_boundary.items.len >= 2) {
                const D = self.real_boundary.items[1];
                try self.conv.reinitFromTwoSegments(C, D, self.time_base);

                // Reset arcs
                if (self.conv.returnEndmost(RightMost)) |right_pt| {
                    self.ceil_arc.reset(right_pt, D.upper, self.time_base);
                }
                if (self.conv.returnEndmost(LeftMost)) |left_pt| {
                    self.flor_arc.reset(left_pt, D.lower, self.time_base);
                }
            }
        }
    }

    // Restart for connected new round.
    // Fix 5: Add cleanup for bias_lseg when it's no longer needed.
    fn restartContNewRound(self: *Fittable) !void {
        if (self.bias_lseg) |lseg| {
            if (self.bias_chain) |chain| {
                try self.restartNewRound(lseg, chain);
            }
            // Free the allocated DataSegment.
            self.allocator.destroy(lseg);
        }

        // Clean up bias structures
        self.bias_lseg = null;
        self.bias_chain = null;
    }

    // Generic restart with left segment and chain
    fn restartNewRound(self: *Fittable, lseg: *const DataSegment, chain: *Arc) !void {
        // This would implement the continuous restart logic
        // For now, simplified version
        _ = lseg;
        _ = chain;
        try self.restartUncontNewRound();
    }

    // Clone from another fittable
    pub fn cloneFittable(self: *Fittable, other: *const Fittable) !void {
        // Copy fitting window
        self.fw = other.fw;

        // Clear bias structures - free bias_lseg if it exists
        if (self.bias_lseg) |lseg| {
            self.allocator.destroy(lseg);
        }
        self.bias_chain = null;
        self.bias_lseg = null;

        // Copy other basic fields
        self.knot_type = other.knot_type;
        self.time_base = other.time_base;
        self.current_time = other.current_time;
        self.beg_point = other.beg_point;
        self.end_point = other.end_point;

        // Clone real boundary - first clear existing segments
        for (self.real_boundary.items) |segment| {
            self.allocator.destroy(segment);
        }
        self.real_boundary.clearRetainingCapacity();

        // Clone each segment from other
        for (other.real_boundary.items) |segment| {
            const new_segment = try self.allocator.create(DataSegment);
            new_segment.* = segment.*;
            try self.real_boundary.append(new_segment);
        }

        // Clone light window
        self.light_window = other.light_window;
    }

    // Close fitting when data stream ends
    pub fn closeFitting(self: *Fittable) void {
        // Get the last data segment
        if (self.real_boundary.items.len > 0) {
            const last_segment = self.real_boundary.items[self.real_boundary.items.len - 1];

            // Get solution line from parameter space
            if (self.conv.selectSolution(
                @as(f64, @floatFromInt(self.time_base)),
                last_segment.getValue(),
            )) |sol_line| { // Compute end point
                const lastT = @as(f64, @floatFromInt(self.current_time)) + 0.0001; // timeStep equivalent
                self.computeEndPoint(sol_line, lastT);

                // Update fitting window
                self.fw.assign(lastT, lastT);
            }
        }
    }

    // Update with the last data point (similar to updateImmediately but for final point)
    pub fn updateLast(self: *Fittable) !bool {
        if (self.real_boundary.items.len == 0) {
            return true;
        }

        // Get the last segment C
        const C = self.real_boundary.items[self.real_boundary.items.len - 1];

        // Step 1.1: Update with upper point constraint
        const upper_halfplane = HalfPlane.fromDiscretePoint(@as(f64, @floatFromInt(
            self.time_base,
        )) - @as(f64, @floatFromInt(C.upper.time)), C.upper.value, true // is_upper
        );

        const rst_uh = try self.conv.intersect(&upper_halfplane);

        if (rst_uh == .ContainSomeChain) {
            // Reset upper arc since light changed
            if (self.conv.returnEndmost(RightMost)) |right_pt| {
                self.ceil_arc.reset(right_pt, C.upper, self.time_base);
            }
        }

        // Step 1.2: Update with lower point constraint
        const lower_halfplane = HalfPlane.fromDiscretePoint(@as(f64, @floatFromInt(
            self.time_base,
        )) - @as(f64, @floatFromInt(C.lower.time)), C.lower.value, false // is_lower
        );

        const rst_lh = try self.conv.intersect(&lower_halfplane);

        if (rst_lh == .ContainSomeChain) {
            // Reset lower arc since light changed
            if (self.conv.returnEndmost(LeftMost)) |left_pt| {
                self.flor_arc.reset(left_pt, C.lower, self.time_base);
            }
        }

        // Step 2: Check if we need to restart or can continue
        if (rst_uh == .ContainNoneChain or rst_lh == .ContainNoneChain) {
            // Same logic as updateImmediately for closing
            if (rst_uh == .ContainNoneChain) {
                self.closed_direction = .UpperChain;
                self.ceil_arc.clear();
                self.bias_chain = &self.flor_arc;
            } else {
                self.closed_direction = .LowerChain;
                self.flor_arc.clear();
                self.bias_chain = &self.ceil_arc;
            }

            // Compute end points using extreme light
            if (self.bias_chain.?.returnExLight()) |exl| {
                self.computeEndPoint(exl, @as(f64, @floatFromInt(C.upper.time)));
            }

            // Compute new light window and bias segment
            // Clean up any existing bias_lseg before assigning new one
            if (self.bias_lseg) |old_lseg| {
                self.allocator.destroy(old_lseg);
            }
            self.bias_lseg = try self.computeWindowLseg(C, self.closed_direction);

            // Prepare fitting window information
            if (self.closed_direction == .UpperChain) {
                self.fw.tu = @as(f64, @floatFromInt(C.upper.time));
                self.fw.tg = @as(f64, @floatFromInt(self.light_window.lower.time));
            } else {
                self.fw.tu = @as(f64, @floatFromInt(self.light_window.upper.time));
                self.fw.tg = @as(f64, @floatFromInt(C.lower.time));
            }

            return false; // Closed
        } else {
            // Update arcs if needed
            if (rst_uh == .ContainAllChain) {
                self.ceil_arc.update(C.upper);
            }
            if (rst_lh == .ContainAllChain) {
                self.flor_arc.update(C.lower);
            }

            return true; // Continue
        }
    }

    // Check if fitting is closed
    pub fn isClosed(self: *const Fittable) bool {
        return self.fw.tu >= 0 and self.fw.tg >= 0;
    }

    // Get the size of internal structures
    pub fn getSize(self: *const Fittable) usize {
        return self.real_boundary.items.len * 3 + // Each segment uses ~3 parameters
            self.conv.getSize() + // Convex polygon size
            6; // Fixed overhead for arcs and other structures
    }
};

// Helper extension for DataSegment
const DataSegmentExt = struct {
    pub fn intersectWithLine(self: DataSegment, line: shared.LinearFunction) shared.DiscretePoint {
        // Simplified intersection - find where line intersects the vertical segment
        const time_f64 = @as(f64, @floatFromInt(self.upper.time));
        const line_value = line.slope * time_f64 + line.intercept;

        // Clamp to segment bounds
        const clamped_value = @max(self.lower.value, @min(self.upper.value, line_value));

        return shared.DiscretePoint{ .time = self.upper.time, .value = clamped_value };
    }
    pub fn copy(dest: *DataSegment, src: *const DataSegment) void {
        dest.upper = src.upper;
        dest.lower = src.lower;
    }
};

// Represents a knot in the DP solution (C[k] in the paper)
const Ck = struct {
    k: usize,
    knot_type: bool, // true = connected, false = disconnected
    ref_count: i32,
    prev: ?*Ck,
    last_knot: shared.DiscretePoint,
    end_point: shared.DiscretePoint,
    fw: FittingWindow,
    allocator: mem.Allocator,

    pub fn init(allocator: mem.Allocator, k: usize, knot_type: bool, prev: ?*Ck) !*Ck {
        const ck = try allocator.create(Ck);
        ck.* = .{
            .k = k,
            .knot_type = knot_type,
            .ref_count = 1,
            .prev = prev,
            .last_knot = shared.DiscretePoint{ .time = 0, .value = 0.0 },
            .end_point = shared.DiscretePoint{ .time = 0, .value = 0.0 },
            .fw = FittingWindow.init(),
            .allocator = allocator,
        };

        if (prev) |p| {
            p.incRef();
        }

        return ck;
    }

    pub fn deinit(self: *Ck) void {
        if (self.prev) |p| {
            p.decRef();
        }
        self.allocator.destroy(self);
    }

    pub fn incRef(self: *Ck) void {
        self.ref_count += 1;
    }

    pub fn decRef(self: *Ck) void {
        self.ref_count -= 1;
    }
};

// Main mixed-type PLA algorithm
pub const MixedContApr = struct {
    // Dynamic programming state
    k: i32, // Current index (-1 initially)
    ck_list: ArrayList(*Ck), // List of Ck nodes
    C: [3]?*Ck, // C[k], C[k+1], C[k+2]

    // Five parallel fitting instances
    base: [5]*Fittable,
    fg: [5]bool, // Flags indicating if each base can continue

    // Solution tracking
    opt: usize, // Index of optimal base
    knot_flags: ArrayList(bool), // Joint vs disjoint knots
    segments: ArrayList(shared.DiscretePoint), // Output segments
    current_time: f64,
    delay_info: i32,

    allocator: mem.Allocator,
    epsilon: f64,

    pub fn init(allocator: mem.Allocator, epsilon: f64) !MixedContApr {
        var self = MixedContApr{
            .k = -1,
            .ck_list = ArrayList(*Ck).init(allocator),
            .C = .{ null, null, null },
            .base = undefined,
            .fg = .{ true, true, true, true, true },
            .opt = 0,
            .knot_flags = ArrayList(bool).init(allocator),
            .segments = ArrayList(shared.DiscretePoint).init(allocator),
            .current_time = 0,
            .delay_info = 0,
            .allocator = allocator,
            .epsilon = epsilon,
        };

        // Initialize the 5 bases:
        // base[0] = C[k] + disjoint
        // base[1] = C[k+1] + connected
        // base[2] = C[k+1] + disjoint
        // base[3] = C[k+2] + connected
        // base[4] = C[k+2] + disjoint
        self.base[0] = try self.allocator.create(Fittable);
        self.base[0].* = try Fittable.init(allocator, epsilon, false);

        self.base[1] = try self.allocator.create(Fittable);
        self.base[1].* = try Fittable.init(allocator, epsilon, true);

        self.base[2] = try self.allocator.create(Fittable);
        self.base[2].* = try Fittable.init(allocator, epsilon, false);

        self.base[3] = try self.allocator.create(Fittable);
        self.base[3].* = try Fittable.init(allocator, epsilon, true);

        self.base[4] = try self.allocator.create(Fittable);
        self.base[4].* = try Fittable.init(allocator, epsilon, false);

        return self;
    }

    pub fn deinit(self: *MixedContApr) void {
        // Clean up Ck nodes
        for (self.ck_list.items) |ck| {
            ck.deinit();
        }
        self.ck_list.deinit();

        // Clean up C array
        for (self.C) |maybe_ck| {
            if (maybe_ck) |ck| {
                ck.deinit();
            }
        }

        // Clean up bases
        for (self.base) |base| {
            base.deinit();
            self.allocator.destroy(base);
        }

        self.knot_flags.deinit();
        self.segments.deinit();
    }

    pub fn update(self: *MixedContApr, point: shared.DiscretePoint) !bool {
        // Update all active bases
        for (self.base, 0..) |base, i| {
            if (self.fg[i]) {
                self.fg[i] = try base.update(point);
            }
        }

        self.current_time = @as(f64, @floatFromInt(point.time));

        // When both base[0] and base[1] close, process DP transition
        while (!self.fg[0] and !self.fg[1]) {
            try self.dpToBases();
        }

        return true;
    }

    fn dpToBases(self: *MixedContApr) !void {
        // Step 1: Determine winner between base[0] and base[1]
        self.opt = self.isBetter(0, 1);

        // Step 1.1: Create C[k+3]
        const ck3 = try self.createCk3(self.opt);

        // Step 1.2: Try to erase unused Cks
        _ = try self.pushCkAndTryToErase(self.C[0]);

        // Step 2: Prepare new bases
        var ck3c: *Fittable = undefined;
        var ck3d: *Fittable = undefined;

        if (ck3 != null) {
            // Copy the worse to the better (for disjoint start)
            try self.base[1 - self.opt].cloneFittable(self.base[self.opt]);

            ck3c = self.base[self.opt];
            ck3d = self.base[1 - self.opt];

            // Restart new round for C[k+3]
            try ck3c.initNewRoundWithType(.Connected);
            try ck3d.initNewRoundWithType(.Disjoint);
        } else {
            ck3c = self.base[self.opt];
            ck3d = self.base[1 - self.opt];
            ck3c.fw = FittingWindow{ .tu = -1, .tg = -1 };
            ck3d.fw = FittingWindow{ .tu = -1, .tg = -1 };
        }

        // Step 3: Shift arrays and increment k
        self.k += 1;

        // Step 3.1: Move flags
        self.fg[0] = self.fg[2];
        self.fg[1] = self.fg[3];
        self.fg[2] = self.fg[4];
        if (ck3 != null) {
            self.fg[3] = true;
            self.fg[4] = true;
        } else {
            self.fg[3] = false;
            self.fg[4] = false;
        }

        // Step 3.2: Move bases
        self.base[0] = self.base[2];
        self.base[1] = self.base[3];
        self.base[2] = self.base[4];
        self.base[3] = ck3c;
        self.base[4] = ck3d;

        // Step 3.3: Move C[k]
        self.C[0] = self.C[1];
        self.C[1] = self.C[2];
        self.C[2] = ck3;

        // Step 4: Try to output fixed pieces
        try self.detectAndOutputFixedPieces();
    }

    fn createCk3(self: *MixedContApr, win: usize) !?*Ck {
        var f2 = FittingWindow.init();
        const f3 = self.base[win].fw;

        if (self.C[2]) |c2| {
            f2 = c2.fw;
        }

        if (win > 1 or win > self.base.len) {
            return null;
        } else if (f2.tg >= f3.tg and f2.tu >= f3.tu) {
            // No need to construct Ck3
            return null;
        } else {
            const ck3 = if (win == 0)
                try Ck.init(self.allocator, @as(usize, @intCast(self.k + 3)), false, self.C[0])
            else
                try Ck.init(self.allocator, @as(usize, @intCast(self.k + 3)), true, self.C[1]);

            ck3.last_knot = self.base[win].beg_point;
            ck3.end_point = self.base[win].end_point;
            ck3.fw = self.base[win].fw;

            return ck3;
        }
    }

    fn isBetter(self: *MixedContApr, i: usize, j: usize) usize {
        if (!self.fg[i] and self.fg[j]) {
            return j;
        } else if (self.fg[i] and !self.fg[j]) {
            return i;
        } else if (!self.fg[i] and !self.fg[j]) {
            const fw_i = self.base[i].fw;
            const fw_j = self.base[j].fw;

            if (fw_i.tu > fw_j.tu) {
                return i;
            } else if (fw_i.tu < fw_j.tu) {
                return j;
            } else {
                return if (fw_i.tg > fw_j.tg) i else j;
            }
        } else {
            return 1;
        }
    }

    fn pushCkAndTryToErase(self: *MixedContApr, ck_opt: ?*Ck) !i32 {
        var ck: *Ck = ck_opt orelse return -1;

        // Push to list
        try self.ck_list.append(ck);

        // Try to erase if no future
        var index = self.ck_list.items.len - 1;
        while (true) {
            ck.decRef();

            if (ck.ref_count < 0) {
                return @as(i32, @intCast(ck.k));
            } else if (ck.ref_count > 0) {
                return ck.ref_count;
            }

            // Can erase current ck
            const prev = ck.prev;
            ck.deinit();
            _ = self.ck_list.orderedRemove(index);

            if (prev) |p| {
                ck = p;
            } else {
                break;
            }

            // Find ck in list
            var found = false;
            for (self.ck_list.items, 0..) |item, i| {
                if (item == ck) {
                    index = i;
                    found = true;
                    break;
                }
            }
            if (!found) break;
        }

        return -1;
    }

    fn detectAndOutputFixedPieces(self: *MixedContApr) !void {
        while (self.ck_list.items.len > 0) {
            const fir = self.ck_list.items[0];
            if (fir.ref_count != 1) return;

            var sec: *Ck = undefined;
            if (self.ck_list.items.len > 1) {
                sec = self.ck_list.items[1];
            } else {
                if (self.C[0]) |c0| {
                    sec = c0;
                } else if (self.C[1]) |c1| {
                    sec = c1;
                } else if (self.C[2]) |c2| {
                    sec = c2;
                } else {
                    return;
                }
            }

            if (sec.prev == fir) {
                // Output this piece
                try self.knot_flags.append(fir.knot_type);
                try self.segments.append(fir.last_knot);

                if (!sec.knot_type) {
                    try self.segments.append(fir.end_point);
                }

                self.delay_info += @as(i32, @intFromFloat(self.current_time - @as(f64, @floatFromInt(fir.last_knot.time))));

                // Delete and continue
                fir.deinit();
                _ = self.ck_list.orderedRemove(0);
                sec.prev = null;
            } else {
                return;
            }
        }
    }

    pub fn closeFitting(self: *MixedContApr) !void {
        // Update last points
        for (0..5) |i| {
            if (self.fg[i]) {
                self.fg[i] = try self.base[i].updateLast();
            }
        }

        // Process remaining DP transitions
        while (!self.fg[0] and !self.fg[1]) {
            try self.dpToBases();
        }

        // Finalize
        self.opt = self.isBetter(0, 1);
        self.base[self.opt].closeFitting();

        const ck3 = try self.createCk3(self.opt);
        _ = try self.pushCkAndTryToErase(self.C[0]);
        try self.detectAndOutputFixedPieces();

        // Shift final state
        self.k += 1;
        self.C[0] = self.C[1];
        self.C[1] = self.C[2];
        self.C[2] = ck3;

        if (self.C[2] == null) {
            self.C[2] = self.C[1];
        }

        try self.outputList(self.C[2]);
        self.clearData();

        // Compute average delay
        if (self.knot_flags.items.len > 0) {
            self.delay_info = @divTrunc(self.delay_info, @as(i32, @intCast(self.knot_flags.items.len)));
        }
    }

    fn outputList(self: *MixedContApr, ck_opt: ?*Ck) !void {
        var ck: *Ck = ck_opt orelse return;

        var flags = ArrayList(bool).init(self.allocator);
        defer flags.deinit();
        var points = ArrayList(shared.DiscretePoint).init(self.allocator);
        defer points.deinit();

        while (true) {
            ck.decRef();

            if (flags.items.len == 0 or !flags.items[flags.items.len - 1]) {
                try points.append(ck.end_point);
            }

            try points.append(ck.last_knot);
            try flags.append(ck.knot_type);

            self.delay_info += @as(i32, @intFromFloat(self.current_time - @as(f64, @floatFromInt(ck.last_knot.time))));

            if (ck.prev) |p| {
                ck = p;
            } else {
                break;
            }
        }

        // Reverse and append to main lists
        while (flags.items.len > 0) {
            if (flags.pop()) |flag| {
                try self.knot_flags.append(flag);
            }
        }
        while (points.items.len > 0) {
            if (points.pop()) |point| {
                try self.segments.append(point);
            }
        }
    }

    fn clearData(self: *MixedContApr) void {
        for (self.C) |maybe_ck| {
            if (maybe_ck) |ck| {
                ck.deinit();
            }
        }
        self.C = .{ null, null, null };

        while (self.ck_list.items.len > 0) {
            if (self.ck_list.pop()) |ck| {
                ck.deinit();
            }
        }
    }
};
