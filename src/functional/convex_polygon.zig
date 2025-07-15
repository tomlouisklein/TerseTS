const std = @import("std");
const shared = @import("../utilities/shared_structs.zig");
const mixedPLA = @import("mixed_PLA_cpp_version.zig");
const ConvexList = mixedPLA.ConvexList;
const ParameterPoint = mixedPLA.ParameterPoint;
const DataSegment = mixedPLA.DataSegment;
const HalfPlane = mixedPLA.HalfPlane;
const IntersectionResult = mixedPLA.IntersectionResult;
const ClosureDirection = mixedPLA.ClosureDirection;

// Main convex polygon structure
pub const ConvexPolygon = struct {
    upper_edges: ConvexList,
    lower_edges: ConvexList,
    instantiated: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !ConvexPolygon {
        return .{
            .upper_edges = try ConvexList.init(allocator, true),
            .lower_edges = try ConvexList.init(allocator, false),
            .instantiated = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ConvexPolygon) void {
        self.upper_edges.deinit();
        self.lower_edges.deinit();
    }

    // Initialize with two data points to create initial quadrilateral in parameter space
    pub fn initWithPoints(self: *ConvexPolygon, p1: shared.DiscretePoint, p2: shared.DiscretePoint, epsilon: f64) !void {
        // Create constraint lines for each point
        // For point (x, y) with error ε, we get constraints:
        // y ≤ y + ε at x → intercept ≥ (y + ε) - slope * x
        // y ≥ y - ε at x → intercept ≤ (y - ε) - slope * x

        const p1_time = @as(f64, @floatFromInt(p1.time));
        const p2_time = @as(f64, @floatFromInt(p2.time));

        // In parameter space, for point (t, v) with error ε:
        // Upper constraint: intercept ≤ (v + ε) - slope * t
        // Lower constraint: intercept ≥ (v - ε) - slope * t

        // Create the four constraint lines
        const p1_upper_line = shared.LinearFunction{ .slope = -p1_time, .intercept = p1.value + epsilon };
        const p1_lower_line = shared.LinearFunction{ .slope = -p1_time, .intercept = p1.value - epsilon };
        const p2_upper_line = shared.LinearFunction{ .slope = -p2_time, .intercept = p2.value + epsilon };
        const p2_lower_line = shared.LinearFunction{ .slope = -p2_time, .intercept = p2.value - epsilon };

        // Compute the four intersection points (corners of feasible region)
        const left_most = computeLineIntersection(p1_upper_line, p2_lower_line); // pu1 ∩ pl2
        const right_most = computeLineIntersection(p1_lower_line, p2_upper_line); // pl1 ∩ pu2
        const middle_top = computeLineIntersection(p1_upper_line, p2_upper_line); // pu1 ∩ pu2
        const middle_bottom = computeLineIntersection(p1_lower_line, p2_lower_line); // pl1 ∩ pl2

        // Initialize chains using the correct pattern from C++ code:
        // Upper chain: left_most -> middle_top -> right_most (edges: lm, mT; endpoint: rm)
        try self.upper_edges.initConvexChain(left_most, middle_top, right_most);

        // Lower chain: right_most -> middle_bottom -> left_most (edges: rm, mB; endpoint: lm)
        try self.lower_edges.initConvexChain(right_most, middle_bottom, left_most);

        self.instantiated = true;
    }

    // This function is called during the continuous approximation process when:
    // A fitting attempt closes (can no longer extend the current segment).
    // The algorithm needs to start a new round for continuous fitting.
    // Reinitialize with a data segment and closing direction (for continuous approximation).
    // More elegant reInitWithSegment - focuses on the core geometry
    pub fn reInitWithSegment(self: *ConvexPolygon, left_seg: *const DataSegment, current_seg: *const DataSegment, time_base: usize, closed_direction: ClosureDirection) !void {
        const base_time = @as(f64, @floatFromInt(time_base));

        // Create the four constraint lines from the two segments
        const constraints = [4]shared.LinearFunction{
            // Left segment constraints
            .{ .slope = -(base_time - @as(f64, @floatFromInt(left_seg.upper.time))), .intercept = left_seg.upper.value },
            .{ .slope = -(base_time - @as(f64, @floatFromInt(left_seg.lower.time))), .intercept = left_seg.lower.value },
            // Current segment constraints
            .{ .slope = -(base_time - @as(f64, @floatFromInt(current_seg.upper.time))), .intercept = current_seg.upper.value },
            .{ .slope = -(base_time - @as(f64, @floatFromInt(current_seg.lower.time))), .intercept = current_seg.lower.value },
        };

        // Compute the four potential corner points of the feasible region
        const corners = [4]ParameterPoint{
            computeLineIntersection(constraints[0], constraints[3]), // left_upper ∩ current_lower
            computeLineIntersection(constraints[0], constraints[2]), // left_upper ∩ current_upper
            computeLineIntersection(constraints[1], constraints[2]), // left_lower ∩ current_upper
            computeLineIntersection(constraints[1], constraints[3]), // left_lower ∩ current_lower
        };

        // Check if we have a valid convex quadrilateral
        const feasible_region = validateAndFixRegion(corners, closed_direction);

        // Initialize the convex chains from the feasible region
        try self.initializeFromRegion(feasible_region);
        self.instantiated = true;
    }

    // Reinitialize with two data segments (for restart after closure)
    pub fn reinitFromTwoSegments(self: *ConvexPolygon, seg1: *const DataSegment, seg2: *const DataSegment, time_base: usize) !void {
        // Convert segments to points for initialization
        const p1 = shared.DiscretePoint{ .time = seg1.upper.time, .value = (seg1.upper.value + seg1.lower.value) / 2.0 };
        const p2 = shared.DiscretePoint{ .time = seg2.upper.time, .value = (seg2.upper.value + seg2.lower.value) / 2.0 };

        // Use the existing initWithPoints method
        const epsilon = (seg1.upper.value - seg1.lower.value) / 2.0;
        try self.initWithPoints(p1, p2, epsilon);

        _ = time_base; // Time base is handled in initWithPoints
    }
    // Helper struct for the feasible region
    const FeasibleRegion = struct {
        left_bottom: ParameterPoint,
        left_top: ParameterPoint,
        right_top: ParameterPoint,
        right_bottom: ParameterPoint,
    };

    // Validate the computed region and fix degenerate cases
    pub fn validateAndFixRegion(corners: [4]ParameterPoint, closed_direction: ClosureDirection) FeasibleRegion {
        var region = FeasibleRegion{
            .left_bottom = corners[0], // left_upper ∩ current_lower
            .left_top = corners[1], // left_upper ∩ current_upper
            .right_top = corners[2], // left_lower ∩ current_upper
            .right_bottom = corners[3], // left_lower ∩ current_lower
        };

        // Check for degenerate cases by examining the geometry
        const is_degenerate = isRegionDegenerate(region);

        if (is_degenerate) {
            // Resolve degeneracy based on which chain caused the closure
            region = resolveDegenerate(region, closed_direction);
        }

        return region;
    }

    // Check if the region is degenerate (invalid convex quadrilateral)
    fn isRegionDegenerate(region: FeasibleRegion) bool {
        // A region is degenerate if:
        // 1. Left and right sides cross over (left_top.slope >= right_top.slope)
        // 2. Top and bottom sides cross over (left_bottom.slope > right_bottom.slope)
        // 3. The "quadrilateral" is inside-out

        const left_right_crossed = region.left_top.slope >= region.right_top.slope;
        const top_bottom_crossed = region.left_bottom.slope > region.right_bottom.slope;

        return left_right_crossed or top_bottom_crossed;
    }

    // Resolve degenerate cases using the closure direction as guidance
    fn resolveDegenerate(region: FeasibleRegion, closed_direction: ClosureDirection) FeasibleRegion {
        var fixed_region = region;

        switch (closed_direction) {
            .UpperChain => {
                // Upper chain caused closure, so bias towards left segment constraints
                // Collapse the right side to match the left side
                fixed_region.right_top = region.left_top;
                fixed_region.right_bottom = region.left_bottom;
            },
            .LowerChain => {
                // Lower chain caused closure, so bias towards current segment constraints
                // Collapse the left side to match the right side
                fixed_region.left_top = region.right_top;
                fixed_region.left_bottom = region.right_bottom;
            },
        }

        return fixed_region;
    }

    // Initialize the convex chains from the validated feasible region
    pub fn initializeFromRegion(self: *ConvexPolygon, region: FeasibleRegion) !void {
        // Upper chain: left_bottom -> left_top -> right_top
        try self.upper_edges.initConvexChain(region.left_bottom, region.left_top, region.right_top);

        // Lower chain: right_top -> right_bottom -> left_bottom
        try self.lower_edges.initConvexChain(region.right_top, region.right_bottom, region.left_bottom);
    }

    // Intersect with a half-plane
    pub fn intersect(self: *ConvexPolygon, hp: *const HalfPlane) !IntersectionResult {
        if (!self.instantiated) return .NonExistChain;

        var relationship: IntersectionResult = .ContainAllChain;

        if (hp.direction == .PointToBelow) {
            // Upper bound constraint
            relationship = try self.upper_edges.pad(hp);
            const end_point = self.lower_edges.cut(hp);
            if (end_point) |ep| {
                self.upper_edges.end_most = ep;
            }
        } else {
            // Lower bound constraint
            relationship = try self.lower_edges.pad(hp);
            const end_point = self.upper_edges.cut(hp);
            if (end_point) |ep| {
                self.lower_edges.end_most = ep;
            }
        }

        return relationship;
    }

    // Get the best line (solution) from the polygon
    pub fn selectSolution(self: *ConvexPolygon, time_shift: f64, fallback_value: f64) ?shared.LinearFunction {
        if (!self.instantiated) {
            // If not instantiated, return a horizontal line
            return shared.LinearFunction{ .slope = 0, .intercept = fallback_value };
        }

        // Choose between the endpoints of upper and lower chains
        // The "best" solution is typically the one furthest from the origin in parameter space
        const upper_end = self.upper_edges.end_most;
        const lower_end = self.lower_edges.end_most;

        var chosen_point: ParameterPoint = undefined;

        if (upper_end != null and lower_end != null) {
            // Choose the point with larger distance from origin
            const upper_dist = upper_end.?.slope * upper_end.?.slope + upper_end.?.intercept * upper_end.?.intercept;
            const lower_dist = lower_end.?.slope * lower_end.?.slope + lower_end.?.intercept * lower_end.?.intercept;

            chosen_point = if (@abs(upper_dist) > @abs(lower_dist)) upper_end.? else lower_end.?;
        } else if (upper_end != null) {
            chosen_point = upper_end.?;
        } else if (lower_end != null) {
            chosen_point = lower_end.?;
        } else {
            return null;
        }

        // Transform the parameter point back to a linear function
        // Apply time shift: new_intercept = old_intercept - slope * time_shift
        return shared.LinearFunction{
            .slope = chosen_point.slope,
            .intercept = chosen_point.intercept - chosen_point.slope * time_shift,
        };
    }

    // Check if the polygon is instantiated
    pub fn isInstantiated(self: *const ConvexPolygon) bool {
        return self.instantiated;
    }

    // Get the size (number of vertices) of the polygon
    pub fn getSize(self: *const ConvexPolygon) usize {
        if (!self.instantiated) return 0;
        return self.upper_edges.getSize() + self.lower_edges.getSize();
    }

    // Set the polygon as non-instantiated (empty)
    pub fn setEmpty(self: *ConvexPolygon) void {
        self.instantiated = false;
        self.upper_edges.clear();
        self.lower_edges.clear();
    }

    // Reset the end points of both chains
    pub fn resetEndPoints(self: *ConvexPolygon, upper_end: ?ParameterPoint, lower_end: ?ParameterPoint) void {
        self.upper_edges.end_most = upper_end;
        self.lower_edges.end_most = lower_end;
    }

    // Get the endpoint of a specific chain
    pub fn getEndPoint(self: *const ConvexPolygon, is_upper: bool) ?ParameterPoint {
        return if (is_upper) self.upper_edges.end_most else self.lower_edges.end_most;
    }

    // Get the endpoint of a specific chain (matches C++ returnEndmost interface)
    pub fn returnEndmost(self: *const ConvexPolygon, right_or_left: bool) ?ParameterPoint {
        // true (RightMost) -> upper_edges, false (LeftMost) -> lower_edges
        return if (right_or_left) self.upper_edges.end_most else self.lower_edges.end_most;
    }

    // Helper function to compute intersection of two lines in parameter space
    pub fn computeLineIntersection(line1: shared.LinearFunction, line2: shared.LinearFunction) ParameterPoint {
        // Solve: line1.slope * x + line1.intercept = line2.slope * x + line2.intercept
        // (line1.slope - line2.slope) * x = line2.intercept - line1.intercept

        const slope_diff = line1.slope - line2.slope;

        if (@abs(slope_diff) < 1e-10) {
            // Lines are parallel, return a point on one of them
            return ParameterPoint{ .slope = 0, .intercept = line1.intercept };
        }

        const x = (line2.intercept - line1.intercept) / slope_diff;
        const y = line1.slope * x + line1.intercept;

        return ParameterPoint{ .slope = x, .intercept = y };
    }
};
